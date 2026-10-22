"""
menu_alias.py
=============
菜名「同菜異寫」的等價層。

問題背景
--------
日文同一道菜可以用漢字或假名描述（唐揚げ / から揚げ / から揚），而
menus_mapping.yaml 是客戶 xlsx 的投影，只會收錄其中一種寫法。不同店家的
貼紙印表機用哪一種寫法我們無法事先知道，所以：

    貼紙印 から揚げ丼並、yaml 只有 唐揚げ丼並
        -> 舊流程最佳匹配是 'から揚げ'(~80 分) -> 不在待核對清單 -> 誤發 WRONG_ITEM

解法是把比對放到「折疊後的座標系」做，而不是在最後多加一道確認。

三層知識（刻意分開，因為生命週期與擁有者不同）
----------------------------------------------
1. readings     通用日文讀音（唐 -> から）。跨客戶共用，放 tasks/ocr/ja_readings.yaml。
2. equivalents  某客戶的命名慣例（牛丼特 = 牛丼特盛）。人工判定，放各 task 的
                menu_aliases.yaml —— 不可放進 menus_mapping.yaml，那份會被 xlsx 整份重產。
3. distinct     人工判定「不是同一道菜」的配對。記錄「否」跟記錄「是」一樣重要，
                否則每次菜單改版都要重新問一次同樣的問題。

重要界線
--------
fold() 產生的字串**只用於比對**，絕對不可以拿去顯示、發 MQTT 或寫 CSV。
對外一律用 MenuIndex.canonical() 回來的「真實菜名」。

風險不對稱
----------
錯誤合併（把兩道菜併成一道）會讓錯餐點被靜默判為 MATCHED，比漏合併
（顯示未確認，作業員看得到）嚴重得多。因此本模組對「數量軸」與「份量軸」
設有硬性護欄：即使折疊後字串相同，只要數字或份量詞不一致就拒絕合併，
並把衝突記在 conflicts 裡供 tools/verify_menu_alias.py 報告。
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from loguru import logger

from logic.text_utils import normalize_text

# 片假名 -> 平假名（ｶﾞ 之類半角已由 NFKC 轉成全角，這裡只處理全角片假名）
_KATA2HIRA = {chr(c): chr(c - 0x60) for c in range(0x30A1, 0x30F7)}
_KATA2HIRA_TABLE = str.maketrans(_KATA2HIRA)

# 折疊時視為雜訊的符號。菜名裡的 ･ ・ 全角空白 波浪號等在不同 POS/貼紙上
# 出現與否並不代表不同餐點（實測 460 個 sub_items 中有 14 個只差這些符號）。
_NOISE_SYMBOLS = re.compile(r"[\s　・/／\\\-−–—_、，,。．.]")

# 份量軸：這些詞不同就一定是不同的餐點，永遠不准折疊在一起。
# 以「長詞優先」排序，掃描時才不會讓 特 先吃掉 特盛。
# 注意：頭 / あたま（アタマ）是同一個份量，靠 readings 折成同一個 token，
# 所以這裡只列折疊後的形式。
PORTION_TOKENS: Tuple[str, ...] = (
    "超特盛", "並盛", "大盛", "小盛", "特盛", "中盛",
    "超特", "あたま", "みに",
    "並", "大", "小", "特", "頭",
)

_DIGIT_RE = re.compile(r"\d")


def _apply_readings(text: str, readings: Sequence[Tuple[str, str]]) -> str:
    """套用讀音/送假名規則。

    readings 由呼叫端保證已依「來源字串長度遞減」排序，單趟取代即可：
    長規則先吃掉（焼き -> 焼），短規則再處理（焼 -> やき），組合起來
    焼き鳥 / 焼鳥 / やきとり 會收斂到同一個 key。
    """
    for src, dst in readings:
        if src in text:
            text = text.replace(src, dst)
    return text


def fold_menu_name(text: str, readings: Sequence[Tuple[str, str]] = ()) -> str:
    """把菜名折疊成「只用於比對」的 key。

    NFKC -> 去符號 -> 套讀音規則 -> 片假名轉平假名。

    這個回傳值不是給人看的，也不可以外流到 MQTT / CSV / 畫面。
    """
    s = normalize_text(text, ignore_space=True)
    s = _NOISE_SYMBOLS.sub("", s)
    s = _apply_readings(s, readings)
    return s.translate(_KATA2HIRA_TABLE)


def guard_signature(folded: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """取出折疊後字串的「數量軸 + 份量軸」指紋。

    兩個字串只有在指紋完全相同時才允許被視為同一道菜。這道護欄擋的是
    から揚げ３個 / ６個 / ９個、牛丼並 / 大 / 特 這種「差一個字但絕對
    不是同一道菜」的組合 —— 它們在任何相似度演算法下都長得幾乎一樣。
    """
    digits = tuple(sorted(_DIGIT_RE.findall(folded)))

    portions: List[str] = []
    i = 0
    while i < len(folded):
        for token in PORTION_TOKENS:
            if folded.startswith(token, i):
                portions.append(token)
                i += len(token)
                break
        else:
            i += 1
    return digits, tuple(portions)


class MenuIndex:
    """一個菜名池（傳票側或貼紙側）的等價索引。

    對外只提供三件事：
        canonical_names  比對時要用的候選清單（都是真實菜名，不是 fold key）
        canonical(text)  任何寫法 -> 該群的代表菜名（沒見過的寫法也能靠 fold key 命中）
        fold(text)       折疊 key，給相似度計算當座標系用

    conflicts 收集「護欄擋下來的合併」與「distinct 宣告造成的拆分」，
    供驗證腳本輸出報表；正常情況應該是空的。
    """

    def __init__(
        self,
        readings: Sequence[Tuple[str, str]] = (),
        equivalents: Sequence[Sequence[str]] = (),
        distinct: Sequence[Sequence[str]] = (),
        pool_name: str = "",
    ):
        # 長規則先套用，短規則後套用（見 _apply_readings 的說明）
        self.readings: List[Tuple[str, str]] = sorted(
            ((str(a), str(b)) for a, b in readings),
            key=lambda pair: -len(pair[0]),
        )
        self.equivalents = [list(g) for g in equivalents if len(g) > 1]
        self.distinct_pairs = {
            frozenset((normalize_text(a, ignore_space=True), normalize_text(b, ignore_space=True)))
            for g in distinct
            for a, b in _pairs(g)
        }
        self.pool_name = pool_name

        self._rep_by_name: Dict[str, str] = {}   # 正規化菜名 -> 代表菜名
        self._rep_by_fold: Dict[str, str] = {}   # fold key   -> 代表菜名
        self.canonical_names: List[str] = []
        self.conflicts: List[str] = []

    # ------------------------------------------------------------------
    def fold(self, text: str) -> str:
        return fold_menu_name(text, self.readings)

    def build(
        self,
        names: Iterable[str],
        prefer: Iterable[str] = (),
        display_names: Optional[Dict[str, str]] = None,
    ) -> "MenuIndex":
        """用一個菜名池建立索引。

        Args:
            names: 菜名池，可含重複（重複次數參與代表菜名的挑選）。
            prefer: 優先當代表的寫法集合。貼紙側會傳入「傳票側的菜名」——
                作業員手上拿的是傳票，畫面顯示跟傳票同一種寫法才不會對不上；
                這也讓 から揚げ３個 / ６個 / ９個 整個家族的顯示寫法一致，
                不會因為 xlsx 裡各寫法出現次數的微小差異而東倒西歪。
            display_names: 人工指定的顯示名（menu_aliases.yaml 的 display_names），
                key 可以是該群任一種寫法。優先度最高。
        """
        prefer_set = {normalize_text(n, ignore_space=True) for n in prefer}
        display_map = {
            normalize_text(k, ignore_space=True): normalize_text(v, ignore_space=True)
            for k, v in (display_names or {}).items()
        }

        freq: Counter = Counter()
        order: Dict[str, int] = {}
        for raw in names:
            name = normalize_text(raw, ignore_space=True)
            if not name:
                continue
            freq[name] += 1
            order.setdefault(name, len(order))

        # --- 1. 依 (fold key, 護欄指紋) 分群：指紋不同者天生不會被併在一起 ---
        groups: Dict[Tuple[str, tuple, tuple], List[str]] = {}
        folds_seen: Dict[str, set] = {}
        for name in order:
            fold_key = self.fold(name)
            sig = guard_signature(fold_key)
            groups.setdefault((fold_key, sig[0], sig[1]), []).append(name)
            folds_seen.setdefault(fold_key, set()).add(sig)

        # 同一個 fold key 底下出現多種指紋 = 有讀音規則想做危險合併，記下來
        for fold_key, sigs in folds_seen.items():
            if len(sigs) > 1:
                members = sorted(
                    n for n in order if self.fold(n) == fold_key
                )
                self.conflicts.append(
                    f"[護欄擋下] pool={self.pool_name} fold='{fold_key}' "
                    f"數量/份量不一致，維持分開: {members}"
                )

        group_list: List[List[str]] = list(groups.values())

        # --- 2. 套用人工判定的 equivalents（可跨指紋，屬於人為覆寫）---
        self._equiv_merged = 0   # 逐組明細在 DEBUG，INFO 只在下面總結行報組數
        group_list = self._merge_equivalents(group_list, order)

        # --- 3. 套用人工判定的 distinct：被宣告為不同的成員強制拆開 ---
        group_list = self._split_distinct(group_list)

        # --- 4. 決定代表菜名 ---
        # 優先序：人工指定 > 傳票也在用的寫法 > 出現次數 > yaml 較早出現者。
        # 後兩項只是決勝局，重點是前兩項讓顯示名穩定且跟現場一致。
        self._rep_by_name.clear()
        self._rep_by_fold.clear()
        self.canonical_names = []
        for members in group_list:
            forced = next((display_map[n] for n in members if n in display_map), None)
            if forced is not None and forced not in members:
                logger.warning(
                    f"[menu_alias] pool={self.pool_name} display_names 指定的 "
                    f"'{forced}' 不在該群寫法中 {members}，仍照指定顯示"
                )
            rep = forced if forced is not None else max(
                members,
                key=lambda n: (n in prefer_set, freq[n], -order[n]),
            )
            self.canonical_names.append(rep)
            for name in members:
                self._rep_by_name[name] = rep
                # fold key -> 代表名，是「沒收錄過的寫法」也能命中的關鍵
                self._rep_by_fold.setdefault(self.fold(name), rep)

        merged = len(order) - len(self.canonical_names)
        logger.info(
            f"[menu_alias] pool={self.pool_name} 菜名 {len(order)} -> "
            f"候選 {len(self.canonical_names)}（合併 {merged}，其中人工 equivalents {self._equiv_merged} 組；"
            f"衝突 {len(self.conflicts)}）"
        )
        return self

    # ------------------------------------------------------------------
    def _merge_equivalents(self, group_list, order):
        index_of = {}
        for gi, members in enumerate(group_list):
            for name in members:
                index_of[name] = gi

        for group in self.equivalents:
            targets = [normalize_text(n, ignore_space=True) for n in group]
            present = [n for n in targets if n in index_of]
            if len(present) < 2:
                missing = [n for n in targets if n not in index_of]
                if missing:
                    # 只是 debug：equivalents 是「跨兩個池」宣告的，貼紙側的名字
                    # 本來就不會出現在傳票池裡，在這裡當 WARNING 會變成每次啟動
                    # 都印一堆誤報。真正的錯字（兩個池都找不到）由
                    # tools/verify_menu_alias.py 負責抓，那裡才看得到兩個池。
                    logger.debug(
                        f"[menu_alias] pool={self.pool_name} equivalents 有菜名不在本池中，"
                        f"於本池忽略: {missing}"
                    )
                continue
            keep = index_of[present[0]]
            for name in present[1:]:
                gi = index_of[name]
                if gi == keep:
                    continue
                group_list[keep].extend(group_list[gi])
                for moved in group_list[gi]:
                    index_of[moved] = keep
                group_list[gi] = []
            self._equiv_merged += 1
            logger.debug(
                f"[menu_alias] pool={self.pool_name} 人工 equivalents 合併: {group_list[keep]}"
            )
        return [g for g in group_list if g]

    def _split_distinct(self, group_list):
        if not self.distinct_pairs:
            return group_list
        result = []
        for members in group_list:
            if len(members) < 2:
                result.append(members)
                continue
            kept: List[str] = []
            exiled: List[str] = []
            for name in members:
                if any(frozenset((name, other)) in self.distinct_pairs for other in kept):
                    exiled.append(name)
                else:
                    kept.append(name)
            if exiled:
                self.conflicts.append(
                    f"[distinct 拆分] pool={self.pool_name} 依人工判定拆開: "
                    f"{kept} <-> {exiled}"
                )
            result.append(kept)
            result.extend([[n] for n in exiled])
        return result

    # ------------------------------------------------------------------
    def canonical(self, text: str) -> Optional[str]:
        """任何寫法 -> 代表菜名。沒見過的寫法會先試 fold key，失敗才回 None。"""
        if not text:
            return None
        name = normalize_text(text, ignore_space=True)
        rep = self._rep_by_name.get(name)
        if rep is not None:
            return rep
        return self._rep_by_fold.get(self.fold(name))

    def canonical_or_self(self, text: str) -> str:
        """給「一定要有值」的場合用：查不到就原字串回去，不要吞掉資訊。"""
        return self.canonical(text) or normalize_text(text, ignore_space=True)

    def aliases_of(self, rep: str) -> List[str]:
        """某個代表菜名底下收錄過的所有寫法（供除錯與報表用）。"""
        rep_norm = normalize_text(rep, ignore_space=True)
        return sorted(n for n, r in self._rep_by_name.items() if r == rep_norm)


def _pairs(group: Sequence[str]):
    items = list(group)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            yield items[i], items[j]


class AliasKnowledge(NamedTuple):
    """三層知識載入後的結果。用 NamedTuple 而不是 tuple，之後要加欄位不會打壞呼叫端。"""
    readings: List[Tuple[str, str]]
    equivalents: List[List[str]]
    distinct: List[List[str]]
    display_names: Dict[str, str]


def load_alias_knowledge(readings_path=None, aliases_path=None) -> AliasKnowledge:
    """讀取 ja_readings.yaml（共用）與 menu_aliases.yaml（各 task）。

    兩份都是選填：檔案不存在就退回「只做 NFKC + 去符號 + 片假名折疊」，
    行為仍優於改動前，不會因為少一個檔案就啟動失敗。
    """
    import os

    import yaml

    readings: List[Tuple[str, str]] = []
    equivalents: List[List[str]] = []
    distinct: List[List[str]] = []
    display_names: Dict[str, str] = {}

    def _read_yaml(path):
        if not path or not os.path.exists(path):
            logger.info(f"[menu_alias] 檔案不存在，略過: {path}")
            return {}
        with open(path, "r", encoding="utf-8-sig") as fh:
            return yaml.safe_load(fh) or {}

    shared = _read_yaml(readings_path)
    for entry in shared.get("readings", []) or []:
        pair = _as_pair(entry)
        if pair:
            readings.append(pair)

    local = _read_yaml(aliases_path)
    for entry in local.get("readings_extra", []) or []:
        pair = _as_pair(entry)
        if pair:
            readings.append(pair)
    for entry in local.get("equivalents", []) or []:
        group = entry.get("group") if isinstance(entry, dict) else entry
        if group and len(group) > 1:
            equivalents.append([str(x) for x in group])
    for entry in local.get("distinct", []) or []:
        pair = entry.get("pair") if isinstance(entry, dict) else entry
        if pair and len(pair) > 1:
            distinct.append([str(x) for x in pair])
    for key, value in (local.get("display_names", {}) or {}).items():
        display_names[str(key)] = str(value)

    # equivalents 條目也可以直接帶 display，等價於寫進 display_names
    for entry in local.get("equivalents", []) or []:
        if isinstance(entry, dict) and entry.get("display") and entry.get("group"):
            for member in entry["group"]:
                display_names[str(member)] = str(entry["display"])

    logger.info(
        f"[menu_alias] 載入 readings={len(readings)} equivalents={len(equivalents)} "
        f"distinct={len(distinct)} display_names={len(display_names)}"
    )
    return AliasKnowledge(readings, equivalents, distinct, display_names)


class FoldedMatcher:
    """在「折疊後的座標系」上做比對的小工具，傳票側與貼紙側共用。

    為什麼要包成一個物件：折疊後的候選字典必須快取（菜單有數百筆，每次
    OCR 重算是白花時間），而傳票側與貼紙側都需要同一套邏輯 —— 兩邊各寫
    一份遲早會漂移。

    menu_index 為 None 時整個退化成舊行為（只做 NFKC + 去空白），所以舊
    config 或測試替身也能照常運作。
    """

    def __init__(self, menu_index: Optional[MenuIndex] = None):
        self.menu_index = menu_index
        self._cache: Dict[int, Tuple[int, Dict[str, str]]] = {}

    def fold(self, text: str) -> str:
        if self.menu_index is None:
            return normalize_text(text, ignore_space=True)
        return self.menu_index.fold(text)

    def folded_map(self, candidates: Sequence[str]) -> Dict[str, str]:
        """{折疊 key: 原始候選}。候選清單是長壽物件，用 id + 長度當快取鍵。"""
        cached = self._cache.get(id(candidates))
        if cached is None or cached[0] != len(candidates):
            cached = (len(candidates), {self.fold(c): c for c in candidates})
            self._cache[id(candidates)] = cached
        return cached[1]

    def resolve(self, text: str, candidates: Sequence[str], fuzzy_matcher):
        """把一段 OCR 文字解析成「代表菜名」。

        兩段式：
          1. 先問等價層。命中代表這個寫法（或它折疊後的形式）就是某道菜，
             連 menus_mapping.yaml 沒收錄過的寫法也算 —— 這是「不同店家用
             漢字還是假名」能自動處理的關鍵。
          2. 沒命中才做模糊比對，而且在折疊後的座標系上計分。舊版直接拿
             原字串比，'から揚げ丼並' 會被 'から揚げ'(~80 分) 搶走並誤發
             WRONG_ITEM；折疊後兩者與 '唐揚げ丼並' 同分 100，順序就對了。

        Returns:
            (代表菜名|None, 分數)
        """
        if self.menu_index is not None:
            rep = self.menu_index.canonical(text)
            if rep is not None:
                return rep, 100

        folded_map = self.folded_map(candidates)
        result = fuzzy_matcher.match(self.fold(text), list(folded_map.keys()))
        if not result:
            return None, 0
        folded_hit, score = result
        return folded_map.get(folded_hit), score


def build_menu_index(
    know: AliasKnowledge,
    pool_name: str,
    names: Iterable[str],
    prefer: Iterable[str] = (),
) -> MenuIndex:
    """用載入好的知識組裝一個菜名池的索引。

    刻意做成共用函式：config.py（線上）與 tools/verify_menu_alias.py（稽核）
    必須用完全一樣的組裝方式，否則稽核報表會跟線上行為對不上。
    """
    index = MenuIndex(know.readings, know.equivalents, know.distinct, pool_name)
    return index.build(names, prefer=prefer, display_names=know.display_names)


def _as_pair(entry):
    """readings 支援 [from, to] 與 {from: x, to: y, tag: ...} 兩種寫法。"""
    if isinstance(entry, dict):
        src, dst = entry.get("from"), entry.get("to")
    elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
        src, dst = entry[0], entry[1]
    else:
        logger.warning(f"[menu_alias] 無法解析的 readings 條目，已忽略: {entry!r}")
        return None
    if not src or dst is None:
        logger.warning(f"[menu_alias] readings 條目缺 from/to，已忽略: {entry!r}")
        return None
    return str(src), str(dst)
