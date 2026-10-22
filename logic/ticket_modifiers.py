"""
ticket_modifiers.py
===================
傳票上的「修飾列」：那一列不是一道獨立的菜，而是對**別一列**的修改。

問題背景
--------
吉野家的套餐附湯，但客人可以換湯。POS 會在傳票上另外印一列：

    牛丼・並                 1
    牛丼お新香みそ汁セット      1
    変更しじみ汁              1      <- 這一列是修飾列

`menus_mapping.yaml` 把 `変更しじみ汁` 當成獨立品項（貼紙 `※変更しじみ`），
所以「代入的湯」本來就會正確展開；缺的是另一半 —— **把套餐裡那碗
みそ汁 從需求裡扣掉**。不扣的後果有兩個，後者比前者嚴重：

    1. 前端顯示「一份餐 + 兩種湯」（2026-09-21 客戶回報）
    2. 店家換湯後不會印 みそ汁 貼紙，`_is_order_satisfied()` 永遠等不到，
       這筆訂單到不了 TRAY_COMPLETED

為什麼不用 bbox 的 y 座標
------------------------
`ticket_layout.group_rows()` 已經依 Y 軸重疊率分列並按 cy 排序，
`parse_rows()` 照序輸出、`to_legacy_items()` 原序保留 —— 所以
**品項清單的 index 就是傳票的上下順序**。「往上找最近的套餐」在 list
上往前掃就好，不必再拉一個會隨現場漂動的 y 距離門檻。

三個概念
--------
1. 修飾列  哪些傳票菜名是修飾列，由 `menu_modifiers.yaml` 的 `substitutions`
           台帳認定（人工決定，不是靠字串猜）。文法規則只用來**提案**，
           讓 tools/verify_menu_modifiers.py 在 xlsx 改版後提醒有新的沒歸軸。
2. 軸      湯軸 / 飯軸 / 配菜軸…。軸決定「可以換掉什麼」（`slots`）。
3. 錨定    從修飾列往上找第一個「變體含有該軸成員」且該軸槽位還沒被用掉
           的列。軸感知 + 槽位容量 + 最近往上，三者合起來才不會
           「看到湯就亂換」。

刻意零依賴（只用 re / dataclasses / loguru），所以在 Windows 開發機上
可以完整驗證，不需要 DLA 或相機。菜單知識由呼叫端注入，槽位數量由
呼叫端以 callable 提供（狀態機才知道 menu_variants）。

風險不對稱
----------
扣錯（把本來要來的湯扣掉）會讓正確的貼紙被誤報 WRONG_ITEM，作業員看得到；
漏扣（沒認出修飾列）退回改動前的行為。兩者都比「靜默放過錯餐點」安全，
所以這裡的原則是**認不出來就不動作**：台帳沒登記的列一律當普通品項，
只留 log 與回報旗標，絕不用字串相似度硬猜。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from loguru import logger

# 修飾種類。目前只實作 substitute；add / remove 共用同一套錨定，之後只要
# 在台帳加 kind 與在狀態機加對應的套用分支。
KIND_SUBSTITUTE = "substitute"


# ---------------------------------------------------------------------------
# 知識層
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AxisCfg:
    """一個「可被替換的槽位」的定義。

    slots 是**貼紙側**的菜名（代表菜名），因為要被扣掉的是套餐的 sub_items。
    刻意用明列而不是「含『汁』就算湯」這種規則：菜單裡有 `汁なし担々麺`
    與 `ミニ汁なし担々麺`，那是麵不是湯，規則會把它們一起換掉。
    """

    name: str
    slots: frozenset = frozenset()
    display: str = ""


@dataclass(frozen=True)
class SubstitutionRule:
    """台帳的一筆：這個傳票菜名是修飾列，它換掉的是這個軸的槽位。"""

    ticket: str
    axis: str
    note: str = ""


@dataclass
class TicketGrammar:
    """修飾詞文法。**只用於提案與救援**，不用於認定。

    認定一律走 `menu_modifiers.yaml` 的台帳 —— 因為 `変更半熟` 這種列，
    光看字串無法知道它換掉的是哪個軸，必須人工決定一次。
    """

    substitute_patterns: Tuple[re.Pattern, ...] = ()
    ignore_prefixes: Tuple[str, ...] = ()
    sticker_markers: Tuple[str, ...] = ()

    def strip_prefixes(self, text: str) -> str:
        """去掉通路前綴（`D)` = 外送、`単)` = 單品販售…）。

        會反覆去除，因為實際資料裡出現過 `D)変)ご飯増量(無料)` 這種疊加。

        比對刻意不分大小寫：呼叫端傳進來的菜名通常已經過
        `normalize_text()`，而它的 `lower=True` 會把 `D)` 變成 `d)`。
        """
        changed = True
        while changed:
            changed = False
            for p in self.ignore_prefixes:
                if p and text[: len(p)].lower() == p.lower():
                    text = text[len(p):]
                    changed = True
        return text

    def match(self, ticket_name: str) -> Optional[str]:
        """看起來是修飾列嗎？是的話回傳「被代入的品名」，否則 None。"""
        body = self.strip_prefixes(ticket_name)
        for pat in self.substitute_patterns:
            m = pat.match(body)
            if m:
                item = (m.groupdict().get("item") or "").strip()
                if item:
                    return item
        return None


class ModifierKnowledge:
    """載入後的修飾知識。對外只有三個問題要回答：

        axis_of(傳票菜名)       這一列是修飾列嗎？換的是哪一軸？
        is_slot(軸, 貼紙菜名)   這個 sub_item 是該軸可被換掉的槽位嗎？
        propose(傳票菜名)       文法上看起來像修飾列嗎？（給稽核工具用）
    """

    def __init__(
        self,
        axes: Sequence[AxisCfg] = (),
        rules: Sequence[SubstitutionRule] = (),
        not_modifier: Sequence[str] = (),
        grammar: Optional[TicketGrammar] = None,
        normalize: Optional[Callable[[str], str]] = None,
    ):
        # normalize 由呼叫端注入（線上傳 logic.text_utils.normalize_text，
        # 或包上等價層的 canonical_or_self），這個模組本身不依賴它們。
        self._norm = normalize or (lambda s: str(s).strip())
        self.axes: Dict[str, AxisCfg] = {a.name: a for a in axes}
        self.rules: Dict[str, SubstitutionRule] = {}
        for r in rules:
            key = self._norm(r.ticket)
            if r.axis not in self.axes:
                logger.warning(
                    f"[ticket_modifiers] 台帳 '{r.ticket}' 指向不存在的軸 "
                    f"'{r.axis}'，該筆忽略（可用的軸: {sorted(self.axes)}）"
                )
                continue
            if key in self.rules and self.rules[key].axis != r.axis:
                logger.warning(
                    f"[ticket_modifiers] 台帳 '{r.ticket}' 重複登記且軸不同 "
                    f"({self.rules[key].axis} vs {r.axis})，保留先出現的"
                )
                continue
            self.rules[key] = r
        self.not_modifier: Set[str] = {self._norm(n) for n in not_modifier}
        self.grammar = grammar or TicketGrammar()

    def __bool__(self) -> bool:
        return bool(self.rules)

    # ------------------------------------------------------------------
    def axis_of(self, ticket_name: str) -> Optional[str]:
        rule = self.rules.get(self._norm(ticket_name))
        return rule.axis if rule else None

    def is_slot(self, axis: str, sticker_name: str) -> bool:
        cfg = self.axes.get(axis)
        if cfg is None:
            return False
        return self._norm(sticker_name) in cfg.slots

    def slot_members(self, axis: str) -> frozenset:
        cfg = self.axes.get(axis)
        return cfg.slots if cfg else frozenset()

    # ------------------------------------------------------------------
    def _strip_sticker_markers(self, text: str) -> str:
        """把貼紙文字的「已變更」標記剝掉：`※変更しじみ` -> `しじみ`。

        長標記優先，否則 `※` 會先吃掉 `※変更` 的前半。
        """
        s = self._norm(text)
        for marker in sorted(self.grammar.sticker_markers, key=len, reverse=True):
            m = self._norm(marker)
            if m and s.startswith(m):
                return s[len(m):]
        return s

    def derive_from_menu(self, menu_variants: Dict[str, List[List[str]]]) -> List[SubstitutionRule]:
        """從菜單自動推導「還沒登記」的修飾列，並就地補進台帳。

        為什麼需要自動推導（2026-09-21 實測）
        ------------------------------------
        現場傳票印 `しじみ汁に変更`，於是有人往 menus_mapping.yaml 加了一列
        `しじみ汁変更 -> ※変更しじみ`。台帳裡沒有這個寫法，換湯就靜默失效 ——
        前端又出現兩碗湯。同一道菜的變更寫法會隨店家/POS 版本一直長出新的
        （変更X / Xに変更 / X変更 / 変)X），要求每一個都先人工登記，等於
        保證遲早漏掉，而漏掉是靜默的。

        推導要求**兩個互相獨立的證據同時成立**，所以不是在猜：
          1. 傳票菜名命中 ja_ticket_grammar.yaml 的修飾詞文法
          2. 這一列自己的**貼紙文字帶有「変更」標記**，剝掉標記後剛好落在
             某一個軸的 slots 裡（`※変更しじみ` -> `しじみ` ∈ soup）

        第 2 點是關鍵的那道閘。`変更半熟` 的貼紙就是 `半熟玉子`，沒有標記，
        所以推不出來 —— 那是對的：「把套餐的玉子換成半熟」還是「加點一份
        半熟玉子」，資料上分不出來，必須有人決定一次。

        `not_modifier` 的結案記錄對推導有否決權；人工登記的軸永遠優先。

        Returns:
            這次新推導出來的規則（供呼叫端記 log / 稽核報表）。
        """
        derived: List[SubstitutionRule] = []
        for ticket_name in sorted(menu_variants):
            key = self._norm(ticket_name)
            if key in self.rules or key in self.not_modifier:
                continue                      # 人工判定優先，已結案的不再推
            if not self.grammar.match(key):
                continue                      # 證據一：文法
            hits: Set[str] = set()
            evidence: Dict[str, str] = {}
            for combo in menu_variants.get(ticket_name) or []:
                for sub in combo:
                    bare = self._strip_sticker_markers(sub)
                    if bare == self._norm(sub):
                        continue               # 貼紙沒有變更標記 -> 不算證據
                    for axis_name, cfg in self.axes.items():
                        if bare in cfg.slots:
                            hits.add(axis_name)
                            evidence[axis_name] = f"{sub} -> {bare}"
            if len(hits) == 1:
                axis = next(iter(hits))
                rule = SubstitutionRule(
                    ticket=ticket_name, axis=axis,
                    note=f"由文法+貼紙標記自動推導（{evidence[axis]}）",
                )
                self.rules[key] = rule
                derived.append(rule)
            elif len(hits) > 1:
                # 一個貼紙同時落在兩個軸 = 軸定義重疊，不猜，交給人
                logger.warning(
                    f"[ticket_modifiers] '{ticket_name}' 的貼紙同時落在多個軸 "
                    f"{sorted(hits)}，不自動推導，請在 menu_modifiers.yaml 明確登記"
                )
        if derived:
            logger.info(
                f"[ticket_modifiers] 自動推導出 {len(derived)} 筆修飾列: "
                + ", ".join(f"{r.ticket}->{r.axis}" for r in derived)
            )
        return derived

    def propose(self, ticket_name: str) -> Optional[str]:
        """文法上像修飾列、但台帳還沒登記（也沒被判定為「不是修飾列」）。

        回傳被代入的品名，供稽核工具產生可直接貼進 yaml 的提案。
        """
        key = self._norm(ticket_name)
        if key in self.rules or key in self.not_modifier:
            return None
        return self.grammar.match(key)

    def axis_display(self, axis: str) -> str:
        cfg = self.axes.get(axis)
        return (cfg.display or axis) if cfg else axis


# ---------------------------------------------------------------------------
# 解析結果
# ---------------------------------------------------------------------------

@dataclass
class TicketModifier:
    """一筆修飾。`anchor_index` 為 None 代表錨不到（退回當普通品項）。"""

    kind: str
    axis: str
    ticket_name: str
    quantity: int
    row_index: int
    anchor_index: Optional[int] = None
    anchor_name: str = ""
    applied: int = 0          # 實際扣掉幾份（受錨定列的槽位容量限制）

    @property
    def is_attached(self) -> bool:
        return self.anchor_index is not None and self.applied > 0


@dataclass
class OrderLine:
    """傳票的一列。

    注意 `is_modifier` 為 True 的列**仍然要照常展開成待核對項目** ——
    `変更しじみ汁` 的貼紙 `※変更しじみ` 是真的會被印出來、要核對的。
    is_modifier 只代表「這一列**額外**會扣掉別列的東西」。
    """

    name: str
    quantity: Optional[int]
    row_index: int
    is_modifier: bool = False
    # 掛在這一列上的修飾（這一列是被修改的那個「錨」）
    modifiers: List[TicketModifier] = field(default_factory=list)


SlotCounter = Callable[[str, str], int]
"""(傳票菜名, 軸) -> 該菜名「每一份」提供幾個該軸的槽位。

由狀態機實作（它才有 menu_variants）。這個模組不碰菜單資料。
"""


def resolve_modifiers(
    parsed_items: Sequence[Tuple[str, Optional[int]]],
    know: Optional[ModifierKnowledge],
    slot_counter: SlotCounter,
) -> Tuple[List[OrderLine], List[TicketModifier]]:
    """把 OrderParser 的 [(品名, 數量)] 依傳票順序解析成帶修飾的列。

    Args:
        parsed_items: **必須保持傳票上下順序**（OrderParser 原生就是）。
        know: 修飾知識；None 或空台帳時原樣回傳，行為與改動前完全一致。
        slot_counter: (傳票菜名, 軸) -> 每份提供的槽位數。

    Returns:
        (lines, unattached)
        unattached 是錨不到的修飾，呼叫端應該記 log 並在 MQTT 加旗標 ——
        這代表傳票長得跟我們的假設不一樣，值得現場看一眼。
    """
    lines: List[OrderLine] = [
        OrderLine(name=name, quantity=qty, row_index=idx)
        for idx, (name, qty) in enumerate(parsed_items)
    ]
    if not know or not know.rules:
        return lines, []

    # 每一列在各軸上已經被消耗掉的槽位數。同一個套餐被換兩次湯是不合理的，
    # 這份帳就是用來擋掉「兩個修飾列都錨到同一個套餐」的。
    consumed: Dict[int, Dict[str, int]] = {}
    unattached: List[TicketModifier] = []

    # 先標出所有修飾列，錨定時才知道哪些列不可以當錨
    for line in lines:
        if know.axis_of(line.name) is not None:
            line.is_modifier = True

    for line in lines:
        if not line.is_modifier:
            continue
        axis = know.axis_of(line.name)
        if line.quantity is None:
            # 數量未確認：不知道要換幾份，不動作（與 _expand_parsed_items
            # 對未確認數量的一貫處理相同 —— 不預設 1）
            logger.warning(
                f"[ticket_modifiers] 修飾列 '{line.name}' 數量未確認，不套用替換"
            )
            continue

        mod = TicketModifier(
            kind=KIND_SUBSTITUTE,
            axis=axis,
            ticket_name=line.name,
            quantity=int(line.quantity),
            row_index=line.row_index,
        )

        anchor = _find_anchor(lines, line.row_index, axis, consumed, slot_counter)
        if anchor is None:
            unattached.append(mod)
            logger.warning(
                f"[ticket_modifiers] 修飾列 '{line.name}'（{know.axis_display(axis)}）"
                f"在它上方找不到含該軸槽位的套餐，退回當普通品項處理"
            )
            continue

        anchor_line, capacity = anchor
        take = min(mod.quantity, capacity)
        mod.anchor_index = anchor_line.row_index
        mod.anchor_name = anchor_line.name
        mod.applied = take
        consumed.setdefault(anchor_line.row_index, {})
        consumed[anchor_line.row_index][axis] = (
            consumed[anchor_line.row_index].get(axis, 0) + take
        )
        anchor_line.modifiers.append(mod)

        logger.info(
            f"[ticket_modifiers] '{line.name}' x{mod.quantity} 錨定到上方第 "
            f"{anchor_line.row_index} 列 '{anchor_line.name}'，"
            f"扣除 {know.axis_display(axis)} x{take}"
        )
        if take < mod.quantity:
            logger.warning(
                f"[ticket_modifiers] '{line.name}' 要換 {mod.quantity} 份，但 "
                f"'{anchor_line.name}' 只剩 {capacity} 個 {know.axis_display(axis)} "
                f"槽位，只扣 {take} 份"
            )

    return lines, unattached


def _find_anchor(
    lines: Sequence[OrderLine],
    from_index: int,
    axis: str,
    consumed: Dict[int, Dict[str, int]],
    slot_counter: SlotCounter,
) -> Optional[Tuple[OrderLine, int]]:
    """從 from_index 往上找第一個還有該軸空槽位的列。

    往上找是現場確認過的印法：換湯一定印在套餐下方（2026-09-21）。
    刻意不做「找不到就往下找」的 fallback —— 那會在傳票版面改變時
    靜默地換錯套餐，而漏換只是退回改動前的行為。
    """
    for j in range(from_index - 1, -1, -1):
        cand = lines[j]
        if cand.is_modifier:
            continue          # 修飾列自己不能當錨
        if cand.quantity is None or cand.quantity <= 0:
            continue
        per_unit = int(slot_counter(cand.name, axis) or 0)
        if per_unit <= 0:
            continue          # 這一列沒有該軸的槽位，跳過繼續往上
        capacity = per_unit * int(cand.quantity) - consumed.get(j, {}).get(axis, 0)
        if capacity > 0:
            return cand, capacity
    return None


# ---------------------------------------------------------------------------
# 載入
# ---------------------------------------------------------------------------

def load_ticket_grammar(path: Optional[str]) -> TicketGrammar:
    """讀 `tasks/ocr/ja_ticket_grammar.yaml`（跨品牌共用、選填）。"""
    data = _read_yaml(path, "ticket_grammar")
    patterns: List[re.Pattern] = []
    for raw in data.get("substitute_patterns", []) or []:
        try:
            patterns.append(re.compile(str(raw)))
        except re.error as exc:
            logger.warning(f"[ticket_modifiers] 無法編譯的文法 {raw!r}: {exc}")
    grammar = TicketGrammar(
        substitute_patterns=tuple(patterns),
        ignore_prefixes=tuple(str(p) for p in (data.get("ignore_prefixes") or [])),
        sticker_markers=tuple(str(m) for m in (data.get("sticker_markers") or [])),
    )
    if patterns:
        logger.info(
            f"[ticket_modifiers] 載入修飾文法 {len(patterns)} 條、"
            f"通路前綴 {len(grammar.ignore_prefixes)} 個"
        )
    return grammar


def load_modifier_knowledge(
    grammar_path: Optional[str],
    modifiers_path: Optional[str],
    normalize: Optional[Callable[[str], str]] = None,
    canonical_sticker: Optional[Callable[[str], str]] = None,
) -> ModifierKnowledge:
    """讀文法 + 某品牌的軸台帳，組成 ModifierKnowledge。

    Args:
        normalize: 傳票/貼紙菜名的正規化（線上傳 normalize_text）。
        canonical_sticker: 貼紙側的等價層折疊（`sticker_index.canonical_or_self`）。
            台帳可以照 xlsx 的寫法寫，載入時會折到代表菜名，跟
            `menu_variants` 裡的 sub_items 對得起來。

    兩份檔案都是選填：缺檔就回一份空的知識，`resolve_modifiers()` 會
    原樣回傳品項清單，行為與改動前完全一致。
    """
    grammar = load_ticket_grammar(grammar_path)
    data = _read_yaml(modifiers_path, "menu_modifiers")

    norm = normalize or (lambda s: str(s).strip())
    canon = canonical_sticker or (lambda s: norm(s))

    axes: List[AxisCfg] = []
    for entry in data.get("axes", []) or []:
        if not isinstance(entry, dict) or not entry.get("name"):
            logger.warning(f"[ticket_modifiers] 無法解析的軸定義，已忽略: {entry!r}")
            continue
        slots = frozenset(
            norm(canon(str(s))) for s in (entry.get("slots") or []) if str(s).strip()
        )
        if not slots:
            logger.warning(
                f"[ticket_modifiers] 軸 '{entry['name']}' 沒有任何 slots，"
                "它不會扣掉任何東西"
            )
        axes.append(AxisCfg(
            name=str(entry["name"]),
            slots=slots,
            display=str(entry.get("display") or ""),
        ))

    rules: List[SubstitutionRule] = []
    for entry in data.get("substitutions", []) or []:
        if not isinstance(entry, dict) or not entry.get("ticket") or not entry.get("axis"):
            logger.warning(f"[ticket_modifiers] 無法解析的 substitutions 條目: {entry!r}")
            continue
        rules.append(SubstitutionRule(
            ticket=str(entry["ticket"]),
            axis=str(entry["axis"]),
            note=str(entry.get("note") or ""),
        ))

    not_modifier = [
        str(e.get("ticket") if isinstance(e, dict) else e)
        for e in (data.get("not_modifier") or [])
    ]

    know = ModifierKnowledge(
        axes=axes, rules=rules, not_modifier=not_modifier,
        grammar=grammar, normalize=norm,
    )
    if know.rules:
        logger.info(
            f"[ticket_modifiers] 載入軸 {len(know.axes)} 個"
            f"（{', '.join(f'{a.name}:{len(a.slots)}槽' for a in axes)}）、"
            f"修飾列台帳 {len(know.rules)} 筆、已判定非修飾列 {len(know.not_modifier)} 筆"
        )
    else:
        logger.info("[ticket_modifiers] 沒有修飾列台帳，傳票修飾功能停用")
    return know


def _read_yaml(path: Optional[str], label: str) -> dict:
    import os

    import yaml

    if not path or not os.path.exists(path):
        logger.info(f"[ticket_modifiers] {label} 檔案不存在，略過: {path}")
        return {}
    with open(path, "r", encoding="utf-8-sig") as fh:
        return yaml.safe_load(fh) or {}
