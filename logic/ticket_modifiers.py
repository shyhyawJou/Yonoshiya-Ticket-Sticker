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
    2. 店家換湯後不會印 みそ汁 貼紙，`OrderRequirements.is_order_satisfied()` 永遠等不到，
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
3. 錨定    從修飾列找最近一個「變體含有該軸可刪的成員」且該軸槽位還沒被用掉
           的列。軸感知 + 槽位容量 + 最近，三者合起來才不會「看到湯就亂換」。
           找的方向由軸的 anchor 決定：nearest_above 只往上（2026-09-21 的印法），
           anywhere 先往上、沒有再往下（2026-10-28：POS 可能把變更品項都印在最下面）。
           可刪的成員由軸的 remove_slots 決定（湯軸目前只刪 みそ汁）。

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

# 錨定方式（menu_modifiers.yaml 軸的 anchor）
ANCHOR_ABOVE = "nearest_above"   # 只往上找最近的套餐（2026-09-21 的印法：換湯一定印在套餐下方）
ANCHOR_ANYWHERE = "anywhere"     # 先往上找最近的，上面沒有再往下找最近的
_ANCHOR_MODES = (ANCHOR_ABOVE, ANCHOR_ANYWHERE)


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
    # 變更時實際要刪的槽位（slots 的子集）。空 = slots 全部都可以刪。
    # slots 仍是「哪些貼紙算這一軸」的完整定義（自動推導修飾列要用），兩者分開。
    removable: frozenset = frozenset()
    anchor: str = ANCHOR_ABOVE
    # 被刪的那一份照樣送前端顯示、貼了算 OK，只是不算完成條件（見 menu_modifiers.yaml）
    show_removed: bool = False

    def can_remove(self, sticker_name: str) -> bool:
        return sticker_name in (self.removable or self.slots)


@dataclass(frozen=True)
class SetMainRule:
    """套餐 -> 它下面那個 size 列要對到哪一種主餐（menu_modifiers.yaml 的 set_size_rows.mains）。

    set_contains 為空 = 預設規則（其他所有套餐）。main 是主餐貼紙去掉 size 後的名字，
    with_ 是同一份主餐另外會印的貼紙（ねぎ玉牛丼 = 牛丼 + ねぎ）。
    """

    set_contains: Tuple[str, ...]
    main: str
    with_: frozenset = frozenset()

    def label(self) -> str:
        return self.main + (f" + {'、'.join(sorted(self.with_))}" if self.with_ else "")


@dataclass
class SetSizeCfg:
    """套餐名稱太長、主餐的 size 被印到下一列（`牛丼お新香みそ汁ｾｯﾄ` / `並`）的處理。

    `並` 在 menus_mapping.yaml 是一對多（一般牛丼 / チーズ牛丼 / ねぎ玉 / 明太高菜マヨ），
    客戶的菜單無法改，所以用套餐名稱決定這一列是哪一種主餐，見 resolve_set_sizes。
    """

    size_rows: frozenset = frozenset()       # 只有 size 的傳票品名（並 / 大 / 特 …）
    size_words: Tuple[str, ...] = ()         # 從主餐貼紙去掉 size 時要認得的字（長的先比）
    set_markers: Tuple[str, ...] = ()        # 套餐名稱開頭或結尾的字（セット / S）
    mains: Tuple[SetMainRule, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.size_rows and self.mains)

    def main_of(self, sticker: str) -> Optional[str]:
        """主餐貼紙去掉結尾的 size：`チーズ牛丼並盛` -> `チーズ牛丼`；不是主餐貼紙回 None。"""
        s, changed = sticker, True
        while changed:
            changed = False
            for w in self.size_words:
                if w and s.endswith(w) and len(s) > len(w):
                    s, changed = s[: -len(w)], True
                    break
        return s if s != sticker else None

    def matches(self, rule: SetMainRule, combo: Sequence[str]) -> bool:
        """這個貼紙組合是不是規則說的那一種主餐。"""
        mains = [(sub, self.main_of(sub)) for sub in combo]
        mains = [(sub, m) for sub, m in mains if m is not None]
        if len(mains) != 1 or mains[0][1] != rule.main:
            return False
        extras = frozenset(sub for sub in combo if sub != mains[0][0])
        return extras == rule.with_


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
        self.set_sizes: SetSizeCfg = SetSizeCfg()

    def __bool__(self) -> bool:
        return bool(self.rules) or bool(self.set_sizes)

    # ------------------------------------------------------------------
    def is_set(self, ticket_name: str) -> bool:
        """套餐：去掉通路前綴（D) …）後，開頭或結尾是 set_markers（セット / S）。

        只看頭尾不看中間：`S` 只有一個字，出現在中間的不算（2026-10 菜單盤點 32 筆全是套餐）。
        """
        body = self.grammar.strip_prefixes(self._norm(ticket_name))
        return any(body.startswith(m) or body.endswith(m) for m in self.set_sizes.set_markers if m)

    def set_main_rule(self, set_name: str) -> Optional[SetMainRule]:
        """套餐名稱 -> 主餐規則（由上往下第一個命中；set_contains 空的是預設）。"""
        name = self._norm(set_name)
        for rule in self.set_sizes.mains:
            if not rule.set_contains or any(c in name for c in rule.set_contains):
                return rule
        return None

    # ------------------------------------------------------------------
    def axis_of(self, ticket_name: str) -> Optional[str]:
        rule = self.rules.get(self._norm(ticket_name))
        return rule.axis if rule else None

    def is_slot(self, axis: str, sticker_name: str) -> bool:
        cfg = self.axes.get(axis)
        if cfg is None:
            return False
        return self._norm(sticker_name) in cfg.slots

    def is_removable(self, axis: str, sticker_name: str) -> bool:
        """這個 sub_item 是「變更時要刪的」那一份嗎？（軸的 remove_slots；沒設時 = is_slot）

        錨定容量與實際扣除都用這個；is_slot 只用在認出修飾列（derive_from_menu）。
        """
        cfg = self.axes.get(axis)
        return cfg is not None and cfg.can_remove(self._norm(sticker_name))

    def anchor_mode(self, axis: str) -> str:
        cfg = self.axes.get(axis)
        return cfg.anchor if cfg else ANCHOR_ABOVE

    def show_removed(self, axis: str) -> bool:
        cfg = self.axes.get(axis)
        return bool(cfg and cfg.show_removed)

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
    # size 列（`並`）屬於哪個套餐、對到哪一種主餐（見 resolve_set_sizes）；None = 照原本一對多
    size_rule: Optional[SetMainRule] = None
    size_set: str = ""


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
    if not know:
        return lines, []
    resolve_set_sizes(lines, know)
    if not know.rules:
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
            # 數量未確認：不知道要換幾份，不動作（與 OrderRequirements.expand_parsed_items
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

        anywhere = know.anchor_mode(axis) == ANCHOR_ANYWHERE
        # 一個修飾列可以分攤到好幾個套餐：`変更あさり汁 x2` 配兩個各附一碗 みそ汁 的套餐
        # （2026-10-28 現場傳票 853：3 個 ｾｯﾄポテトみそ汁 + 変更とん汁 x1 + 変更あさり汁 x2
        # 全部印在最下面）。只錨一個套餐的話只會扣一碗，訂單就一直等那碗不會來的湯。
        remaining = mod.quantity
        while remaining > 0:
            anchor = _find_anchor(lines, line.row_index, axis, consumed, slot_counter, anywhere=anywhere)
            if anchor is None:
                break
            anchor_line, capacity = anchor
            take = min(remaining, capacity)
            part = TicketModifier(
                kind=mod.kind, axis=axis, ticket_name=line.name, quantity=mod.quantity,
                row_index=line.row_index, anchor_index=anchor_line.row_index,
                anchor_name=anchor_line.name, applied=take,
            )
            consumed.setdefault(anchor_line.row_index, {})
            consumed[anchor_line.row_index][axis] = (
                consumed[anchor_line.row_index].get(axis, 0) + take
            )
            anchor_line.modifiers.append(part)
            remaining -= take
            logger.info(
                f"[ticket_modifiers] '{line.name}' x{mod.quantity} 錨定到"
                f"{'上方' if anchor_line.row_index < line.row_index else '下方'}第 "
                f"{anchor_line.row_index} 列 '{anchor_line.name}'，"
                f"扣除 {know.axis_display(axis)} x{take}"
            )

        if remaining == mod.quantity:
            unattached.append(mod)
            logger.warning(
                f"[ticket_modifiers] 修飾列 '{line.name}'（{know.axis_display(axis)}）"
                f"在{'整張傳票' if anywhere else '它上方'}找不到含可替換槽位的套餐，退回當普通品項處理"
            )
        elif remaining > 0:
            # 換了一部分：剩下的份數錨不到，跟「整列錨不到」一樣回報
            unattached.append(TicketModifier(
                kind=mod.kind, axis=axis, ticket_name=line.name, quantity=remaining,
                row_index=line.row_index,
            ))
            logger.warning(
                f"[ticket_modifiers] '{line.name}' 要換 {mod.quantity} 份，但只找到 "
                f"{mod.quantity - remaining} 個可換的 {know.axis_display(axis)} 槽位，"
                f"剩 {remaining} 份錨不到"
            )

    return lines, unattached


def resolve_set_sizes(lines: Sequence[OrderLine], know: ModifierKnowledge) -> None:
    """size 列（`並`）往上找最近的套餐，用套餐名稱決定它是哪一種主餐，結果寫在 line.size_rule。

    客戶的套餐名稱太長時，主餐的 size 會印到下一列：

        1 牛丼お新香みそ汁ｾｯﾄ        <- 套餐（貼紙只有 お新香、みそ汁）
        1   並                      <- 主餐的 size：`並` 一對多，可能是一般牛丼 / チーズ牛丼 / …

    往上找時跳過其他 size 列（`D)牛からペアｾｯﾄ` 下面接兩列 `並`）與修飾列（換湯…）。
    找不到套餐、或主餐規則對不上任何變體時不動作，維持原本的「任一種主餐都算」
    （展開時才知道變體，見 OrderRequirements.expand_parsed_items）。
    """
    cfg = know.set_sizes
    if not cfg:
        return
    for line in lines:
        if know._norm(line.name) not in cfg.size_rows:
            continue
        owner = None
        for j in range(line.row_index - 1, -1, -1):
            cand = lines[j]
            if know._norm(cand.name) in cfg.size_rows or know.axis_of(cand.name) is not None:
                continue
            if know.is_set(cand.name):
                owner = cand
            break   # 碰到第一個不是 size 列 / 修飾列的列就停：不是套餐就代表這列 size 不屬於套餐
        if owner is None:
            logger.info(
                f"[ticket_modifiers] size 列 '{line.name}'（第 {line.row_index} 列）上方找不到套餐，"
                f"維持原本的「任一種主餐都算」"
            )
            continue
        rule = know.set_main_rule(owner.name)
        if rule is None:
            logger.warning(
                f"[ticket_modifiers] size 列 '{line.name}' 屬於套餐 '{owner.name}'，但主餐規則沒有"
                f"任何一條命中（set_size_rows.mains 少了預設規則？），維持原本的一對多"
            )
            continue
        line.size_rule, line.size_set = rule, owner.name
        logger.info(
            f"[ticket_modifiers] size 列 '{line.name}'（第 {line.row_index} 列）屬於上方套餐 "
            f"'{owner.name}'（第 {owner.row_index} 列）→ 主餐 {rule.label()}"
        )


def _find_anchor(
    lines: Sequence[OrderLine],
    from_index: int,
    axis: str,
    consumed: Dict[int, Dict[str, int]],
    slot_counter: SlotCounter,
    anywhere: bool = False,
) -> Optional[Tuple[OrderLine, int]]:
    """從 from_index 找最近一個還有該軸空槽位的列。

    anywhere=False（nearest_above）：只往上找。2026-09-21 現場確認的印法是換湯一定印在
    套餐下方，所以當時刻意不做「找不到就往下找」—— 版面改變時會靜默地換錯套餐。

    anywhere=True：先往上、上面沒有再往下。2026-10-28 客戶說明 POS 可能把所有變更品項
    印在傳票最下面，換湯不再對應「正上方那個套餐」，而是「這張傳票裡附的那碗湯」。
    刪哪一個套餐的湯對核對結果沒有差別（扣的都是同一種湯），只影響顯示上是哪一份少了湯，
    所以仍先取上方最近的，舊印法的結果跟以前一樣。
    """
    order = list(range(from_index - 1, -1, -1))
    if anywhere:
        order += list(range(from_index + 1, len(lines)))
    for j in order:
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
        removable = frozenset(
            norm(canon(str(s))) for s in (entry.get("remove_slots") or []) if str(s).strip()
        )
        if removable - slots:
            logger.warning(
                f"[ticket_modifiers] 軸 '{entry['name']}' 的 remove_slots 有不在 slots 裡的: "
                f"{sorted(removable - slots)}（照樣會刪，但請確認是不是寫錯）"
            )
        anchor = str(entry.get("anchor") or ANCHOR_ABOVE)
        if anchor not in _ANCHOR_MODES:
            logger.warning(
                f"[ticket_modifiers] 軸 '{entry['name']}' 的 anchor={anchor!r} 不認得，"
                f"改用 {ANCHOR_ABOVE}（可用: {list(_ANCHOR_MODES)}）"
            )
            anchor = ANCHOR_ABOVE
        axes.append(AxisCfg(
            name=str(entry["name"]),
            slots=slots,
            display=str(entry.get("display") or ""),
            removable=removable,
            anchor=anchor,
            show_removed=bool(entry.get("show_removed", False)),
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
    know.set_sizes = _load_set_sizes(data.get("set_size_rows") or {}, norm, canon)
    if know.rules:
        def _axis_txt(a):
            rm = f"、只刪 {sorted(a.removable)}" if a.removable else ""
            return f"{a.name}:{len(a.slots)}槽{rm}、錨定={a.anchor}、被刪的{'照樣顯示' if a.show_removed else '不顯示'}"
        logger.info(
            f"[ticket_modifiers] 載入軸 {len(know.axes)} 個"
            f"（{'；'.join(_axis_txt(a) for a in axes)}）、"
            f"修飾列台帳 {len(know.rules)} 筆、已判定非修飾列 {len(know.not_modifier)} 筆"
        )
    else:
        logger.info("[ticket_modifiers] 沒有修飾列台帳，傳票修飾功能停用")
    return know


def _load_set_sizes(data: dict, norm, canon) -> SetSizeCfg:
    """menu_modifiers.yaml 的 set_size_rows -> SetSizeCfg（沒寫就是停用）。"""
    if not data:
        return SetSizeCfg()
    size_rows = frozenset(norm(str(s)) for s in (data.get("size_rows") or []) if str(s).strip())
    words = {norm(str(s)) for s in list(data.get("size_rows") or []) + list(data.get("size_suffixes") or [])}
    mains = []
    for entry in data.get("mains") or []:
        if not isinstance(entry, dict) or not entry.get("main"):
            logger.warning(f"[ticket_modifiers] 無法解析的 set_size_rows.mains 條目: {entry!r}")
            continue
        mains.append(SetMainRule(
            set_contains=tuple(norm(str(c)) for c in (entry.get("set_contains") or []) if str(c).strip()),
            main=norm(str(entry["main"])),
            with_=frozenset(norm(canon(str(w))) for w in (entry.get("with") or []) if str(w).strip()),
        ))
    cfg = SetSizeCfg(
        size_rows=size_rows,
        size_words=tuple(sorted((w for w in words if w), key=len, reverse=True)),
        set_markers=tuple(norm(str(m)) for m in (data.get("set_markers") or []) if str(m).strip()),
        mains=tuple(mains),
    )
    if cfg:
        logger.info(
            f"[ticket_modifiers] 套餐 size 列：size {sorted(size_rows)}、套餐標記 {list(cfg.set_markers)}、"
            f"主餐規則 {len(mains)} 條（{'；'.join(r.label() for r in mains)}）"
        )
    return cfg


def load_extra_sets(modifiers_path: Optional[str]) -> List[Tuple[str, List[str]]]:
    """set_size_rows.extra_sets：客戶菜單還沒收錄、但現場會出現的套餐 [(傳票名, 貼紙)]。

    config.load_config 在建菜名索引之前呼叫，讓解析器認得這個名字。貼紙可以是空的
    （套餐本身不印貼紙，內容由下面的 size 列 / 獨立列帶出）。
    """
    data = _read_yaml(modifiers_path, "menu_modifiers").get("set_size_rows") or {}
    out = []
    for entry in data.get("extra_sets") or []:
        if isinstance(entry, dict) and entry.get("ticket"):
            out.append((str(entry["ticket"]), [str(s) for s in (entry.get("sub_items") or [])]))
    return out


def _read_yaml(path: Optional[str], label: str) -> dict:
    import os

    import yaml

    if not path or not os.path.exists(path):
        logger.info(f"[ticket_modifiers] {label} 檔案不存在，略過: {path}")
        return {}
    with open(path, "r", encoding="utf-8-sig") as fh:
        return yaml.safe_load(fh) or {}
