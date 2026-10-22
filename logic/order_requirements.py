"""
order_requirements.py
=====================
訂單需求：傳票解析出的品項，換算成「要核對哪些貼紙、湊到什麼程度算完成」。

從 TrayStateMachine 拆出來的一層，ticket reader（開單 / 第二張合併）跟 sticker reader
（確認進度、完成判定）共用同一份。內容跟拆出來之前一字不差，只是方法名稱去掉了
底線（現在是給別的模組呼叫的介面）：

    expand_parsed_items        傳票品項 -> 聯集 / 顯示清單 / 組合假設 / 改名表（含換湯換飯、[傳票→貼紙] 那行）
    recompute_expected_combos  從還活著的每張傳票重算整筆訂單的組合假設
    is_order_satisfied         任一個組合假設被湊齊就算完成
    resolve_effective_combo    挑最能解釋現場的那個組合（CSV 用）
    update_confirmed_count     貼紙確認後更新歷史最高確認數、達標鎖定

最近一次展開的附帶結果放在屬性上（last_substitutions / last_unattached_modifiers /
last_mapping_lines / last_unconfirmed_quantity），開單那一刻組 TICKET_READY 用。
"""

from __future__ import annotations

import itertools
from collections import Counter
from typing import Dict, List

from loguru import logger

from logic.text_utils import normalize_text
from logic.models import Tray
from logic.ticket_modifiers import resolve_modifiers


class OrderRequirements:
    def __init__(self, menu_variants: Dict[str, List[List[str]]] = None,
                 menu_distinct: List[List[str]] = None, modifier_knowledge=None):
        # 傳票名稱 -> 多個貼紙組合變體。值一定是 List[List[str]]。
        self.menu_variants = menu_variants or {}
        # menu_aliases.yaml 的 distinct 台帳，轉成「無序配對」方便查詢。
        # 用途見 expand_parsed_items：人工判定不是同一道菜的配對，
        # 不准被「同名多變體」自動推論成異寫。
        self.menu_distinct_pairs = {
            frozenset((normalize_text(a, ignore_space=True),
                       normalize_text(b, ignore_space=True)))
            for group in (menu_distinct or [])
            for a, b in itertools.combinations(group, 2)
        }
        if self.menu_distinct_pairs:
            logger.info(
                f"[tray_state_machine] 載入 distinct 台帳 {len(self.menu_distinct_pairs)} 組配對，"
                f"這些不會被自動視為異寫"
            )
        # 傳票修飾列知識（換湯/換飯…）。None 或空台帳時所有相關流程都是
        # no-op，行為與改動前完全一致 —— 沒有 menu_modifiers.yaml 的品牌
        # （例如 suzumo）不需要任何額外設定。
        self.modifier_knowledge = modifier_knowledge
        if modifier_knowledge:
            logger.info(
                f"[tray_state_machine] 傳票修飾列已啟用，台帳 "
                f"{len(modifier_knowledge.rules)} 筆"
            )
        # 最近一次解析出的替換與「錨不到的修飾列」，供 TICKET_READY 回報
        self.last_substitutions: List[dict] = []
        self.last_unattached_modifiers: List[dict] = []
        # 最近一次解析：每一列傳票品項轉成哪些貼紙（[傳票→貼紙] 那行 log 用）
        self.last_mapping_lines: List[str] = []

        # 最近一次展開時數量未確認的品名（開單時放進 TICKET_READY 的 unconfirmed_quantity）
        self.last_unconfirmed_quantity: List[str] = []

    def update_confirmed_count(self, tray: Tray, item_name: str) -> bool:
        """
        根據目前畫面中實際存在的 checked item 數量，
        更新前端歷史確認數。

        confirmed_counts 採 high-water mark：
        只能增加，不能因餐點暫時離開而下降。

        回傳：
            True  = confirmed count 有增加
            False = 沒有增加
        """
        if not item_name:
            return False

        expected_count = Counter(tray.expected_items).get(item_name, 0)
        if expected_count <= 0:
            return False

        # checked_items 代表目前實際仍存在的已確認物件
        visible_count = Counter(tray.checked_items).get(item_name, 0)

        old_confirmed = tray.confirmed_counts.get(item_name, 0)

        # high-water mark
        new_confirmed = min(
            expected_count,
            max(old_confirmed, visible_count)
        )

        if new_confirmed <= old_confirmed:
            return False

        tray.confirmed_counts[item_name] = new_confirmed

        logger.info(
            f"[tray_state_machine] 確認進度增加 "
            f"tray={tray.id}, "
            f"item={item_name}, "
            f"visible={visible_count}, "
            f"confirmed={old_confirmed}->{new_confirmed}, "
            f"expected={expected_count}"
        )

        # 真正達標才 lock
        if new_confirmed >= expected_count:
            tray.locked_items[item_name] = expected_count

            logger.info(
                f"[tray_state_machine] 品項達標並鎖定 "
                f"tray={tray.id}, "
                f"item={item_name}, "
                f"count={expected_count}"
            )

        return True

    # ------------------------------------------------------------
    # 多重假設：同一個傳票名稱可能對到多個貼紙組合
    # ------------------------------------------------------------
    MAX_COMBOS = 64   # 組合假設上限，超過就退回只取各品項的第一個變體

    def cartesian_merge(self, groups: List[List[Counter]]) -> List[Counter]:
        """把「每個來源各自的變體選項」展開成整筆訂單的組合假設。

        groups = [[品項A的變體1, 品項A的變體2], [品項B的變體1], ...]
        回傳   = [整筆訂單需求表1, 整筆訂單需求表2, ...]

        刻意做笛卡兒積而不是「每個品項各自獨立判斷」：不同變體可能共用同一個
        子品項（例如兩列都含 ご飯），獨立判斷會讓同一份 ご飯 被重複計算成
        滿足兩邊。組合數有上限保護，正常資料遠遠用不到。
        """
        combos: List[Counter] = [Counter()]
        for options in groups:
            if not options:
                continue
            if len(combos) * len(options) > self.MAX_COMBOS:
                logger.warning(
                    f"[tray_state_machine] 組合假設數超過上限 {self.MAX_COMBOS}，"
                    f"該品項只採用第一個變體（變體數={len(options)}）"
                )
                options = options[:1]
            merged_list: List[Counter] = []
            for base in combos:
                for option in options:
                    merged = base.copy()
                    merged.update(option)   # Counter.update = 需求量相加
                    merged_list.append(merged)
            combos = merged_list

        # 去重：Counter 不可 hash，用 items 的 frozenset 當指紋
        unique: List[Counter] = []
        seen = set()
        for combo in combos:
            fingerprint = frozenset(combo.items())
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique.append(combo)
        return [c for c in unique if c]

    @staticmethod
    def pick_primary_variant(item_name: str, variants: List[List[str]]) -> List[str]:
        """挑一個變體當「對外顯示用」的版本。

        規則（依序）：
          1. 品項數最少的 —— 顯示絕不要求「可能永遠不會來」的東西。
             チーズ 的兩個變體是 [チーズ] 與 [チーズ, 黒カレー]，取前者。
          2. 同樣品項數時，優先取「名稱剛好等於傳票品名」的那個，
             讓畫面上的寫法跟作業員手上的傳票一致（牛丼小 / 牛丼小盛 取前者）。
          3. 還是平手就取 menus_mapping.yaml 裡較前面的那一列（穩定、可重現）。
        """
        def rank(indexed):
            idx, combo = indexed
            distinct = len(set(combo))
            same_as_ticket = 0 if (distinct == 1 and combo[0] == item_name) else 1
            return (distinct, same_as_ticket, idx)

        return min(enumerate(variants), key=rank)[1]

    # ------------------------------------------------------------
    # 傳票修飾列（換湯 / 換飯…）
    # ------------------------------------------------------------
    def slot_count(self, ticket_name: str, axis: str) -> int:
        """這個傳票名稱「每一份」提供幾個該軸的可替換槽位。

        用主變體算，因為主變體就是我們對外顯示、也認定最可能為真的那一版
        （見 pick_primary_variant）。交給 ticket_modifiers.resolve_modifiers
        當錨定的容量上限，所以「牛丼お新香みそ汁セット x2 只換一碗湯」會
        正確地只扣一份。

        單品湯不算槽位：`あさり汁` 的主變體就是 `['あさり汁']`，它本身就是
        那碗湯，不是「附湯的套餐」。不排除的話，傳票上單獨點的一碗湯會變成
        合法的錨定對象，換湯就可能扣掉它而不是套餐裡那碗（實測傳票
        441 號同時有 `しじみ汁変更` 與單品 `あさり汁`，只是順序剛好沒踩到）。
        """
        if not self.modifier_knowledge:
            return 0
        variants = self.menu_variants.get(ticket_name) or [[ticket_name]]
        primary = self.pick_primary_variant(ticket_name, variants)
        slots = sum(1 for sub in primary if self.modifier_knowledge.is_slot(axis, sub))
        if slots and len(primary) <= 1:
            return 0
        return slots

    def _remove_axis_slot(self, counter: Counter, axis: str, amount: int) -> Counter:
        """從一份需求表扣掉 amount 份該軸的槽位，回傳實際扣掉的內容。

        一份組合通常只有一個該軸的成員（套餐不會附兩碗湯）。真的有多個時
        依 Counter 既有的鍵順序扣（= menus_mapping.yaml 的 sub_items 順序），
        讓行為可重現。
        """
        removed: Counter = Counter()
        left = int(amount)
        for sub in list(counter.keys()):
            if left <= 0:
                break
            if not self.modifier_knowledge.is_slot(axis, sub):
                continue
            take = min(left, counter[sub])
            if take <= 0:
                continue
            counter[sub] -= take
            if counter[sub] <= 0:
                del counter[sub]
            removed[sub] += take
            left -= take
        return removed

    def _apply_substitution(self, mod, variant_counters, union, primary_counts) -> Counter:
        """把被換掉的槽位從這一列的三份輸出裡扣掉。

        **三份都扣**，理由各自不同：
          combos（完成判定）  不扣就是等一碗永遠不會來的湯，訂單永遠到不了
                             TRAY_COMPLETED。這是最嚴重的那個。
          display（顯示）     不扣就是客戶回報的「一份餐 + 兩種湯」。
          union（歸屬判定）   2026-09-21 客戶確認：換湯後**原本那碗湯的貼紙
                             不會印**。既然不會印，出現就是異常（例如前一單
                             殘留的貼紙），留在聯集裡等於靜默算成本訂單的一
                             部分 —— 那正是本專案最該避免的失效方式，所以
                             也扣，讓它走 WRONG_ITEM。

        回傳實際被換掉的品項（給 log 與 MQTT 回報用）。
        """
        removed: Counter = Counter()
        for counter in variant_counters:
            removed.update(self._remove_axis_slot(counter, mod.axis, mod.applied))
        self._remove_axis_slot(union, mod.axis, mod.applied)
        self._remove_axis_slot(primary_counts, mod.axis, mod.applied)
        return removed

    @staticmethod
    def describe_mapping(item_name, item_num, primary_counts, union, variants,
                          is_spelling_group, substitutions) -> str:
        """一列傳票品項 -> 送給前端的貼紙（主變體），以及比對時另外接受的寫法"""
        shown = " + ".join(f"{sub}×{n}" for sub, n in primary_counts.items()) or "（無）"
        notes = []
        others = [sub for sub in union if sub not in primary_counts]
        if others:
            if is_spelling_group:
                notes.append(f"同菜異寫也接受 {others}，回報時改名")
            else:
                notes.append(f"共 {len(variants)} 種組合，也接受 {others}")
        for sub in substitutions:
            notes.append(f"替換 {sub['axis']} 槽 {sub['from']}→{sub['to']}")
        return f"{item_name}×{item_num} → {shown}" + (f"（{'；'.join(notes)}）" if notes else "")

    def log_mapping(self, tray_id, order_number, expected_items_list, tag="") -> None:
        """[訂單解析結果] 之後補一行：傳票名稱各自轉成哪些貼紙、實際送前端的 expected_items"""
        logger.info(
            f"[傳票→貼紙]{tag} tray={tray_id} 單號 {order_number}｜"
            + "｜".join(self.last_mapping_lines)
            + f"｜送前端 expected_items={expected_items_list}"
        )

    def expand_parsed_items(self, parsed_items):
        """把 OrderParser 解析出的品項展開成待核對清單。

        回傳四份東西：
          expanded_expected_items  所有變體的**聯集**（攤平成一個個品項）。
                                   只負責「這張貼紙算不算屬於本訂單」的判定，
                                   以及 confirmed 進度的上限 —— 不拿來顯示。
          expected_items_list      前端顯示用的逐行清單 [{"菜單名": 數量}]，
                                   只取「主變體」。
          combos                   組合假設清單 List[Counter]，負責「訂單完成了
                                   沒有」的判定 —— 任一份被滿足就算完成。
          report_alias             {其他寫法: 主變體寫法}，對外回報時改名用。

        為什麼聯集、顯示、假設要分成三份：
          * 4切牛ﾀﾝ 有「含唐辛子 / 含たれ / 都不含」三個版本，傳票讀不到 POS
            代碼所以無法分辨。聯集拿來判斷歸屬（放たれ不會被誤報 WRONG_ITEM），
            假設拿來判斷完成（不會因為等一份永遠不會來的唐辛子而卡住）。
          * 但**顯示不能用聯集**。2026-09-18 現場回報：チーズ×3 的訂單被顯示成
            チーズ×3 + 黒カレー×3、牛丼小 被顯示成 牛丼小盛 + 牛丼小，
            「一份變成兩份」。所以顯示只取主變體。

        純異寫群組（每個變體都只有一個品項，例如 牛丼小 / 牛丼小盛）除了顯示
        取一種，對外回報也統一改成主變體的寫法 —— 否則貼紙讀到另一種寫法時，
        前端會收到一個它沒列出來的品名，進度對不起來。這類群組的組合假設也
        因此收斂成一份。

        2026-09-10：OrderParser 的數量可能是 None（未確認）。
        None 代表「讀不到，不知道」，不是 1 —— 因此：
          * 不展開為待核對項目（無法決定要核對幾份）
          * 仍以 {name: None} 保留在 expected_items_list，讓該列不會整列消失
          * 另外回傳 unconfirmed 清單，供呼叫端發布/告警

        2026-09-21：傳票修飾列（換湯）。`変更しじみ汁` 這種列不是獨立一道菜，
        而是「把上方最近那個套餐的湯換掉」。它自己照常展開（貼紙
        `※変更しじみ` 是真的會印、要核對的），但**額外**會從錨定到的那一列
        扣掉一個湯槽位。錨定與軸的判定在 logic/ticket_modifiers.py，扣除在
        _apply_substitution。沒有 menu_modifiers.yaml 的品牌完全不受影響。
        """
        expanded_expected_items = []
        expected_items_list = []
        unconfirmed = []
        groups: List[List[Counter]] = []
        report_alias: Dict[str, str] = {}

        # 先依傳票順序解析修飾列並完成錨定。parsed_items 的 index 就是傳票的
        # 上下順序（ticket_layout.group_rows 已按 cy 排序），所以這裡不需要
        # 任何 bbox 座標。
        lines, unattached = resolve_modifiers(
            parsed_items, self.modifier_knowledge, self.slot_count
        )
        self.last_substitutions = []
        self.last_unattached_modifiers = [
            {"item": m.ticket_name, "axis": m.axis, "quantity": m.quantity}
            for m in unattached
        ]
        self.last_mapping_lines = []

        for line in lines:
            item_name, item_num = line.name, line.quantity
            if item_num is None:
                unconfirmed.append(item_name)
                expected_items_list.append({item_name: None})
                self.last_mapping_lines.append(f"{item_name}×? → 數量未確認，不核對")
                logger.warning(
                    f"[tray_state_machine] 品名 '{item_name}' 數量未確認，"
                    "不展開為待核對項目（不預設 1）"
                )
                continue

            # 沒登記變體的品項，視為「只有一個變體、內容就是它自己」，
            # 這樣底下的流程不需要為單品項再分岔一次。
            variants = self.menu_variants.get(item_name) or [[item_name]]
            primary = self.pick_primary_variant(item_name, variants)

            # 純異寫群組：同一道菜的不同寫法，統一成主變體的名字。
            #
            # 「每個變體都只有一個品項」只是**推論**，不是事實 —— 例如傳票
            # ご飯増量 的兩個變體是 ['鰻ご飯増量'] 與 ['ご飯増量']，那是鰻魚飯
            # 加量 vs 白飯加量，根本不同的東西，統一改名會把錯的餐點說成對的。
            # 所以 menu_aliases.yaml 的 distinct 台帳在這裡有否決權：人工判定
            # 過「不是同一道菜」的配對，一律不做自動改名。
            is_spelling_group = len(variants) > 1 and all(len(set(c)) == 1 for c in variants)
            if is_spelling_group:
                names = [c[0] for c in variants]
                vetoed = [
                    pair for pair in itertools.combinations(names, 2)
                    if frozenset(pair) in self.menu_distinct_pairs
                ]
                if vetoed:
                    is_spelling_group = False
                    logger.info(
                        f"[tray_state_machine] 品名 '{item_name}' 的變體 {names} "
                        f"因 distinct 台帳判定不同，不做自動改名: {vetoed}"
                    )
            if is_spelling_group:
                target = primary[0]
                for combo in variants:
                    if combo[0] != target:
                        report_alias[combo[0]] = target
                effective_variants = [[target]]
            else:
                effective_variants = variants

            variant_counters = [
                Counter({sub: item_num * count for sub, count in Counter(combo).items()})
                for combo in effective_variants
            ]

            if len(variants) > 1:
                logger.info(
                    f"[tray_state_machine] 品名 '{item_name}' 有 {len(variants)} 個貼紙組合變體，"
                    f"任一變體湊齊即算完成: {variants}；"
                    f"顯示用主變體={primary}"
                    + (f"，回報改名={report_alias}" if is_spelling_group else "")
                )

            # 聯集：保留所有原始寫法（含非主變體），這樣貼紙印哪一種都不會
            # 被誤報 WRONG_ITEM。同一個品項在不同變體裡取需求量較大者。
            union: Counter = Counter()
            for combo in variants:
                for sub, count in Counter(combo).items():
                    union[sub] = max(union[sub], item_num * count)

            # 顯示：只用主變體，避免「一份變成兩份」
            primary_counts: Counter = Counter()
            for sub in primary:
                primary_counts[sub] += item_num

            # 掛在這一列上的修飾（換湯…）：在攤平之前先把被換掉的槽位扣掉，
            # 三份輸出一起扣（理由見 _apply_substitution）。
            subs_before = len(self.last_substitutions)
            for mod in line.modifiers:
                removed = self._apply_substitution(
                    mod, variant_counters, union, primary_counts
                )
                if not removed:
                    logger.warning(
                        f"[tray_state_machine] '{mod.ticket_name}' 錨定到 "
                        f"'{item_name}' 但它的組合裡找不到 {mod.axis} 槽位可扣，"
                        "本次替換未生效"
                    )
                    continue
                for sub, count in removed.items():
                    self.last_substitutions.append({
                        "axis": mod.axis,
                        "anchor": item_name,
                        "from": sub,
                        "to": mod.ticket_name,
                        "quantity": count,
                    })
                logger.info(
                    f"[tray_state_machine] '{item_name}' 依 '{mod.ticket_name}' "
                    f"扣除 {dict(removed)}（軸={mod.axis}）"
                )

            for sub, count in union.items():
                expanded_expected_items.extend([sub] * count)
            for sub, count in primary_counts.items():
                expected_items_list.append({sub: count})

            groups.append(variant_counters)
            self.last_mapping_lines.append(self.describe_mapping(
                item_name, item_num, primary_counts, union, variants, is_spelling_group,
                self.last_substitutions[subs_before:],
            ))

        self.last_unconfirmed_quantity = unconfirmed
        return (expanded_expected_items, expected_items_list,
                self.cartesian_merge(groups), report_alias)

    def recompute_expected_combos(self, tray) -> None:
        """從目前還活著的貢獻者重算整筆訂單的組合假設。

        假設記在貢獻者（tray.ticket / 各 extra_ticket）身上，所以候選票離開時
        只要重算一次就能精準扣掉它的貢獻，不需要做「反向笛卡兒積」這種不可逆
        的運算。
        """
        sources: List[List[Counter]] = []
        if tray.ticket is not None and getattr(tray.ticket, "contributed_combos", None):
            sources.append(tray.ticket.contributed_combos)
        for et in tray.extra_tickets:
            if (et.is_ticket_merged
                    and not et.has_notified_missing
                    and getattr(et, "contributed_combos", None)):
                sources.append(et.contributed_combos)
        tray.expected_combos = self.cartesian_merge(sources)

    def resolve_effective_combo(self, tray, counts) -> Counter:
        """挑出「最能解釋現場狀況」的那一個組合假設，供 CSV / 報表使用。

        訂單完成時就是被湊齊的那一版；沒完成時取缺最少的那一版（最可能為真的
        假設）。CSV 若直接記聯集，多變體訂單會把「其實不會來的品項」也列成
        missing，事後查帳的人會被誤導。
        """
        combos = getattr(tray, "expected_combos", None)
        if not combos:
            return Counter(tray.expected_items)

        def shortfall(combo: Counter) -> int:
            return sum(max(0, need - counts.get(item, 0)) for item, need in combo.items())

        return min(combos, key=shortfall)

    def is_order_satisfied(self, tray, counts) -> bool:
        """訂單是否已滿足 —— 任一個組合假設被湊齊即成立。

        Args:
            counts: {品項: 已確認份數}。呼叫端決定要用 confirmed_counts
                （歷史最高，不因餐點暫時離開而下降）還是目前畫面的 checked。
        """
        combos = getattr(tray, "expected_combos", None)
        if not combos:
            # 沒有假設資訊（舊資料、或品項全部數量未確認）時，退回
            # 「聯集全部湊滿」的舊判斷，行為與改動前一致。
            if not tray.expected_items:
                return False
            return all(
                counts.get(item, 0) >= need
                for item, need in Counter(tray.expected_items).items()
            )
        return any(
            all(counts.get(item, 0) >= need for item, need in combo.items())
            for combo in combos
        )
