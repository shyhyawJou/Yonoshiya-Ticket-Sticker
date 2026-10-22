#!/usr/bin/env python3
"""
verify_menu_modifiers.py
========================
傳票修飾列（logic/ticket_modifiers.py）的稽核與回歸檢查工具。

為什麼需要它
------------
修飾列的認定是**人工台帳**（tasks/ocr/<brand>/menu_modifiers.yaml），而菜單
本身會被工具從客戶 xlsx 整份重產。所以每次重產之後，新增的變更品項如果
沒有人去歸軸，它就會靜默地退回「當成獨立一道菜」—— 前端又會出現
「一份餐 + 兩種湯」，而且訂單到不了 TRAY_COMPLETED。

這支程式就是那個提醒：它用 ja_ticket_grammar.yaml 的文法掃整份菜單，把
「看起來像修飾列、但台帳沒登記也沒結案」的列印成可直接貼進 yaml 的提案。

用法
----
    python3 tools/verify_menu_modifiers.py                     # 檢查所有品牌
    python3 tools/verify_menu_modifiers.py --brand yoshinoya
    python3 tools/verify_menu_modifiers.py --brand yoshinoya --coverage

    # 場景回歸：錨定規則的案例表（純邏輯，不需要相機/DLA）
    python3 tools/verify_menu_modifiers.py --brand yoshinoya --scenarios

    # 模擬一張傳票（用 | 分隔各列，數量用 *N）
    python3 tools/verify_menu_modifiers.py --brand yoshinoya \
        --ticket "牛丼・並|牛丼お新香みそ汁セット|変更しじみ汁"

離開碼
------
    0 = 沒有未歸軸的修飾列，場景回歸也全過
    1 = 有未歸軸的修飾列（要嘛登記進 substitutions，要嘛寫進 not_modifier），
        或台帳指向不存在的軸，或場景回歸有失敗
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

from logic.menu_alias import build_menu_index, load_alias_knowledge  # noqa: E402
from logic.text_utils import normalize_text  # noqa: E402
from logic.ticket_modifiers import (  # noqa: E402
    load_modifier_knowledge, resolve_modifiers,
)

TASKS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks", "ocr"
)
SHARED_READINGS = os.path.join(TASKS_DIR, "ja_readings.yaml")
SHARED_GRAMMAR = os.path.join(TASKS_DIR, "ja_ticket_grammar.yaml")

N = lambda s: normalize_text(s, ignore_space=True)  # noqa: E731


# ---------------------------------------------------------------------------
# 載入：刻意跟 config.load_config 走同一組函式，稽核結果才跟線上行為一致
# ---------------------------------------------------------------------------

class Brand:
    def __init__(self, brand: str):
        self.brand = brand
        self.dir = os.path.join(TASKS_DIR, brand)
        mapping_path = os.path.join(self.dir, "menus_mapping.yaml")
        with open(mapping_path, "r", encoding="utf-8-sig") as fh:
            data = yaml.safe_load(fh) or {}

        ticket_pool, sticker_pool = [], []
        rows = []
        for row in data.get("mapping", []) or []:
            name = str(row["name"])
            subs = [str(s) for s in (row.get("sub_items") or [])]
            ticket_pool.append(name)
            sticker_pool.extend(subs)
            rows.append((name, subs))

        know = load_alias_knowledge(
            SHARED_READINGS, os.path.join(self.dir, "menu_aliases.yaml")
        )
        self.ticket_index = build_menu_index(know, "ticket", ticket_pool)
        self.sticker_index = build_menu_index(
            know, "sticker", sticker_pool, prefer=ticket_pool
        )

        # 與 config.load_config 完全相同的 menu_variants 建法
        self.menu_variants = {}
        for name, subs in rows:
            key = self.ticket_index.canonical_or_self(name)
            combo = [self.sticker_index.canonical_or_self(s) for s in subs]
            combos = self.menu_variants.setdefault(key, [])
            if combo and combo not in combos:
                combos.append(combo)

        self.sticker_names = {self.sticker_index.canonical_or_self(s)
                              for s in sticker_pool}

        self.modifiers = load_modifier_knowledge(
            SHARED_GRAMMAR,
            os.path.join(self.dir, "menu_modifiers.yaml"),
            normalize=N,
            canonical_sticker=self.sticker_index.canonical_or_self,
        )
        # 跟 config.load_config 一樣要跑自動推導，否則稽核報表跟線上行為不一致
        self.manual_rules = set(self.modifiers.rules)
        self.derived = (
            self.modifiers.derive_from_menu(self.menu_variants)
            if self.modifiers.axes else []
        )

    # ------------------------------------------------------------------
    def slot_count(self, ticket_name: str, axis: str) -> int:
        """必須跟 TrayStateMachine._slot_count 一致，否則稽核對不上線上。"""
        from logic.tray_state_machine import TrayStateMachine
        variants = self.menu_variants.get(ticket_name) or [[ticket_name]]
        primary = TrayStateMachine._pick_primary_variant(ticket_name, variants)
        return sum(1 for sub in primary if self.modifiers.is_slot(axis, sub))

    def state_machine(self):
        """只為了呼叫 _expand_parsed_items —— 它不會用到 bus/parser/matcher。"""
        from logic.tray_state_machine import TrayStateMachine

        class _NullBus:
            def publish_det_status(self, payload):
                pass

        return TrayStateMachine(
            bus=_NullBus(),
            order_parser=None,
            sticker_matcher=None,
            menu_variants=self.menu_variants,
            n_settle_frame=1,
            modifier_knowledge=self.modifiers,
        )


# ---------------------------------------------------------------------------
# 檢查
# ---------------------------------------------------------------------------

def check_brand(b: Brand, show_coverage: bool = False) -> int:
    problems = 0
    print(f"\n{'=' * 72}\n品牌: {b.brand}\n{'=' * 72}")

    if not b.modifiers.rules:
        print("  沒有 menu_modifiers.yaml（或台帳為空）→ 傳票修飾功能停用")

    # --- 1. 軸定義 -----------------------------------------------------
    print(f"\n[軸] {len(b.modifiers.axes)} 個")
    for name, axis in b.modifiers.axes.items():
        unknown = sorted(s for s in axis.slots if s not in b.sticker_names)
        print(f"  {name:8} ({axis.display or '-'}) {len(axis.slots)} 個槽位")
        for s in sorted(axis.slots):
            mark = "  ← 菜單裡沒有這個貼紙品名" if s in unknown else ""
            print(f"      {s}{mark}")
        if unknown:
            # 不當錯誤：xlsx 改版可能移掉某個湯。但打錯字也長這樣，所以要印出來。
            print(f"      ! {len(unknown)} 個槽位不在目前菜單的 sub_items 裡"
                  f"（可能是 xlsx 改版移除，也可能是打錯字，請看一眼）")

    # --- 2. 台帳 vs 菜單 ------------------------------------------------
    menu_tickets = set(b.menu_variants)
    registered = set(b.modifiers.rules)
    stale = sorted(registered - menu_tickets)
    print(f"\n[台帳] 人工登記 {len(b.manual_rules)} 筆 + 自動推導 {len(b.derived)} 筆"
          f" = {len(registered)} 筆修飾列；已結案的非修飾列 "
          f"{len(b.modifiers.not_modifier)} 筆")
    by_axis = defaultdict(list)
    for key, rule in b.modifiers.rules.items():
        tag = "" if key in b.manual_rules else " (推導)"
        by_axis[rule.axis].append(key + tag)
    for axis in sorted(by_axis):
        print(f"  {axis}: {', '.join(sorted(by_axis[axis]))}")
    if b.derived:
        print("\n  自動推導的明細（文法命中 + 貼紙帶変更標記且剝掉後落在該軸）:")
        for r in b.derived:
            print(f"      {r.ticket} -> {r.axis}   {r.note}")
        print("      這些不需要寫進 yaml；要覆寫或否決才在 substitutions /")
        print("      not_modifier 明確登記。")
    if stale:
        print(f"\n  台帳有登記、但目前菜單沒有這一列（{len(stale)} 筆，不是錯誤 ——")
        print("  人工判定不會因為 xlsx 改版而失效，保留著下次出現就能用）:")
        for s in stale:
            print(f"      {s}")

    # --- 3. 未歸軸的提案（這是本工具的主要目的）--------------------------
    proposals = []
    # 沒有定義任何軸 = 這個品牌還沒採用傳票修飾機制，不要拿共用文法去吵它
    if b.modifiers.axes:
        for name in sorted(menu_tickets):
            item = b.modifiers.propose(name)
            if item:
                proposals.append((name, item))
    if proposals:
        problems += 1
        print(f"\n  [!] 有 {len(proposals)} 個傳票品名在文法上像修飾列，"
              f"但台帳沒登記也沒結案。")
        print("      它們現在被當成獨立一道菜 —— 如果其實是「換掉上一列的某樣東西」，")
        print("      前端就會多顯示一份，而且訂單可能到不了 TRAY_COMPLETED。")
        print("      確認後把下面的片段貼進 menu_modifiers.yaml 的 substitutions")
        print("      （或確定不是修飾列 -> 貼進 not_modifier，那是結案記錄）:\n")
        for name, item in proposals:
            subs = b.menu_variants.get(name, [[]])[0]
            print(f"  - ticket: {name}")
            print(f"    axis: ???        # 代入的是 '{item}'，貼紙={subs}")
            print(f"    decided_by: ???")
            print(f"    decided_at: ???")
    else:
        print("\n  [ok] 文法命中的修飾列全部都已歸軸或已結案")

    # --- 4. 覆蓋率：哪些套餐提供哪一軸的槽位 -----------------------------
    if show_coverage:
        for axis in b.modifiers.axes:
            providers = [
                (name, b.slot_count(name, axis))
                for name in sorted(menu_tickets)
                if b.slot_count(name, axis) > 0
            ]
            print(f"\n[覆蓋率] 軸 '{axis}': {len(providers)} 個傳票品名提供槽位")
            for name, n in providers:
                print(f"      {name} ({n})")

    return problems


# ---------------------------------------------------------------------------
# 場景回歸：錨定規則的案例表
# ---------------------------------------------------------------------------
SET = "牛丼お新香みそ汁セット"      # sub_items = [お新香, みそ汁]
SET2 = "牛丼キムチみそ汁セット"      # sub_items = [キムチ, みそ汁]
CHG_SHIJIMI = "変更しじみ汁"        # 貼紙 ※変更しじみ
CHG_TON = "変更とん汁"              # 貼紙 ※変更豚汁
CHG_SUFFIX = "しじみ汁変更"         # 2026-09-21 現場加的寫法，靠自動推導認出
PLAIN_SOUP = "しじみ汁"             # 單品湯，不是修飾列
PLAIN_ASARI = "あさり汁"            # 單品湯，且是湯軸成員 -> 不可以當錨
BOWL = "牛丼・並"

SCENARIOS = [
    # (說明, 傳票列 [(品名, 數量)], 檢查函式)
    (
        "套餐 + 換湯：みそ汁 被換成 ※変更しじみ",
        [(BOWL, 1), (SET, 1), (CHG_SHIJIMI, 1)],
        lambda r: (
            r["union"]["みそ汁"] == 0
            and r["display"].get("みそ汁", 0) == 0
            and r["union"]["※変更しじみ"] == 1
            and all(c.get("みそ汁", 0) == 0 for c in r["combos"])
            and r["union"]["お新香"] == 1
        ),
    ),
    (
        "沒有 変更 字樣的單品湯：完全不動作，兩碗湯都在",
        [(SET, 1), (PLAIN_SOUP, 1)],
        lambda r: r["union"]["みそ汁"] == 1 and r["display"].get("みそ汁") == 1,
    ),
    (
        "套餐 x2 只換一碗：みそ汁 剩 1 份",
        [(SET, 2), (CHG_SHIJIMI, 1)],
        lambda r: (
            r["union"]["みそ汁"] == 1
            and r["display"].get("みそ汁") == 1
            and r["union"]["お新香"] == 2
        ),
    ),
    (
        "兩個套餐各換一次：各自錨定最近的上方套餐，兩碗 みそ汁 都被扣掉",
        [(SET, 1), (CHG_SHIJIMI, 1), (SET2, 1), (CHG_TON, 1)],
        lambda r: (
            r["union"]["みそ汁"] == 0
            and r["union"]["お新香"] == 1
            and r["union"]["キムチ"] == 1
        ),
    ),
    (
        "只有換湯列、上方沒有套餐：錨不到，退回當普通品項",
        [(CHG_SHIJIMI, 1)],
        lambda r: r["union"]["※変更しじみ"] == 1 and r["unattached"] == 1,
    ),
    (
        "換湯列印在套餐上方：規則只往上找，所以錨不到（不亂猜）",
        [(CHG_SHIJIMI, 1), (SET, 1)],
        lambda r: r["union"]["みそ汁"] == 1 and r["unattached"] == 1,
    ),
    (
        "一個套餐被換兩次：第二次沒有槽位可用，錨不到",
        [(SET, 1), (CHG_SHIJIMI, 1), (CHG_TON, 1)],
        lambda r: r["union"]["みそ汁"] == 0 and r["unattached"] == 1,
    ),
    (
        "錨定會跳過沒有湯槽的列（牛丼單品），錨到更上方的套餐",
        [(SET, 1), (BOWL, 1), (CHG_SHIJIMI, 1)],
        lambda r: r["union"]["みそ汁"] == 0 and r["unattached"] == 0,
    ),
    (
        "數量未確認的換湯列：不預設 1，不套用替換",
        [(SET, 1), (CHG_SHIJIMI, None)],
        lambda r: r["union"]["みそ汁"] == 1,
    ),
    (
        "後綴寫法 'しじみ汁変更'（台帳沒登記，靠自動推導）也要生效",
        [(SET, 1), (CHG_SUFFIX, 1)],
        lambda r: (
            r["union"]["みそ汁"] == 0
            and r["union"]["※変更しじみ"] == 1
            and r["unattached"] == 0
        ),
    ),
    (
        "單品湯不可以當錨：錨定要跳過 あさり汁，扣套餐的 みそ汁",
        [(SET, 1), (PLAIN_ASARI, 1), (CHG_SHIJIMI, 1)],
        lambda r: (
            r["union"]["みそ汁"] == 0
            and r["union"]["あさり汁"] == 1      # 單獨點的那碗湯不可以被扣掉
            and r["unattached"] == 0
        ),
    ),
    (
        "現場傳票 441：套餐 + しじみ汁変更 + 單品あさり汁",
        [(SET, 1), (CHG_SUFFIX, 1), (PLAIN_ASARI, 1)],
        lambda r: (
            r["union"]["みそ汁"] == 0
            and r["union"]["※変更しじみ"] == 1
            and r["union"]["あさり汁"] == 1
            and r["display"].get("みそ汁", 0) == 0
        ),
    ),
]


def run_scenarios(b: Brand) -> int:
    sm = b.state_machine()
    failed = 0
    print(f"\n{'=' * 72}\n場景回歸（錨定規則）\n{'=' * 72}")
    for desc, items, check in SCENARIOS:
        norm_items = [(N(n), q) for n, q in items]
        missing = [n for n, _ in norm_items if n not in b.menu_variants]
        if missing:
            print(f"  [skip] {desc}\n         菜單裡沒有: {missing}")
            continue
        union, display, combos, _alias = sm._expand_parsed_items(norm_items)
        result = {
            "union": Counter(union),
            "display": Counter({k: v for d in display for k, v in d.items()
                                if v is not None}),
            "combos": combos,
            "unattached": len(sm.last_unattached_modifiers),
        }
        ok = False
        try:
            ok = bool(check(result))
        except Exception as exc:
            print(f"  [error] {desc}: {exc}")
        status = "ok  " if ok else "FAIL"
        print(f"  [{status}] {desc}")
        if not ok:
            failed += 1
            print(f"         傳票: {items}")
            print(f"         聯集: {dict(result['union'])}")
            print(f"         顯示: {display}")
            print(f"         假設: {[dict(c) for c in combos]}")
            print(f"         錨不到: {sm.last_unattached_modifiers}")
    print(f"\n  {len(SCENARIOS) - failed}/{len(SCENARIOS)} 通過")
    return failed


def simulate(b: Brand, spec: str) -> None:
    """--ticket "A|B*2|C"：模擬一張傳票，印出三份輸出。"""
    items = []
    for part in spec.split("|"):
        part = part.strip()
        if not part:
            continue
        qty = 1
        if "*" in part:
            part, _, q = part.rpartition("*")
            qty = int(q)
        items.append((N(part), qty))

    print(f"\n{'=' * 72}\n模擬傳票\n{'=' * 72}")
    for i, (name, qty) in enumerate(items):
        known = "" if name in b.menu_variants else "   ← 菜單裡沒有這個傳票品名"
        axis = b.modifiers.axis_of(name)
        tag = f"   [修飾列 軸={axis}]" if axis else ""
        print(f"  {i}: {name} x{qty}{tag}{known}")

    lines, unattached = resolve_modifiers(items, b.modifiers, b.slot_count)
    sm = b.state_machine()
    union, display, combos, alias = sm._expand_parsed_items(items)

    print("\n  錨定結果:")
    for line in lines:
        for mod in line.modifiers:
            print(f"    '{mod.ticket_name}' x{mod.quantity} -> 第 {line.row_index} 列 "
                  f"'{line.name}'（扣 {mod.applied} 份 {mod.axis}）")
    for mod in unattached:
        print(f"    '{mod.ticket_name}' x{mod.quantity} -> 錨不到，退回普通品項")

    print(f"\n  替換明細: {sm.last_substitutions}")
    print(f"  顯示清單 (前端): {display}")
    print(f"  聯集 (歸屬判定): {dict(Counter(union))}")
    print(f"  組合假設 (完成判定): {[dict(c) for c in combos]}")
    if alias:
        print(f"  回報改名: {alias}")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="傳票修飾列台帳的稽核工具")
    ap.add_argument("--brand", help="只檢查一個品牌（預設全部）")
    ap.add_argument("--coverage", action="store_true",
                    help="列出每個軸有哪些傳票品名提供槽位")
    ap.add_argument("--scenarios", action="store_true",
                    help="跑錨定規則的場景回歸")
    ap.add_argument("--ticket", help='模擬一張傳票，例如 "A|B*2|C"')
    args = ap.parse_args()

    if args.brand:
        brands = [args.brand]
    else:
        brands = sorted(
            d for d in os.listdir(TASKS_DIR)
            if os.path.isfile(os.path.join(TASKS_DIR, d, "menus_mapping.yaml"))
        )

    problems = 0
    for name in brands:
        b = Brand(name)
        problems += check_brand(b, show_coverage=args.coverage)
        if args.scenarios:
            problems += run_scenarios(b)
        if args.ticket:
            simulate(b, args.ticket)

    print()
    if problems:
        print(f"[結果] 有 {problems} 項需要處理（見上方 [!] / FAIL）")
        return 1
    print("[結果] 全部通過")
    return 0


if __name__ == "__main__":
    sys.exit(main())
