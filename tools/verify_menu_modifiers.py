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
    ModifierKnowledge, apply_set_overrides, load_modifier_knowledge, resolve_modifiers,
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

        rows = [
            (str(row["name"]), [str(s) for s in (row.get("sub_items") or [])])
            for row in data.get("mapping", []) or []
        ]
        # 與 config.load_config 相同：不印貼紙的套餐（label_less_sets）與人工補的套餐（extra_sets）
        self.raw_names = {N(n) for n, _ in rows}
        rows, self.extra_sets = apply_set_overrides(
            rows, os.path.join(self.dir, "menu_modifiers.yaml"), normalize=N,
        )
        ticket_pool = [n for n, _ in rows]
        sticker_pool = [s for _, subs in rows for s in subs]

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
            if (combo or name in self.extra_sets) and combo not in combos:
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
        self.takeout = self.modifiers.derive_takeout_rows(self.menu_variants)

    # ------------------------------------------------------------------
    def slot_count(self, ticket_name: str, axis: str) -> int:
        """必須跟 OrderRequirements.slot_count 一致，否則稽核對不上線上。"""
        from logic.order_requirements import OrderRequirements
        variants = self.menu_variants.get(ticket_name) or [[ticket_name]]
        primary = OrderRequirements.pick_primary_variant(ticket_name, variants)
        slots = sum(1 for sub in primary if self.modifiers.is_removable(axis, sub))
        return 0 if (slots and len(primary) <= 1) else slots   # 單點的湯不算槽位

    def state_machine(self):
        """只為了呼叫 requirements.expand_parsed_items —— 它不會用到 bus/parser/matcher。"""
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
    takeout_axis = ModifierKnowledge.is_takeout_axis
    axes = {n: a for n, a in b.modifiers.axes.items() if not takeout_axis(n)}
    print(f"\n[軸] {len(axes)} 個（持帰り列自動產生的另外列在下面）")
    for name, axis in axes.items():
        unknown = sorted(s for s in axis.slots if s not in b.sticker_names)
        print(f"  {name:8} ({axis.display or '-'}) {len(axis.slots)} 個槽位，被刪的{b.modifiers.removed_mode(name)}")
        for s in sorted(axis.slots):
            mark = "  ← 菜單裡沒有這個貼紙品名" if s in unknown else ""
            print(f"      {s}{mark}")
        if unknown:
            # 不當錯誤：xlsx 改版可能移掉某個湯。但打錯字也長這樣，所以要印出來。
            print(f"      ! {len(unknown)} 個槽位不在目前菜單的 sub_items 裡"
                  f"（可能是 xlsx 改版移除，也可能是打錯字，請看一眼）")

    # --- 2. 台帳 vs 菜單 ------------------------------------------------
    menu_tickets = set(b.menu_variants)
    registered = {k for k, r in b.modifiers.rules.items() if not takeout_axis(r.axis)}
    stale = sorted(registered - menu_tickets)
    print(f"\n[台帳] 人工登記 {len(b.manual_rules)} 筆 + 自動推導 {len(b.derived)} 筆"
          f" = {len(registered)} 筆修飾列；已結案的非修飾列 "
          f"{len(b.modifiers.not_modifier)} 筆")
    by_axis = defaultdict(list)
    for key, rule in b.modifiers.rules.items():
        if takeout_axis(rule.axis):
            continue
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

    # --- 4. 持帰り列：每一列能扣哪些套餐 ---------------------------------
    if b.modifiers.takeout:
        print(f"\n[持帰り列] {len(b.takeout)} 筆（標記 {list(b.modifiers.takeout.markers)}，"
              f"錨定={b.modifiers.takeout.anchor}）")
        for rule in b.takeout:
            providers = [n for n in sorted(menu_tickets) if b.slot_count(n, rule.axis) > 0]
            stickers = "、".join(sorted(b.modifiers.slot_members(rule.axis)))
            print(f"  {rule.ticket}  貼紙={stickers}  可扣的套餐 {len(providers)} 個")
            for n in providers if show_coverage else providers[:6]:
                print(f"      {n}")
            if not show_coverage and len(providers) > 6:
                print(f"      …（--coverage 列出全部）")
            if not providers:
                # 不算錯誤：這種外帶品項只會被單點。但寫法跟套餐的 sub_items 不一致時
                # （たれ / だれ）也長這樣，所以印出來讓人看一眼。
                print("      （沒有任何套餐附這個貼紙：只會被當成加點。若套餐其實有附，"
                      "多半是寫法不一致，要在 menu_aliases.yaml 補等價）")

    # --- 5. 不印貼紙的套餐 ------------------------------------------------
    label_less = sorted(n for n in b.extra_sets)
    if label_less:
        print(f"\n[不印貼紙的套餐] {len(label_less)} 筆（label_less_sets / extra_sets）")
        for n in label_less:
            src = "客戶菜單已有，貼紙清空" if N(n) in b.raw_names else "客戶菜單沒有，補上"
            print(f"  {n}  （{src}；套餐判定={'是' if b.modifiers.is_set(n) else '否 ← size 列不會歸給它'}）")

    # --- 6. 覆蓋率：哪些套餐提供哪一軸的槽位 -----------------------------
    if show_coverage:
        for axis in axes:
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
PLAIN_MISO = "みそ汁"               # 單點的味噌湯 -> 不可以被刪
SET_TON = "牛丼お新香とん汁ｾｯﾄ"     # sub_items = [お新香, 豚汁]：預設湯不是 みそ汁
SET_POTATO = "ｾｯﾄポテトみそ汁"      # sub_items = [ポテトサラダ, みそ汁]
DRESSING = "ごまドレッシング"
CHG_ASARI = "変更あさり汁"          # 貼紙 ※変更あさり
BOWL = "牛丼・並"
# 套餐的主餐 size 印在下一列（set_size_rows）
SET_CHEESE = "D)ﾁｰｽﾞ牛丼ｻﾗﾀﾞS"       # サラダ、みそ汁
SET_NEGITAMA = "D)ねぎ玉牛丼ｻﾗﾀﾞS"   # サラダ、みそ汁、玉子
SET_MENTAI = "D)明高ﾏﾖ牛丼ｻﾗﾀﾞS"    # サラダ、みそ汁
SET_PAIR = "D)牛からペアｾｯﾄ"        # label_less_sets：客戶確認本身不印貼紙
SET_PARTY = "D)牛からﾊﾟｰﾃｨｾｯﾄ"       # label_less_sets
NAMI2 = "D)牛丼 並 ２個セット"       # label_less_sets，下面接兩列 並
NAMI3 = "D)牛丼 並 ３個セット"       # label_less_sets，下面接三列 並
KARAAGE6 = "から揚げ６個"        # 傳票印 唐揚げ６個，等價層折成 から揚げ6個
# 持帰り列（takeout_rows）
TOJI = "とじ丼 並(玉子)"            # 月見牛とじ丼 並、玉子
EGG = "玉子"                        # 單品玉子
EGG_TO = "玉子(持帰り)"             # 外帶時另外印的那一列，跟套餐的玉子是同一張貼紙
NINNIKU = "ﾆﾝﾆｸ唐揚超特"            # …、ﾆﾝﾆｸﾏｼﾏｼだれ、玉子
NINNIKU_TO = "ニンニクマシマシたれ(持帰り"   # 截斷的名稱，貼紙寫成 たれ（menu_aliases.yaml 補等價）
OMU = "D)オム･牛ハヤシ"             # 牛オムハヤシライス、オム玉子
OMU_TO = "TPオム玉子(持帰り)"       # 貼紙 TPオム玉子（menu_aliases.yaml 補等價）

# 2026-10-28 起湯軸的設定（menu_modifiers.yaml）：anchor=anywhere、remove_slots=[みそ汁]。
# 場景的第 4 欄可以暫時改掉軸設定（例如 show_removed），只在該場景有效。
# 沒寫第 4 欄的場景一律用 show_removed: false 跑（BASE_KNOBS）：它們驗的是錨定與扣除，
# 「被扣的那碗從聯集消失」是最嚴格的檢查。yaml 實際用的 optional 另有專門的場景。
BASE_KNOBS = {"show_removed": False, "optional_removed": False}
OPTIONAL = {"show_removed": False, "optional_removed": True}

SCENARIOS = [
    # (說明, 傳票列 [(品名, 數量)], 檢查函式)
    (
        "size 列：チーズ牛丼套餐下面的 並 -> 只留 チーズ牛丼並盛",
        [(SET_CHEESE, 1), ("並", 1)],
        lambda r: (
            r["union"]["チーズ牛丼並盛"] == 1 and r["union"]["牛丼並"] == 0
            and len(r["combos"]) == 1 and r["display"].get("チーズ牛丼並盛") == 1
        ),
    ),
    (
        "size 列：ねぎ玉牛丼套餐下面的 並 -> 牛丼並 + ねぎ（玉子 在套餐裡）",
        [(SET_NEGITAMA, 1), ("並", 1)],
        lambda r: (
            r["union"]["牛丼並"] == 1 and r["union"]["ねぎ"] == 1 and r["union"]["玉子"] == 1
            and r["union"]["チーズ牛丼並盛"] == 0 and len(r["combos"]) == 1
        ),
    ),
    (
        "size 列：明太高菜マヨ牛丼套餐下面的 大 -> 明太高菜マヨ牛丼大 + 明太高菜マヨ",
        [(SET_MENTAI, 1), ("大", 1)],
        lambda r: (
            r["union"]["明太高菜マヨ牛丼大"] == 1 and r["union"]["明太高菜マヨ"] == 1
            and r["union"]["牛丼大"] == 0 and len(r["combos"]) == 1
        ),
    ),
    (
        "size 列：一般套餐（牛丼お新香みそ汁セット）下面的 並 -> 一般牛丼",
        [(SET, 1), ("並", 1)],
        lambda r: r["union"]["牛丼並"] == 1 and r["union"]["チーズ牛丼並盛"] == 0 and len(r["combos"]) == 1,
    ),
    (
        "現場傳票 319：D)牛からペアｾｯﾄ（不印貼紙）+ 並 + 並 + 唐揚げ６個 -> 牛丼並 x2",
        [(SET_PAIR, 1), ("並", 1), ("並", 1), (KARAAGE6, 1)],
        lambda r: (
            r["union"]["牛丼並"] == 2 and r["union"]["チーズ牛丼並盛"] == 0
            and len(r["combos"]) == 1 and N(SET_PAIR) not in r["union"]
        ),
    ),
    (
        "不印貼紙：D)牛丼 並 ２個セット + 並 + 並 -> 只要 牛丼並 x2，套餐本身不要求貼紙",
        [(NAMI2, 1), ("並", 1), ("並", 1)],
        lambda r: (
            r["union"]["牛丼並"] == 2 and sum(r["union"].values()) == 2
            and len(r["combos"]) == 1 and r["combos"][0] == Counter({"牛丼並": 2})
        ),
    ),
    (
        "不印貼紙：D)牛丼 並 ３個セット + 並 x3 列 -> 牛丼並 x3",
        [(NAMI3, 1), ("並", 1), ("並", 1), ("並", 1)],
        lambda r: r["union"]["牛丼並"] == 3 and sum(r["union"].values()) == 3 and len(r["combos"]) == 1,
    ),
    (
        "不印貼紙：D)牛からﾊﾟｰﾃｨｾｯﾄ + 並 + 並 + 唐揚げ６個 -> 牛丼並 x2 + 唐揚げ",
        [(SET_PARTY, 1), ("並", 1), ("並", 1), (KARAAGE6, 1)],
        lambda r: (
            r["union"]["牛丼並"] == 2 and r["union"]["チーズ牛丼並盛"] == 0
            and N(SET_PARTY) not in r["union"] and sum(r["union"].values()) == 3
        ),
    ),
    (
        "持帰り：とじ丼 並(玉子) + 玉子(持帰り) -> 玉子 只要 1 顆（同一張貼紙）",
        [(TOJI, 1), (EGG_TO, 1)],
        lambda r: (
            r["union"]["玉子"] == 1 and r["display"].get("玉子") == 1
            and all(c.get("玉子", 0) == 1 for c in r["combos"])
            and r["union"][r["c"]("月見牛とじ丼 並")] == 1
            and r["unattached"] == 0 and r["subs"]
        ),
    ),
    (
        "內用：とじ丼 並(玉子) 沒有持帰り列 -> 玉子 照常要核對",
        [(TOJI, 1)],
        lambda r: r["union"]["玉子"] == 1 and all(c.get("玉子", 0) == 1 for c in r["combos"]),
    ),
    (
        "持帰り列印在套餐上方：往下找，照樣只要 1 顆",
        [(EGG_TO, 1), (TOJI, 1)],
        lambda r: r["union"]["玉子"] == 1 and r["unattached"] == 0,
    ),
    (
        "外帶套餐 x2 + 玉子(持帰り) x2 -> 玉子 2 顆",
        [(TOJI, 2), (EGG_TO, 2)],
        lambda r: r["union"]["玉子"] == 2 and all(c.get("玉子", 0) == 2 for c in r["combos"]),
    ),
    (
        "單點外帶 玉子(持帰り)，沒有套餐 -> 照常核對、不回報錨不到",
        [(EGG_TO, 1)],
        lambda r: r["union"]["玉子"] == 1 and r["unattached"] == 0,
    ),
    (
        "單品 玉子 不當錨：玉子 + 玉子(持帰り) -> 兩顆都要",
        [(EGG, 1), (EGG_TO, 1)],
        lambda r: r["union"]["玉子"] == 2 and r["unattached"] == 0,
    ),
    (
        "持帰り扣的是自己那張貼紙：ﾆﾝﾆｸ唐揚超特 + ニンニクマシマシたれ(持帰り -> だれ 1、玉子 照常",
        [(NINNIKU, 1), (NINNIKU_TO, 1)],
        lambda r: (
            r["union"][r["c"]("ﾆﾝﾆｸﾏｼﾏｼだれ")] == 1 and r["union"]["玉子"] == 1
            and r["unattached"] == 0
        ),
    ),
    (
        "TPオム玉子(持帰り) 扣 D)オム･牛ハヤシ 附的 オム玉子（等價）-> 只要 1",
        [(OMU, 1), (OMU_TO, 1)],
        lambda r: r["union"][r["c"]("オム玉子")] == 1 and r["unattached"] == 0,
    ),
    (
        "持帰り + 換湯在同一張傳票：互不干擾",
        [(SET, 1), (TOJI, 1), (EGG_TO, 1), (CHG_SHIJIMI, 1)],
        lambda r: (
            r["union"]["みそ汁"] == 0 and r["union"]["玉子"] == 1
            and r["union"]["※変更しじみ"] == 1 and r["unattached"] == 0
        ),
    ),
    (
        "size 列跟套餐之間隔了持帰り列：跳過，照樣認得套餐",
        [(SET_CHEESE, 1), (EGG_TO, 1), ("並", 1)],
        lambda r: r["union"]["チーズ牛丼並盛"] == 1 and r["union"]["牛丼並"] == 0 and len(r["combos"]) == 1,
    ),
    (
        "size 列上方沒有套餐：維持原本的一對多（4 種主餐任一）",
        [("並", 1)],
        lambda r: len(r["combos"]) == 4,
    ),
    (
        "size 列跟套餐之間隔了普通品項：跳過普通品項，照樣認得上方最近的套餐（2026-11 起）",
        [(SET_CHEESE, 1), (KARAAGE6, 1), ("並", 1)],
        lambda r: r["union"]["チーズ牛丼並盛"] == 1 and r["union"]["牛丼並"] == 0 and len(r["combos"]) == 1,
    ),
    (
        "size 列跟套餐之間隔了換湯列：跳過修飾列，照樣認得套餐",
        [(SET_CHEESE, 1), (CHG_SHIJIMI, 1), ("並", 1)],
        lambda r: r["union"]["チーズ牛丼並盛"] == 1 and r["union"]["みそ汁"] == 0 and len(r["combos"]) == 1,
    ),
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
        "換湯列印在套餐上方：上面沒有套餐就往下找，みそ汁 照樣刪（2026-10-28 起）",
        [(CHG_SHIJIMI, 1), (SET, 1)],
        lambda r: r["union"]["みそ汁"] == 0 and r["unattached"] == 0,
    ),
    (
        "變更品項全部印在最下面：兩個套餐、兩個換湯列都在最後，兩碗 みそ汁 都刪",
        [(SET, 1), (SET2, 1), (BOWL, 1), (CHG_SHIJIMI, 1), (CHG_TON, 1)],
        lambda r: (
            r["union"]["みそ汁"] == 0
            and r["display"].get("みそ汁", 0) == 0
            and r["union"]["※変更しじみ"] == 1
            and r["unattached"] == 0
        ),
    ),
    (
        "現場傳票 853：3 個 ｾｯﾄポテトみそ汁 + 最下面 変更とん汁 x1、変更あさり汁 x2，三碗 みそ汁 全刪",
        [(SET_POTATO, 1), (DRESSING, 1), (SET_POTATO, 1), (DRESSING, 1), (SET_POTATO, 1), (DRESSING, 1),
         (CHG_TON, 1), (CHG_ASARI, 2)],
        lambda r: (
            r["union"]["みそ汁"] == 0
            and all(c.get("みそ汁", 0) == 0 for c in r["combos"])
            and r["union"]["※変更あさり"] == 2
            and r["unattached"] == 0
        ),
    ),
    (
        "換湯份數比套餐多：変更あさり汁 x2 只有一個套餐，換一碗、剩 1 份回報錨不到",
        [(SET, 1), (CHG_ASARI, 2)],
        lambda r: r["union"]["みそ汁"] == 0 and r["unattached"] == 1,
    ),
    (
        "預設湯是 豚汁 的套餐：不在 remove_slots 裡，看到換湯不刪、錨不到",
        [(SET_TON, 1), (CHG_SHIJIMI, 1)],
        lambda r: r["union"]["豚汁"] == 1 and r["unattached"] == 1,
    ),
    (
        "豚汁套餐 + みそ汁套餐 + 換湯：只刪 みそ汁，豚汁 不動",
        [(SET_TON, 1), (SET, 1), (CHG_SHIJIMI, 1)],
        lambda r: r["union"]["みそ汁"] == 0 and r["union"]["豚汁"] == 1 and r["unattached"] == 0,
    ),
    (
        "單點的 みそ汁 不可以被刪：只有單點湯 + 換湯時錨不到",
        [(PLAIN_MISO, 1), (CHG_SHIJIMI, 1)],
        lambda r: r["union"]["みそ汁"] == 1 and r["unattached"] == 1,
    ),
    (
        "show_removed=true：みそ汁 照樣顯示、貼了算 OK，但不算完成條件",
        [(SET, 1), (CHG_SHIJIMI, 1)],
        lambda r: (
            r["union"]["みそ汁"] == 1                                  # 貼了不報錯誤品項
            and r["display"].get("みそ汁") == 1                        # 前端照樣列出
            and all(c.get("みそ汁", 0) == 0 for c in r["combos"])      # 不算完成條件
            and r["subs"] and all(s["displayed"] for s in r["subs"])
        ),
        {"show_removed": True, "optional_removed": False},
    ),
    (
        "show_removed=optional：前端不列 みそ汁、不算完成條件，但貼了也收",
        [(SET, 1), (CHG_SHIJIMI, 1)],
        lambda r: (
            r["union"]["みそ汁"] == 1                                  # 貼了不報錯誤品項
            and r["display"].get("みそ汁", 0) == 0                     # 前端不列
            and all(c.get("みそ汁", 0) == 0 for c in r["combos"])      # 不算完成條件
            and r["optional"] == Counter({"みそ汁": 1})
            and r["subs"] and all(s["optional"] and not s["displayed"] for s in r["subs"])
            and r["union"]["※変更しじみ"] == 1
        ),
        OPTIONAL,
    ),
    (
        "optional + 另外單點 みそ汁：清單 みそ汁 ×1（單點那碗）、最多收 2 張、完成要 1 張",
        [(SET, 1), (CHG_SHIJIMI, 1), (PLAIN_MISO, 1)],
        lambda r: (
            r["union"]["みそ汁"] == 2 and r["display"].get("みそ汁") == 1
            and all(c.get("みそ汁", 0) == 1 for c in r["combos"])
            and r["optional"] == Counter({"みそ汁": 1})
        ),
        OPTIONAL,
    ),
    (
        "optional + 套餐 x2 只換一碗：清單 みそ汁 ×1、最多收 2 張、完成要 1 張",
        [(SET, 2), (CHG_SHIJIMI, 1)],
        lambda r: (
            r["union"]["みそ汁"] == 2 and r["display"].get("みそ汁") == 1
            and all(c.get("みそ汁", 0) == 1 for c in r["combos"])
        ),
        OPTIONAL,
    ),
    (
        "optional 不影響持帰り列：とじ丼 並(玉子) + 玉子(持帰り) 照樣只收 1 顆玉子",
        [(TOJI, 1), (EGG_TO, 1)],
        lambda r: r["union"]["玉子"] == 1 and not r["optional"],
        OPTIONAL,
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
    import dataclasses
    know = sm.requirements.modifier_knowledge
    for desc, items, check, *knobs in SCENARIOS:
        norm_items = [(N(n), q) for n, q in items]
        missing = [n for n, _ in norm_items if n not in b.menu_variants]
        if missing:
            print(f"  [skip] {desc}\n         菜單裡沒有: {missing}")
            continue
        saved = dict(know.axes)
        if "soup" in know.axes:   # 暫時改湯軸設定，只在這個場景有效
            know.axes["soup"] = dataclasses.replace(know.axes["soup"], **(knobs[0] if knobs else BASE_KNOBS))
        try:
            union, display, combos, _alias = sm.requirements.expand_parsed_items(norm_items)
        finally:
            know.axes.clear(); know.axes.update(saved)
        result = {
            "union": Counter(union),
            "display": Counter({k: v for d in display for k, v in d.items()
                                if v is not None}),
            "combos": combos,
            "unattached": len(sm.requirements.last_unattached_modifiers),
            "subs": list(sm.requirements.last_substitutions),
            "optional": Counter(sm.requirements.last_optional_items),
            "c": b.sticker_index.canonical_or_self,   # 貼紙寫法 -> 代表菜名（checks 用）
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
            print(f"         錨不到: {sm.requirements.last_unattached_modifiers}")
    print(f"\n  {len(SCENARIOS) - failed}/{len(SCENARIOS)} 通過")
    return failed + run_report_checks()


def run_report_checks() -> int:
    """ITEM_CHECKED / TRAY_COMPLETED 的 items（OrderRequirements.report_checked_items）。

    show_removed: optional 多貼的 みそ汁 不能送給前端（不在清單上）；單點的那碗照常報。
    """
    from types import SimpleNamespace
    from logic.order_requirements import OrderRequirements as R

    def tray(display, optional, confirmed):
        return SimpleNamespace(expected_items_display=display, optional_items=optional,
                               confirmed_counts=confirmed)

    cases = [
        ("只換湯、多貼了 みそ汁：items 不含 みそ汁",
         tray([{"お新香": 1}, {"※変更しじみ": 1}], {"みそ汁": 1}, {"お新香": 1, "みそ汁": 1, "※変更しじみ": 1}),
         [{"お新香": 1}, {"※変更しじみ": 1}]),
        ("另外單點 みそ汁、貼了兩張：只報清單上的 1",
         tray([{"※変更しじみ": 1}, {"みそ汁": 1}], {"みそ汁": 1}, {"みそ汁": 2, "※変更しじみ": 1}),
         [{"みそ汁": 1}, {"※変更しじみ": 1}]),
        ("沒有 optional 品項：跟以前一樣照 confirmed_counts",
         tray([{"牛丼並": 1}], {}, {"牛丼並": 1, "たれ": 1}),
         [{"牛丼並": 1}, {"たれ": 1}]),
    ]
    failed = 0
    print(f"\n{'=' * 72}\nITEM_CHECKED 回報\n{'=' * 72}")
    for desc, t, want in cases:
        got = R.report_checked_items(t)
        ok = got == want
        failed += 0 if ok else 1
        print(f"  [{'ok  ' if ok else 'FAIL'}] {desc}" + ("" if ok else f"\n         得到 {got}，應為 {want}"))
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
    union, display, combos, alias = sm.requirements.expand_parsed_items(items)

    print("\n  錨定結果:")
    for line in lines:
        for mod in line.modifiers:
            print(f"    '{mod.ticket_name}' x{mod.quantity} -> 第 {line.row_index} 列 "
                  f"'{line.name}'（扣 {mod.applied} 份 {mod.axis}）")
    for mod in unattached:
        print(f"    '{mod.ticket_name}' x{mod.quantity} -> 錨不到，退回普通品項")

    print(f"\n  替換明細: {sm.requirements.last_substitutions}")
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
