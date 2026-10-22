#!/usr/bin/env python3
"""
simulate_sticker_check.py
=========================
用「真實的 config + 真實的比對層 + 真實的狀態機」跑貼紙核對情境，
不需要相機、DLA 或 OCR —— 直接餵入假的 OCR 文字。

這支程式驗證的是「同菜異寫」與「同名多組合」兩件事的端到端行為：
    貼紙印假名 / 印漢字 / yaml 只收錄一種寫法 / 數量不同 / 套餐組合有多版

它不是 unit_test/ 的一部分（那個目錄不是維護中的測試套件）。
這裡是一支可以重複執行的情境檢查，改動比對層或菜單之後跑一次。

用法：
    python3 tools/simulate_sticker_check.py
    python3 tools/simulate_sticker_check.py --brand yoshinoya -v

離開碼：0 = 全部情境符合預期，1 = 有情境不符
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from logic.models import TrackedItem, Tray, TrayState  # noqa: E402
from logic.sticker_matcher import StickerMatcher  # noqa: E402
from logic.text_utils import FuzzyMatcher  # noqa: E402
from logic.tray_state_machine import TrayStateMachine  # noqa: E402


class FakeBus:
    """只收集事件，不真的連 MQTT。"""

    def __init__(self):
        self.events = []

    def publish_det_status(self, payload):
        self.events.append(payload)

    def statuses(self):
        return [e.get("status") for e in self.events]


def build(brand: str, legacy: bool = False):
    from config import load_config
    from profile_config import select_profile

    from logic.text_utils import normalize_text

    # select_profile 只回傳 (品牌, config 路徑)：runtime.local.yaml 只選品牌，
    # 不再是第二份設定來源，所以這裡也不需要 merge 任何 override。
    _, config_path = select_profile(ROOT, explicit=brand)
    cfg = load_config(str(config_path))

    menus_sticker = cfg.menus_sticker
    menu_variants = cfg.menu_variants
    menu_index = cfg.sticker_index

    if legacy:
        # 重現改動前的行為，用來確認這些情境原本真的是壞的：
        #   1. 沒有等價層（同菜異寫完全靠模糊比對硬碰）
        #   2. 候選是 460 筆各種寫法混在一起，而不是折疊後的代表菜名
        #   3. 傳票名稱 -> 貼紙組合是 dict，同名後者覆蓋前者（28 組變體只剩最後一列）
        menu_index = None
        menus_sticker = sorted({
            normalize_text(sub, ignore_space=True)
            for mc in cfg.menus_mapping for sub in mc.sub_items
        })
        menu_variants = {}
        for mc in cfg.menus_mapping:
            menu_variants[normalize_text(mc.name, ignore_space=True)] = [
                [normalize_text(sub, ignore_space=True) for sub in mc.sub_items]
            ]

    matcher = StickerMatcher(
        menus_sticker=menus_sticker,
        fuzzy_matcher=FuzzyMatcher(
            special_prefixes=["ねぎ"], prefix_len=2,
            special_score_threshold=35, default_score_threshold=40,
        ),
        vocab_sticker=cfg.vocab_sticker or None,
        menu_index=menu_index,
    )
    bus = FakeBus()
    machine = TrayStateMachine(
        bus=bus,
        order_parser=None,           # 本情境直接餵入已解析的品項，不跑傳票 OCR
        sticker_matcher=matcher,
        menu_variants=menu_variants,
        # 必須跟 logic_engine 的接線一致，否則測不到 distinct 台帳的否決行為
        menu_distinct=None if legacy else getattr(cfg, "menu_distinct", None),
        n_settle_frame=6,
        object_class_items=cfg.object_class_items,
    )
    return cfg, matcher, machine, bus


def run_display_check(resolve_ticket, machine, ticket_text, qty, want_display):
    """只看「傳票解析完要推給前端的清單」，不放貼紙。

    2026-09-18 現場回報：聯集被當成顯示清單，チーズ×3 被顯示成
    チーズ×3 + 黒カレー×3、牛丼小 被顯示成 牛丼小盛 + 牛丼小 —— 一份變兩份。
    這裡鎖住「顯示只取主變體」的行為。
    """
    ticket_name = resolve_ticket(ticket_text)
    _, display, _, alias = machine._expand_parsed_items([(ticket_name, qty)])
    got = {k: v for d in display for k, v in d.items()}
    return ticket_name, got, alias


def run_scenario(resolve_ticket, matcher, machine, bus, ticket_text, sticker_texts, verbose=False):
    """模擬一筆訂單：傳票解析出 ticket_text，然後依序放上 sticker_texts。

    resolve_ticket 是「傳票文字 -> OrderParser 會回傳的品項名」，正常模式走
    等價層的代表菜名，legacy 模式只做正規化。
    """
    bus.events.clear()

    ticket_name = resolve_ticket(ticket_text)

    quad = (0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0)
    tray = Tray(id="sim", rect=quad, xywhr=(5.0, 5.0, 10.0, 10.0, 0.0), ticket_crop=None)
    tray.ticket = TrackedItem(bbox=quad, xywhr=(5.0, 5.0, 10.0, 10.0, 0.0))

    items, display, combos, report_alias = machine._expand_parsed_items([(ticket_name, 1)])
    tray.expected_items = items
    tray.expected_items_display = display
    tray.report_alias = report_alias
    tray.ticket.contributed_combos = combos
    machine._recompute_expected_combos(tray)
    tray.state = TrayState.WAITING_STICKERS

    results = []
    for text in sticker_texts:
        matched, status = matcher.match(
            [(text, 0.99)], None, False, tray.expected_items, tray.checked_items
        )
        ts = TrackedItem(bbox=quad, xywhr=(5.0, 5.0, 10.0, 10.0, 0.0))
        tray.stickers.append(ts)
        machine._settle_sticker_match(tray, tray.id, ts, matched, status)
        results.append((text, matched, status))

    if verbose:
        print(f"      傳票 {ticket_text!r} -> 代表菜名 {ticket_name!r}")
        print(f"      聯集待核對 = {sorted(Counter(tray.expected_items).items())}")
        print(f"      組合假設   = {[dict(c) for c in tray.expected_combos]}")
        for text, matched, status in results:
            print(f"      貼紙 {text!r} -> {matched!r} [{status}]")
        print(f"      事件 = {bus.statuses()}")

    return tray, results


# (說明, 傳票文字, [貼紙文字], 期望每張貼紙的狀態, 期望最終是否 COMPLETED)
SCENARIOS = [
    ("客戶回報案例：傳票假名、貼紙也假名",
     "から揚げ３個", ["から揚げ３個"], ["MATCHED"], True),
    ("客戶回報案例：傳票假名、貼紙漢字",
     "から揚げ３個", ["唐揚げ３個"], ["MATCHED"], True),
    ("yaml 只收漢字，店家貼紙印假名（等價層要救回來）",
     "から揚丼･並", ["から揚げ丼並"], ["MATCHED"], True),
    ("yaml 只收漢字，店家貼紙也印漢字",
     "から揚丼･並", ["唐揚げ丼並"], ["MATCHED"], True),
    ("護欄：數量不同不可混（３個 vs ６個）",
     "から揚げ３個", ["から揚げ６個"], ["WRONG_ITEM"], False),
    ("護欄：份量不同不可混（並 vs 大）",
     "から揚丼･並", ["唐揚げ丼大"], ["WRONG_ITEM"], False),
    ("完全無關的餐點",
     "から揚丼･並", ["牛皿並盛"], ["WRONG_ITEM"], False),
    ("同名多組合：放最精簡的那一版（不該卡住等唐辛子）",
     "牛たん･牛皿御膳４切", ["4切牛ﾀﾝ･牛皿", "ご飯", "お新香"],
     ["MATCHED", "MATCHED", "MATCHED"], True),
    ("同名多組合：放含たれ的那一版",
     "牛たん･牛皿御膳４切", ["4切牛ﾀﾝ･牛皿", "ご飯", "たれ", "お新香"],
     ["MATCHED", "MATCHED", "MATCHED", "MATCHED"], True),
    ("同名多組合：放含唐辛子的那一版",
     "牛たん･牛皿御膳４切", ["4切牛ﾀﾝ･牛皿", "ご飯", "唐辛子", "お新香"],
     ["MATCHED", "MATCHED", "MATCHED", "MATCHED"], True),
    ("純異寫變體：貼紙印主變體那一種",
     "牛丼小", ["牛丼小"], ["MATCHED"], True),
    ("純異寫變體：貼紙印另一種寫法（仍須 MATCHED，不可誤報 WRONG_ITEM）",
     "牛丼小", ["牛丼小盛"], ["MATCHED"], True),
]

# (說明, 傳票文字, 數量, 期望的顯示清單, 期望的回報改名表|None=不檢查)
#
# 這幾項同時鎖住兩件事：
#   * 2026-09-18 現場回報的「一份變兩份」（顯示只取主變體）
#   * 三層機制各自的分工 —— 哪一層該接手哪種情況
DISPLAY_SCENARIOS = [
    # 第 1 層：ja_readings.yaml 的通用規則。盛->省略 讓 牛丼小/牛丼小盛 直接
    # 變成同一個品項，連變體都不存在，所以改名表是空的。
    ("規則層：牛丼小 / 牛丼小盛 由 盛->省略 解決，不需要改名",
     "牛丼小", 1, {"牛丼小": 1}, {}),

    # 第 2 層：menu_aliases.yaml 的 distinct 台帳否決自動推論。
    # 鰻ご飯増量 與 ご飯増量 是不同商品，不可以被統一改名。
    ("台帳層：distinct 否決自動改名（鰻魚飯加量 != 白飯加量）",
     "ご飯増量", 1, None, {}),

    # 第 3 層：程式的顯示政策。真的組合差異，永遠不能併成同一個品項，
    # 只能顯示挑一種。
    ("顯示層：套餐變體只顯示最精簡的那一版", "チーズ", 3, {"チーズ": 3}, None),
    # 期望值用的是 NFKC 正規化後的代表菜名（半角片假名 ﾀﾝ -> タン、･ -> ・）
    ("顯示層：三種組合變體只顯示最精簡的那一版", "牛たん･牛皿御膳４切", 1,
     {"4切牛タン・牛皿": 1, "ご飯": 1, "お新香": 1}, None),

    ("沒有變體的品項照舊", "から揚丼･並", 2, {"唐揚げ丼並": 2}, {}),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="貼紙核對情境模擬")
    ap.add_argument("--brand", default="yoshinoya")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--legacy", action="store_true",
                    help="重現改動前的行為（無等價層 + 同名後者覆蓋），用來對照")
    args = ap.parse_args()

    from logic.text_utils import normalize_text

    cfg, matcher, machine, bus = build(args.brand, legacy=args.legacy)
    resolve_ticket = (
        (lambda t: normalize_text(t, ignore_space=True)) if args.legacy
        else cfg.ticket_index.canonical_or_self
    )

    mode = "legacy（改動前）" if args.legacy else "current（改動後）"
    print(f"\n模式：{mode}")
    print(f"菜單：傳票候選 {len(cfg.menus_ticket)} / 貼紙候選 {len(machine.sticker_matcher.menus_sticker)} "
          f"/ 多變體傳票名 {sum(1 for v in machine.menu_variants.values() if len(v) > 1)}\n")

    failed = 0
    for i, (desc, ticket, stickers, want_status, want_completed) in enumerate(SCENARIOS, 1):
        tray, results = run_scenario(resolve_ticket, matcher, machine, bus, ticket, stickers, args.verbose)
        got_status = [s for _, _, s in results]
        got_completed = tray.state == TrayState.COMPLETED

        ok = got_status == want_status and got_completed == want_completed
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {i}. {desc}")
        if not ok:
            failed += 1
            print(f"       期望狀態 {want_status} / COMPLETED={want_completed}")
            print(f"       實際狀態 {got_status} / COMPLETED={got_completed}")
            if not args.verbose:
                run_scenario(resolve_ticket, matcher, machine, bus, ticket, stickers, verbose=True)

    print()
    for i, (desc, ticket, qty, want_display, want_alias) in enumerate(DISPLAY_SCENARIOS, 1):
        ticket_name, got_display, alias = run_display_check(
            resolve_ticket, machine, ticket, qty, want_display
        )
        ok = (want_display is None or got_display == want_display) and \
             (want_alias is None or alias == want_alias)
        print(f"[{'PASS' if ok else 'FAIL'}] 顯示 {i}. {desc}")
        if not ok:
            failed += 1
            print(f"       傳票 {ticket!r} -> {ticket_name!r}")
            if want_display is not None:
                print(f"       期望顯示 {want_display}")
            if want_alias is not None:
                print(f"       期望改名 {want_alias}")
            print(f"       實際顯示 {got_display}")
            print(f"       實際改名 {alias}")
        elif args.verbose:
            print(f"       {ticket_name!r} -> 顯示 {got_display}"
                  + (f"  回報改名={alias}" if alias else "  （無改名）"))

    total = len(SCENARIOS) + len(DISPLAY_SCENARIOS)
    print(f"\n結果：{total - failed}/{total} 通過\n")
    # legacy 模式是拿來「證明改動前確實會壞」的，全過反而代表對照組沒設對
    if args.legacy:
        print("（legacy 模式的 FAIL 是預期的，代表改動前這些情境真的不通）\n")
        return 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
