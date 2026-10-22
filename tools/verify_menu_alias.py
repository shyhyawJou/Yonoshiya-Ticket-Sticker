#!/usr/bin/env python3
"""
verify_menu_alias.py
====================
菜名等價層（logic/menu_alias.py）的稽核與回歸檢查工具。

為什麼需要它
------------
折疊規則是雙面刃：漏合併只會讓餐點顯示「未確認」，作業員看得到；
但**錯誤合併會讓錯的餐點被靜默判為 MATCHED**。所以每次動到
tasks/ocr/ja_readings.yaml、某品牌的 menu_aliases.yaml，或用工具重產
menus_mapping.yaml 之後，都應該跑一次這支程式，人工看一眼合併清單。

用法
----
    python3 tools/verify_menu_alias.py                    # 檢查所有品牌
    python3 tools/verify_menu_alias.py --brand yoshinoya  # 只檢查一個
    python3 tools/verify_menu_alias.py --show-all         # 連單一寫法的群組也列出

    # 模擬：某張貼紙印這樣、傳票印那樣，會不會過
    python3 tools/verify_menu_alias.py --brand yoshinoya \
        --simulate "から揚げ丼並=から揚丼･並" --simulate "から揚げ３個=から揚げ３個"

離開碼
------
    0 = 沒有護欄衝突
    1 = 有護欄衝突（代表某條讀音規則想把數量/份量不同的菜併在一起）
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

from logic.menu_alias import MenuIndex, build_menu_index, load_alias_knowledge  # noqa: E402
from logic.text_utils import normalize_text  # noqa: E402

TASKS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks", "ocr"
)
SHARED_READINGS = os.path.join(TASKS_DIR, "ja_readings.yaml")


def load_pools(brand: str):
    """回傳 (ticket 菜名清單, sticker 菜名清單, name -> [變體組合])。清單保留重複次數。"""
    path = os.path.join(TASKS_DIR, brand, "menus_mapping.yaml")
    with open(path, "r", encoding="utf-8-sig") as fh:
        data = yaml.safe_load(fh) or {}

    ticket_names, sticker_names = [], []
    variants = defaultdict(list)
    for row in data.get("mapping", []) or []:
        name = str(row["name"])
        subs = [str(s) for s in row.get("sub_items", []) or []]
        ticket_names.append(name)
        sticker_names.extend(subs)
        variants[normalize_text(name, ignore_space=True)].append(subs)
    return ticket_names, sticker_names, variants


def report_groups(index: MenuIndex, label: str, show_all: bool) -> None:
    groups = defaultdict(list)
    for rep in index.canonical_names:
        groups[rep] = index.aliases_of(rep)

    multi = {rep: names for rep, names in groups.items() if len(names) > 1}
    print(f"\n--- {label}：{len(groups)} 個候選，其中 {len(multi)} 群含多種寫法 ---")
    for rep, names in sorted(multi.items()):
        others = [n for n in names if n != rep]
        print(f"  {rep}  <=  {others}")
    if show_all:
        for rep, names in sorted(groups.items()):
            if len(names) == 1:
                print(f"  {rep}  (單一寫法)")


def report_variants(variants) -> None:
    multi = {
        name: combos
        for name, combos in variants.items()
        if len({tuple(c) for c in combos}) > 1
    }
    print(f"\n--- 同名多組合（多重假設判定的來源）：{len(multi)} 個傳票名稱 ---")
    for name, combos in sorted(multi.items()):
        distinct_combos = sorted({tuple(c) for c in combos})
        print(f"  {name}")
        for combo in distinct_combos:
            print(f"      - {list(combo)}")


def simulate(pairs, sticker_index: MenuIndex, ticket_index: MenuIndex, variants) -> None:
    """用等價層模擬「貼紙 vs 傳票」的核對結果，不含 OCR 與相似度。"""
    print("\n--- 模擬 ---")
    for spec in pairs:
        if "=" not in spec:
            print(f"  [格式錯誤] 需要 '貼紙文字=傳票文字'，收到: {spec!r}")
            continue
        sticker_text, ticket_text = spec.split("=", 1)

        ticket_rep = ticket_index.canonical(ticket_text)
        sticker_rep = sticker_index.canonical(sticker_text)

        combos = variants.get(normalize_text(ticket_text, ignore_space=True))
        if combos is None and ticket_rep:
            combos = variants.get(ticket_rep)

        expected_sets = []
        for combo in combos or []:
            expected_sets.append(
                Counter(sticker_index.canonical_or_self(s) for s in combo)
            )

        if sticker_rep is None:
            verdict = "UNRECOGNIZED（等價層查不到這個寫法）"
        elif not expected_sets:
            verdict = "傳票名稱不在菜單中，無法判定"
        elif any(c.get(sticker_rep, 0) > 0 for c in expected_sets):
            verdict = "MATCHED"
        else:
            verdict = "WRONG_ITEM"

        print(f"  貼紙 {sticker_text!r} -> 代表菜名 {sticker_rep!r}")
        print(f"  傳票 {ticket_text!r} -> 變體 {[list(c) for c in expected_sets]}")
        print(f"  => {verdict}\n")


def check_brand(brand: str, show_all: bool, sims) -> int:
    print("=" * 72)
    print(f"品牌: {brand}")
    print("=" * 72)

    aliases_path = os.path.join(TASKS_DIR, brand, "menu_aliases.yaml")
    know = load_alias_knowledge(SHARED_READINGS, aliases_path)

    ticket_names, sticker_names, variants = load_pools(brand)

    ticket_index = build_menu_index(know, f"{brand}/ticket", ticket_names)
    # 貼紙側的顯示寫法優先跟傳票一致（見 MenuIndex.build 的 prefer 說明）
    sticker_index = build_menu_index(
        know, f"{brand}/sticker", sticker_names, prefer=ticket_names
    )

    report_groups(ticket_index, "傳票菜名", show_all)
    report_groups(sticker_index, "貼紙菜名", show_all)
    report_variants(variants)

    conflicts = ticket_index.conflicts + sticker_index.conflicts
    print(f"\n--- 護欄衝突：{len(conflicts)} 筆 ---")
    for line in conflicts:
        print(f"  {line}")
    if not conflicts:
        print("  (無)")

    # 台帳錯字偵測：equivalents / distinct 指到的菜名，兩個池都找不到就是打錯了
    # 或菜單已經改版。只有在這裡查得出來 —— MenuIndex 一次只看一個池。
    known = {normalize_text(n, ignore_space=True) for n in ticket_names + sticker_names}
    dangling = []
    for label, groups in (("equivalents", know.equivalents), ("distinct", know.distinct)):
        for group in groups:
            missing = [n for n in group
                       if normalize_text(n, ignore_space=True) not in known]
            if missing:
                dangling.append(f"{label} {group} -> 菜單中不存在: {missing}")
    print(f"\n--- 台帳指向不存在的菜名：{len(dangling)} 筆 ---")
    for line in dangling:
        print(f"  {line}")
    if not dangling:
        print("  (無)")

    if sims:
        simulate(sims, sticker_index, ticket_index, variants)

    return 1 if (conflicts or dangling) else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="菜名等價層稽核工具")
    ap.add_argument("--brand", action="append", help="只檢查這個品牌，可重複；預設全部")
    ap.add_argument("--show-all", action="store_true", help="連單一寫法的群組也列出")
    ap.add_argument("--simulate", action="append", default=[],
                    metavar="貼紙文字=傳票文字", help="模擬一次核對，可重複")
    args = ap.parse_args()

    brands = args.brand or sorted(
        d for d in os.listdir(TASKS_DIR)
        if os.path.isfile(os.path.join(TASKS_DIR, d, "menus_mapping.yaml"))
    )

    rc = 0
    for brand in brands:
        rc |= check_brand(brand, args.show_all, args.simulate)
    return rc


if __name__ == "__main__":
    sys.exit(main())
