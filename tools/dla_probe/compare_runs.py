# -*- coding: utf-8 -*-
"""比較 ocr_batch 前後兩次執行的結果，統計改善 / 退化 / 不變。"""
import re
import sys

LINE = re.compile(
    r"^\s+(\S+\.jpg)\s+\S+\s+框=(\d+)\s+flip=(\w+)\s+rot90=(\w+)\s+conf_max=(\S+)\s+「(.*)」\s*$"
)


def load(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LINE.match(line.rstrip("\n"))
            if m:
                name, nb, flip, rot, conf, text = m.groups()
                out[name] = {
                    "boxes": int(nb),
                    "rot90": rot == "True",
                    "conf": 0.0 if conf == "-" else float(conf),
                    "text": text,
                }
    return out


old = load(sys.argv[1])
new = load(sys.argv[2])

common = sorted(set(old) & set(new))
print(f"兩邊都有的影像: {len(common)} 張 (old={len(old)}, new={len(new)})")

improved, degraded, same_text, retried = [], [], 0, []
for k in common:
    o, n = old[k], new[k]
    if n["rot90"] != o["rot90"]:
        retried.append(k)
    if n["text"] == o["text"]:
        same_text += 1
        continue
    if n["conf"] > o["conf"]:
        improved.append((k, o, n))
    else:
        degraded.append((k, o, n))

print(f"\n文字不變 : {same_text} 張")
print(f"改善     : {len(improved)} 張")
print(f"退化     : {len(degraded)} 張")
print(f"轉正判斷改變: {len(retried)} 張")

if improved:
    print("\n--- 改善 (前 25 筆) ---")
    for k, o, n in improved[:25]:
        print(f"  {k}")
        print(f"      舊: rot90={o['rot90']} conf={o['conf']:.3f} 「{o['text']}」")
        print(f"      新: rot90={n['rot90']} conf={n['conf']:.3f} 「{n['text']}」")

if degraded:
    print("\n--- 退化 (全部列出) ---")
    for k, o, n in degraded:
        print(f"  {k}")
        print(f"      舊: rot90={o['rot90']} conf={o['conf']:.3f} 「{o['text']}」")
        print(f"      新: rot90={n['rot90']} conf={n['conf']:.3f} 「{n['text']}」")

oc = sum(v["conf"] for v in old.values()) / max(len(old), 1)
nc = sum(v["conf"] for v in new.values()) / max(len(new), 1)
print(f"\n平均 conf_max: 舊 {oc:.3f} -> 新 {nc:.3f}")
lo = sum(1 for v in old.values() if v["conf"] < 0.5)
ln = sum(1 for v in new.values() if v["conf"] < 0.5)
print(f"conf < 0.5 的張數: 舊 {lo} -> 新 {ln}")
