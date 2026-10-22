# -*- coding: utf-8 -*-
"""量測「正方形框一律重試」對 predict 延遲的影響。"""
import glob
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ocr_engine import TextSystemDLA

D = sys.argv[1] if len(sys.argv) > 1 else "/mnt/reserved/record/stream/20260914/ocr_cut"
files = sorted(glob.glob(os.path.join(D, "*.jpg")))

ts = TextSystemDLA("tasks/ocr/yoshi_det_v01_03.dla", "tasks/ocr/cls_v2.dla",
                   "tasks/ocr/yoshi_rec_v01_05.dla", "tasks/ocr/yoshinoya/japan_dict.txt")

# 先挑出「正方形框」的樣本，只有這些會受影響
square = []
for p in files:
    img = cv2.imread(p)
    if img is None:
        continue
    b, _ = ts.run_det(img)
    if len(b) == 0 or ts._check_boxes_need_rotate_90(b):
        continue
    if ts._orientation_is_ambiguous(b):
        square.append(p)

print(f"正方形框樣本 {len(square)} 張 (總共 {len(files)} 張)")

for label, thr in (("不重試 (thresh=0.0)", 0.0), ("一律重試 (thresh=1.01)", 1.01)):
    ts.orient_retry_conf_thresh = thr
    # 暖機
    for p in square[:3]:
        ts.predict(cv2.imread(p), obj_type="sticker")
    times = []
    for p in square:
        img = cv2.imread(p)
        t0 = time.time()
        ts.predict(img, obj_type="sticker")
        times.append((time.time() - t0) * 1000)
    a = np.array(times)
    print(f"{label:24} 平均 {a.mean():7.1f} ms   中位數 {np.median(a):7.1f} ms   "
          f"p95 {np.percentile(a, 95):7.1f} ms   最大 {a.max():7.1f} ms")
