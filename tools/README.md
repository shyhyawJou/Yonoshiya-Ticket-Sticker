# tools/ — OCR 流程的評估與除錯工具

全部在**開發機**可跑。執行環境：Python 3.12 + `annotation_deps/`（cp312 編譯）。
所有指令都從 `DLA_on_720/` 目錄執行。

```bash
cd DLA_on_720
export PYTHONPATH="../annotation_deps;.;./unit_test"     # bash
# PowerShell: $env:PYTHONPATH="../annotation_deps;.;./unit_test"
```

## 單元測試（純邏輯，無模型）

```bash
py -3.12 unit_test/test_ticket_layout.py                  # 19 項：行列配對 / 拒絕邏輯
py -3.12 unit_test/test_order_parser_and_sticker_matcher.py   # 既有 5 項迴歸
```

## 評估

```bash
# 1) 後處理品質（理想 OCR 輸入 = 標註框位+標註文字）
py -3.12 tools/eval_parser_ideal.py --split dev
py -3.12 tools/eval_parser_ideal.py --split test

# 2) 端到端（開發機用 ONNX 替身，非 edge DLA 權重）
py -3.12 tools/eval_ticket_parsing.py --split test
py -3.12 tools/eval_sticker.py

# 3) 上游方向判定的判準比較
py -3.12 tools/eval_orientation.py

# 重跑 OCR 推論（清掉快取）
py -3.12 tools/eval_ticket_parsing.py --split all --refresh
```

## 手動查看單張結果

```bash
py -3.12 tools/ocr_probe.py --type ticket  ../images/order_ticket/xxx.jpg
py -3.12 tools/ocr_probe.py --type sticker ../images/order_sticker/xxx.jpg
py -3.12 tools/ocr_probe.py --type ticket ../images/order_ticket --limit 5 --verbose
py -3.12 tools/ocr_probe.py --type ticket xxx.jpg --json
py -3.12 tools/ocr_probe.py --type ticket xxx.jpg --backend dla    # 在 edge 上
```

## 重要限制

`AI.cpython-312-aarch64-linux-gnu.so` 是 ARM aarch64，**開發機無法載入**，
所以本機一律走 `ONNX_on_nuc` 的 ONNX 權重當替身。ONNX 與 DLA 是不同權重檔
（字典相同，md5 `be1b3350...`），因此：

* 本機可驗證：後處理（分列、欄位閘門、詞彙比對、拒絕邏輯、座標轉換）
* 必須在 edge 驗證：DLA det/cls/rec 的辨識率、NPU 延遲與記憶體

快取檔在 `tools/_eval_cache/`（可安全刪除，會自動重建）。
