# OCR 文字框分流設定

在 `tasks/ocr/<品牌>/config.yaml` 設定（品牌為 yoshinoya 或 suzumo）。`runtime.local.yaml` 只負責選擇品牌，不接受 OCR 路由或其他設定覆寫：

```yaml
ocr_routes:
  order_ticket: cv_table
  order_sticker: det
```

- `cv_table`：使用表格格線、條碼方向及儲存格墨跡取框，後續仍使用現有 DLA REC。
- `cv_ink`：單詞貼紙墨跡輪廓取框，沿用 CLS 及 REC。貼紙字距變化大，預設不用此法。
- `det`：保留原 DET → 方向判定 → CLS／REC。
- 欄位省略時回到 DET。`ticket`／`sticker` 內部任務名稱會自動對應正式類別名稱。
- 這只是既有 OCR 任務的處理方式，不會讓未被狀態機排入 OCR 的其他物件自動產生任務。

吉野家的 `ticket_cv_assist`（見 `tasks/ocr/yoshinoya/README.md`）不是這裡的路由：大傳票仍走 `det`，只是 AsyncOCR 在 det 之前先用 `logic/yoshinoya/ticket_cv_layout.py` 決定方向（跳過翻轉比對），版面再交給解析器做 det 框分格與漏框補讀。找不到表格時照 `det` 原流程。

流程接線位於 `manager/task_context.py` 與 `ocr_engine.AsyncOCR`。
CV 路徑使用 `logic/szm/cv_text_regions.py`，同時回傳轉正後圖片、文字框與座標變換，合成 `M_inv_full` 後再交給既有回呼，避免畫框位置偏移。
表格找不到時回空框；不會在背後改用 DET。DET 模型目前仍在 OCR 服務初始化時載入，CV 是略過該次推論，不代表完全不占 DET 模型記憶體。

共用純 CV 程式位於同層 `../ONNX_on_nuc/suzumo/table_grid.py`、`sticker_box.py`，部署時須保留這兩個檔案及相對位置；它們不依賴 OpenVINO。DLA 主程式依然需原本的 NPU `AI` 模組及廠商環境。
本次沒有把 ONNX REC 轉成 DLA，現有設定仍指向原 DLA REC，辨識成效必須在 edge 另測。

本機已用假 NPU 回傳驗證 CV worker 不呼叫 DET，並驗證矩陣轉換和既有文字解析；不是實體 NPU 成功證明。
切换方式後，在 edge 使用原 `run_app.sh`／離線入口驗證收據各方向、品名與數量及畫框回投位置。

更多 NUC 執行、相機設定與測試指令見 [NUC 指南](../ONNX_on_nuc/README.md)。
