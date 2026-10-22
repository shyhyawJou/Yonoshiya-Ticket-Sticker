# tools/dla_probe/ — 在邊緣平台上診斷 DLA 的 OCR 流程

跟隔壁 `tools/` 的評估工具（走開發機 ONNX）**不同**：這一批**只能在 G720 上跑**，
因為它們直接載入 `AI.cpython-312-aarch64-linux-gnu.so` 的 DLA 模型。

用途是回答「換了模型 / 改了參數之後，到底是哪一段變差」這類問題：把 det 的原始
機率圖、每一道閘門過濾掉多少、方向判斷的輸入、rec 的信心，全部攤開來看，而不是
只看最後有沒有讀到字。

> 這批腳本是 2026-09-14/15 追「兩個字的直式貼紙讀不出來」時寫的。當時的結論：
> det 與 rec 的 DLA 轉檔都正常（機率圖飽和到 1.0、框分數 0.95），真正的失敗點是
> 90 度轉正判斷 —— 兩個字直排的 det 框近乎正方形（長短比 1.03~1.05），過不了
> `logic/orientation.py` 的 `VERTICAL_ASPECT = 1.4`，於是整張不轉正，直立文字
> 進 rec 只讀到一個垃圾字。修法見 `ocr_engine.py::predict` 的「1.6 方向補救」。

## 執行方式

**一律從專案根目錄執行**（腳本的 `--det` / `--rec` 預設值是相對路徑）：

```sh
ssh root@<box-ip>
cd /home/root/ocr_mmr_dla
python3 tools/dla_probe/<script>.py ...
```

跑之前先確認 app 沒在跑，否則會跟它搶 DLA：

```sh
systemctl is-active yoshinoya_ai.service   # 期望 inactive
pgrep -af "python3 app.py"                 # 期望沒有輸出
```

輸入是**已裁切好的** order_ticket / order_sticker 影像（MMRotate 之後的產物），
現成的來源是 `/mnt/reserved/record/stream/<日期>/ocr_cut/`。

## 各腳本

| 腳本 | 回答什麼問題 |
| --- | --- |
| `det_probe.py` | 單張 det 的完整解剖：letterbox 比例、前處理數值範圍、**原始機率圖 min/max/mean/百分位**、超過 `det_thresh` 的像素比例、每個候選輪廓的分數、掃 `box_thresh` 0.6→0.1 各通過幾個。**區分「模型沒反應」與「被門檻擋掉」就看這支** |
| `det_batch.py` | 一整批跑 det，統計零框比例；零框的再回頭看機率圖最大值，分類成「模型對這張沒反應」或「有候選但被 `box_thresh`/`min_size` 擋掉」 |
| `ocr_batch.py` | 一整批跑完整 det→cls→rec（走 production 的 `TextSystemDLA.predict`），印出每張的框數 / `rot90` / `flip` / 信心 / 讀出的文字。**換模型前後各跑一次存檔，再用 `compare_runs.py` 比對** |
| `compare_runs.py` | 比對兩次 `ocr_batch.py` 的輸出，統計改善 / 退化 / 文字不變 / 轉正判斷改變，並算平均信心與低信心張數。唯一不需要 DLA 的一支（純文字解析，開發機也能跑） |
| `orient_probe.py` | 90 度轉正判斷的輸入全攤開：每個框的 width/height、長短比、是否被判直式、面積佔比，以及最後的面積加權比例 vs 門檻 |
| `rotate_test.py` | 同一張圖分別以原圖 / 順時針 90 / 逆時針 90 餵進 `predict`，比較讀出的文字與信心。**驗「是不是方向問題」最快的一支** |
| `thresh_margin.py` | 量 `orient_retry_conf_thresh` 的安全餘裕：把正方形框樣本分成「轉 90 度後較好」與「原方向就是對的」兩群，比較兩群的信心分佈、找出門檻的安全區、掃各門檻的漏補救/多跑次數 |
| `latency_test.py` | 量「正方形框一律重試」對 `predict` 延遲的影響（只統計會受影響的正方形框樣本） |

## 典型用法：換了 det / rec 模型，想知道有沒有變差

```sh
cd /home/root/ocr_mmr_dla
D=/mnt/reserved/record/stream/20260914/ocr_cut

# 1) 舊模型跑一次存檔
python3 tools/dla_probe/ocr_batch.py --det tasks/ocr/<舊det>.dla \
    --rec tasks/ocr/<舊rec>.dla --type sticker $D 2>/dev/null \
    | grep -E "^  2026|統計|總數" > /tmp/ocr_old.txt

# 2) 新模型跑一次存檔
python3 tools/dla_probe/ocr_batch.py --det tasks/ocr/<新det>.dla \
    --rec tasks/ocr/<新rec>.dla --type sticker $D 2>/dev/null \
    | grep -E "^  2026|統計|總數" > /tmp/ocr_new.txt

# 3) 比對
python3 tools/dla_probe/compare_runs.py /tmp/ocr_old.txt /tmp/ocr_new.txt
```

傳票要另外用 `--type ticket` 跑一次（走的後處理不同），確認沒有被連帶影響。

## 已知限制

* `--dict` 預設指向 `tasks/ocr/yoshinoya/japan_dict.txt`。換品牌 profile 時要自己
  帶對應的字典，**字典對不上 rec 的輸出索引就會整片是亂碼**。
* 這些腳本不做物件偵測（不跑 MMRotate），所以「貼紙根本沒被偵測到」這類問題
  它們看不出來 —— 那要看 app 的 log 與 MJPEG overlay。
* `apusys` / `Input size` / `release the ...` 之類的輸出是 DLA runtime 自己印的，
  不是錯誤，用 `2>/dev/null` 或 grep 過濾掉即可。
