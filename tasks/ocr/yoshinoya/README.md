# 吉野家 (yoshinoya) 設定說明

這個資料夾就是 `app.py` 實際跑的那一組設定（`runtime.yaml` 的 `profile: yoshinoya` 指到這裡）：

| 檔案 | 內容 |
| --- | --- |
| `config.yaml` | 本文件說明的主設定 |
| `classes.yaml` | 偵測器的 11 個類別與 id，**必須跟 `.dla` 模型訓練時一致** |
| `menus_mapping.yaml` | 菜單來源：每筆 `name` 展開成「傳票」品項、`sub_items` 展開成「貼紙」品項 |
| `menu_aliases.yaml` | 菜名身份的人工台帳（同菜異寫 `equivalents` / 確認不同 `distinct`） |
| `menu_modifiers.yaml` | 傳票修飾列的人工台帳（換湯、換飯…）；搭配共用的 `../ja_ticket_grammar.yaml` |
| `japan_dict.txt` | OCR rec 的字典檔 |

四份菜單相關的 yaml 是**四種不同性質的知識**，改東西前先確定該動哪一份：

| 檔案 | 性質 | 會不會被工具重產 |
| --- | --- | --- |
| `../ja_readings.yaml` | 通用日文讀音規則（唐→から、盛→省略），跨品牌 | 不會 |
| `../ja_ticket_grammar.yaml` | 通用傳票修飾詞文法（`変更X` / `Xに変更` / `変)X`），跨品牌。**只做提案不做認定** | 不會 |
| `menu_aliases.yaml` | 本品牌的菜名身份判定 | 不會 |
| `menu_modifiers.yaml` | 本品牌的修飾列判定（哪一列是修改、換掉哪一軸） | 不會 |
| `menus_mapping.yaml` | 客戶 xlsx 的忠實投影 | **會，整份覆蓋** |

所以任何人工判斷都不可以寫進 `menus_mapping.yaml`。

xlsx 改版重產 `menus_mapping.yaml` 之後，跑這兩支確認台帳還跟得上：

```sh
python3 tools/verify_menu_alias.py      --brand yoshinoya   # 菜名身份
python3 tools/verify_menu_modifiers.py  --brand yoshinoya --scenarios   # 修飾列 + 錨定回歸
```

後者會把「文法上像修飾列、但台帳還沒歸軸」的列印成可直接貼進
`menu_modifiers.yaml` 的片段，並以 exit 1 提醒。漏歸軸的後果是前端多顯示
一份，而且訂單可能到不了 `TRAY_COMPLETED`（詳見 `menu_modifiers.yaml` 開頭）。

> **為什麼要有這份 README：**`config.yaml` 在執行期會被程式改寫（見下面「會被程式改寫的設定」），
> 改寫時用 `yaml.safe_dump` 整份重寫，**寫在 yaml 裡的註解會被抹掉**，所以參數說明統一放這裡。

---

## 會被程式改寫的設定

| 設定 | 誰改的 | 改到哪個檔案 |
| --- | --- | --- |
| `preset_roi.<roi>.points` / `.color` | App 的 `cmd/preset_roi`、或開機的自動校正 | **本檔案 `config.yaml`** |
| `camera_params.hik.*.value` | App 的 `cmd/hardware_ctrl`、`cmd/img_reverse_xy` | 專案根目錄的 `runtime.local.yaml`（profile override） |

`runtime.local.yaml` 的內容會蓋掉 `config.yaml`，所以現場調過相機參數之後，
看「實際生效的值」要看 `runtime.local.yaml`，不是只看這裡。

除了上面兩項，其他參數改完都要重啟 app 才會生效。

---

## config.yaml 各區塊

### `mode` — 狀態機種類

`tray` | `single` | `preset_roi`（**目前正式使用 `preset_roi`**）

- `tray`：追蹤多個餐盤，傳票/貼紙依所在餐盤分組，可同時多筆訂單
- `single`：不追蹤餐盤，畫面固定只跑一筆訂單
- `preset_roi`：`single` 的特化版 —— 傳票**中心點進入 `preset_roi.ticket_roi`** 才算一筆訂單開始

也可以由 App 用 `cmd/no_tray_setting` 在執行期切（`no_tray: true` → `preset_roi`，`false` → `tray`），
所以這裡寫的只是「開機時的起始值」。切模式會重建狀態機，進行中的訂單狀態會被丟棄。

### `includes` — 外掛檔案

```yaml
includes:
  classes: classes.yaml            # 必填
  menus_mapping: menus_mapping.yaml # 必填，菜單的唯一來源
  # vocab_ticket: vocab_ticket.txt  # 選填
  # vocab_sticker: vocab_sticker.txt # 選填
```

`vocab_*` 是「這個店真的會出現的詞」（一行一個詞，`#` 開頭是註解）。
有設就拿它當模糊比對的候選字庫；不設則退回用 `menus_mapping.yaml` 展開出來的全部品項。
吉野家品項多（上千筆），小場景直接用全表當候選會大量錯配 —— 那種時候才需要 `vocab_*`。

### `mqtt` — 與 App 的連線

```yaml
mqtt:
  host: localhost
  port: 1883
  base_topic: ocr
  session: v1
```

所有 topic 都是 `<base_topic>/<session>/...`，例如 `ocr/v1/cmd/reset`、`ocr/v1/events/detection_status`。
改 `base_topic`/`session` 就等於換一整組 topic，**必須跟 App 同步改**。

### `runtime.model` — 模型與字典

路徑相對於**本檔案所在資料夾**（`../xxx.dla` 就是 `tasks/ocr/` 底下）。

| key | 用途 |
| --- | --- |
| `object_det` | 旋轉框偵測（傳票 / 貼紙 / 醬料包 / 傳票墊） |
| `ocr_det` | OCR 文字框偵測 |
| `ocr_cls` | 文字方向分類 |
| `ocr_rec` | 文字辨識 |
| `text` | `ocr_rec` 對應的字典檔 |

換 `object_det` 一定要同時確認 `classes.yaml` 對得上；換 `ocr_rec` 要同時換 `text`。

### `runtime.camera` — 取像

```yaml
camera:
  source: 0      # 裝置索引
  width: 4024    # 取像解析度，也是 preset_roi 座標系的基準
  height: 3036
  device: hik    # hik | aravis
```

`width/height` 不只是影像大小 —— **`preset_roi` 的座標、App 送來的 ROI 座標都是以這組解析度為基準**，
改了之後原本的 ROI 座標全部會對不上，要一起重設。

### `runtime.stream` — MJPEG 預覽

```yaml
stream:
  port: 9527
  stream_size: [1280, 960]   # 推流前縮到這個大小
```

看畫面：`ssh -L 9527:127.0.0.1:9527 root@<box-ip>` 之後開 `http://127.0.0.1:9527/meal`。

### `thresholds` — 偵測門檻

```yaml
thresholds:
  ai_conf: 0.5   # 信心度下限，調低會多抓也多誤抓
  ai_iou: 0.4    # NMS 的 IoU 門檻
```

### `placement` — 餐盤分組的幾何門檻

**只有 `mode: tray` 會用到**，`preset_roi` / `single` 模式下這一段不影響任何行為（但 key 仍須存在）。

| key | 意義 |
| --- | --- |
| `roi_strink` | 餐盤 ROI 往內縮的比例，避免邊緣物件被誤收 |
| `iou_assign` | 物件要跟餐盤重疊多少才算「在這個盤裡」 |
| `iou_candidate` | 候選餐盤的合併門檻 |
| `drift_iou_thresh` | 超過這個相似度算「沒有移動」 |

### `stability` — 時間穩定度（幀數）

相機約 15 fps，所以「15 幀 ≈ 1 秒」可以拿來換算。

| key | 意義 | 調大 / 調小 |
| --- | --- | --- |
| `n_settle_frame` | 傳票/貼紙要連續穩定幾幀才送 OCR | 大 = 更穩但更慢；小 = 反應快但容易在手還沒離開時就送辨識 |
| `tray_missing_frame` | 一筆訂單消失幾幀後執行收尾（截圖 + 寫 CSV） | — |
| `sticker_missing_frame` | 貼紙消失幾幀算「真的被拿走」，之後才從追蹤名單移除 | 太小 = 手掠過就被當拿走；太大 = 拿走又放回會被當成同一個 |
| `ticket_leave_frame` | 傳票離開判定幀數（`finalize_mode: ticket_leave` 時用；`order_guard` 超時收尾後，傳票也要連續消失這麼多幀才重新接單） | — |
| `k_container_fail` / `k_container_new` | 餐盤消失 / 新增的確認幀數（**只有 `tray` 模式用到**） | — |
| `still_shift_ratio` | 靜止判定（選填，預設 `0` = 關閉）。相鄰兩幀中心位移超過**框短邊的這個比例**就算還在動，`n_settle_frame` 的累積歸零 —— 避免紙被推動中就送 OCR 拍到糊字 | 例 `0.03` = 3%。太小：框本身的抖動就會一直歸零，OCR 遲遲不觸發；看 log 的 `[OCR#n ■ …]` 那行的「移動重置 N 次」與「出現→送OCR」秒數校正（逐次的 `[still] … 累積 N 幀後移動（位移 x%）` 改在 DEBUG） |
| `wrong_item_confirm` | 錯誤品項確認次數（選填，預設 `0` = 關閉，第一次判錯就成立）。同一張貼紙連續這麼多次 OCR 都沒比對成功才成立：發 `WRONG_ITEM_DETECTED`、CSV 記一筆、畫面標紅 NG。品名不要求每次相同，取出現最多的；中途比對成功或成立前就被拿走都不記 —— 擋畫面模糊、遮擋時一時看錯字造成的 NGtoOK。確認中讀到過不在訂單內品名的貼紙會先擋住訂單完成。全部都讀不出菜名時改發 `Unknown:<讀到的字>`，只提醒，不擋完成、不寫 CSV。醬料包不受影響 | 目前 `2`。每多 1 次，警報晚約 1 秒（貼紙要重新靜止 `n_settle_frame` 幀再讀）；太小：連續兩次模糊誤讀就會誤判。log 關鍵字 `[wrong_confirm]` |
| `wrong_item_strong_score` / `wrong_item_strong_conf` | 強證據一次成立（選填，`strong_score` 預設 `0` = 關閉；`strong_conf` 預設 `0.9`）。讀到不在訂單內的品名，且比對分數 ≥ `strong_score`、rec 信心 ≥ `strong_conf`，第一次就成立，不必等滿 `wrong_item_confirm` 次。依據：歷史 log 裡看錯字（模糊、遮擋）都是模糊比對（分數 < 100），一字不差讀成另一個菜名幾乎都是真的放錯。以下仍要等滿次數：不是真的一字不差（ねぎ 前綴邏輯的 100 分）、品名本來就在訂單裡（已核對的餐點被推開換了框、或真的多放一份同名）、品名跟還沒核對的品項互為子序列（`牛黒カレー` 被遮成 `黒カレー`） | 目前 `100` / `0.9`。rec 信心本身分不出看錯字（10/06 的誤讀信心 0.99），所以主要靠分數；log 會寫 `強證據（一字不差、信心夠高）` |

> `preset_roi` 模式目前用的收尾判斷是 `ticket_mat`（傳票墊重新完整露出 + ROI 內沒有傳票 → 收尾），
> 不是 `ticket_leave`。這個選擇與它的確認幀數目前寫在 `logic_engine.py` / `PresetRoiTracker`，
> 還不能從 yaml 設定。

### `order_guard` — 訂單防護（選填）

整段可以不寫，不寫就是下表的預設值，所以機器上執行期會被改寫的 config.yaml 不需要為它改動。

| key | 預設 | 意義 |
| --- | --- | --- |
| `timeout_sec` | `600` | 一筆訂單從傳票確認（`TICKET_READY`）起最多進行幾秒。超過就照一般收尾寫 CSV、重置、停錄影（`final_status` 照原規則判 OK/NG，CSV 欄位不變）。之後要等 ROI 內的傳票連續消失 `ticket_leave_frame` 幀才重新接單，避免還放在桌上的同一張票被再讀成一筆新單；App 的 reset 會直接放行。`0` = 關閉 |
| `max_item_qty` | `20` | 傳票上任一品項數量**超過**這個值就視為 OCR 讀錯，整張作廢、重新偵測（第二張同單號傳票則只淘汰那一張，進行中的訂單不動）。`0` = 關閉 |

為什麼需要超時：`preset_roi` 模式唯一的正常出口是護貝板重新露出。作業員一直沒露出時訂單永遠不會結束，
之後放進同一格的傳票會被當成同一張主票（位置重疊）而不送 OCR，這些訂單就不會出現在 CSV 裡。
log 搜尋 `[order_timeout]` / `[ticket_guard]` 可以看到這兩個機制什麼時候動作。

app.py 的錄影 watchdog 仍在，但改成 `timeout_sec + 30` 秒，只當保險用。

### `camera_params` — 相機參數

`aravis` 那一段是舊的 Aravis 後端用的簡單鍵值；實際在用的是 `hik`，每個參數是一筆記錄：

```yaml
hik:
  ExposureTime:
    value: 50000.0     # 目前值（App 改參數時改的就是這個）
    default: 75000.0   # cmd/hardware_ctrl reset_parameter 會回到這個值
    type: float        # int | float | bool | string
    max: 500000        # 數值型的上下限
    min: 20
  BalanceWhiteAuto:
    value: Continuous
    type: string
    allowed: [Continuous, Once, 'Off']   # 字串型的合法選項（App 的下拉選單）
```

`ReverseX` / `ReverseY` 是畫面上下左右翻轉（`cmd/img_reverse_xy` 會改這兩個並重啟相機）。
**翻轉不會自動翻轉 `preset_roi` 的座標**，翻完要重新設一次 ROI。

### `depth_camera` — 深度相機

```yaml
depth_camera:
  enable: true          # false = 完全不啟動（不用拔插頭）
  id: 0                 # 裝置索引
  max_depth_mm: 1200    # 深度上色的上限，超過就飽和
  fps_log_interval: 5   # 幾秒印一次深度 FPS
  log_tag: Depth
  enable_fps_log: true
```

`enable: false` 時不開 thread、不載入 Orbbec SDK，畫面只有 RGB —— 跟「沒接深度相機」完全同一條路徑。
`enable: true` 但沒插相機也不會出事（啟動失敗後自動關閉自己）。
**改這個要重啟 app**。有接深度相機且啟用時，畫面會變成「RGB 半邊 + 深度 JET 半邊」並排。

### `birghtness_controller` — 自動亮度（目前未啟用）

`stream_vision/hikcam.py` 裡相關程式碼整段是註解掉的，所以這一段目前**不會有任何作用**；
但 `load_config` 仍然要求這個 key 存在，不要刪掉。
（key 名稱的拼字 `birghtness` 是既有的錯字，改名會讓舊設定檔載不起來，故保留。）

### `preset_roi` — 預設 ROI

```yaml
preset_roi:
  ticket_roi:
    source: manual        # manual | auto
    auto:
      class_name: ticket_mat   # 要盯哪個偵測類別（必須是 classes.yaml 裡的名字）
      padding_ratio: 0.12      # 取該物件的外接正矩形後，向外擴這個比例
      confirm_frames: 15       # 連續幾幀穩定才定案
      stable_iou: 0.9          # 幀間 IoU 門檻，判斷是不是同一個物件
      timeout_frames: 600      # 這麼多幀還沒定案就放棄，沿用下面的 points（<=0 = 不逾時）
      persist: true            # 定案後是否寫回 config.yaml
    points:                    # 左上 / 右上 / 右下 / 左下
      - [400, 1400]
      - [1600, 1400]
      - [1600, 2800]
      - [400, 2800]
    color: [0, 255, 255]       # RGB 0~255（畫面上會自動轉成 BGR）
  # object_roi:  預留：辨識物件的篩選區域（目前程式不讀）
  # packing_roi: 預留：餐品打包完成區域（目前程式不讀）
```

- **只有 `ticket_roi` 會影響訂單判斷**：`preset_roi` 模式下，傳票中心點落在這個範圍內才算一筆訂單開始。
  （`single` 模式反過來用它當「排除區」：只處理 ROI **外**的傳票。）
- `object_roi` / `packing_roi` 是預留欄位，可以照同樣結構寫（含 `source` / `auto`），但目前沒有行為掛上去。
- `points` 是四個點、任何形狀的四邊形都可以（不必是正矩形），但要圍得出面積、順序為左上→右上→右下→左下。
- `source: manual`：用 `points` 寫的座標（App 用 `cmd/preset_roi` 設定的值也是存在這裡）。
- `source: auto`：開機後盯 `auto.class_name` 的偵測框，連續 `confirm_frames` 幀穩定就取它旋轉框的
  **外接正矩形**、外擴 `padding_ratio`，寫進 `points`。**一次性**，定案或逾時就停，不會一直改 ROI。
- `class_name` 打錯（不在 `classes.yaml`）時，載入設定就會報錯並自動退回 `manual`，不會白等到逾時。

> **`source: auto` 時，自動優先。** `source` 是「這個 ROI 的座標來源」這個政策本身，
> App 用 `cmd/preset_roi` 設的座標會寫進 `points`、**當下立刻生效**，
> 但**下次開機還是會重新自動對位、把它覆蓋掉**（相機被移動後永遠自動回正，不用人管）。
> 要讓前端設的座標固定下來，就把 `source` 改成 `manual`。

App 端的介面（跟前端約定）：

```
ocr/v1/cmd/preset_roi
{"action":"set","name":"ticket_roi","points":[[400,1400],[1600,1400],[1600,2800],[400,2800]],"color":[0,255,255]}
{"action":"auto","name":"ticket_roi"}    # 重跑一次自動校正
```

App 只送 `name` / `points` / `color`，**它沒送到的 key（`source`、`auto`）會原地保留**，
所以 App 設定 ROI 不會把自動校正的參數洗掉。收到 `action: set` 會把該 ROI 還在跑的自動校正停掉，
以 App 給的座標為準。套用成功會回一則 `events/system_info`：`{"type": "PRESET_ROI_UPDATED", ...}`。

### `csv_log` — 訂單紀錄

```yaml
csv_log:
  root_dir: /mnt/reserved/record/stream   # 跟影片同一個基準路徑
  folder: csv_logs                        # <root_dir>/<YYYYMMDD>/<folder>/
  video_category: event                   # CSV 裡的影片檔名指向哪一類錄影
  max_video_size: null                    # 例 "1GB" / "900MB"；null = 不切檔
```

每完成一筆訂單寫一列，UTF-8-BOM（給 Excel 直接開）。

### `debug_sheet` — 訂單除錯總覽圖（選填）

整段可以不寫，不寫就是下表的預設值，所以機器上執行期會被改寫的 config.yaml 不需要為它改動。

一筆訂單收尾時，把這筆訂單看過的傳票、貼紙、醬料包的裁切圖、OCR det 框、rec 文字、模糊比對結果、
`menus_mapping.yaml` 對應名稱與 OK/NG 判定，排進**一張** 4024 x 3036 的圖，只看這張就知道這筆訂單每個環節的結果。
圖的標頭結論跟 CSV 同一份資料。輸出在 `<root_dir>/<YYYYMMDD>/debug_sheet/<時間>_<tray>_No<單號>_<狀態>_p01.jpg`（分頁見下方），
旁邊有同名（不含 `_p01`）的 `.json`（結構化的判定資料，可以 grep）。

```yaml
debug_sheet:
  enabled: true
  mode: all             # all | ng_only
  root_dir: ''          # 留空 = 沿用 csv_log.root_dir
  folder: debug_sheet
  canvas_w: 4024
  canvas_h: 3036
  jpeg_quality: 85
  save_json: true
  font_path: ''         # 留空 = tasks/ocr/NotoSansJP-Bold.otf
  max_entries: 80
  crop_max_side: 1600
  tiles_per_page: 24
  max_pages: 4
  slow_record_ms: 50.0
  slow_render_s: 5.0
```

| key | 預設 | 意義 | 調整建議 |
| --- | --- | --- | --- |
| `enabled` | `true` | `false` = 完全不收集、不渲染（主迴圈只多一次布林判斷） | 懷疑它影響辨識時先關掉比對 |
| `mode` | `all` | `all` = 每筆訂單都出圖；`ng_only` = 只有最後不是 OK（NG / NGtoOK）、或有缺少品項、或有錯誤品項的訂單才出圖。寫錯會退回 `all` 並記 WARNING | 磁碟吃緊或只關心出問題的單時用 `ng_only` |
| `root_dir` | `''` | 輸出基準路徑，留空 = `csv_log.root_dir`，圖會跟當天的影片放在同一個日期資料夾 | 一般不用改 |
| `folder` | `debug_sheet` | 日期資料夾底下的子資料夾名稱。crontab 的 rclone 會跟著整個日期資料夾一起搬走 | |
| `canvas_w` / `canvas_h` | `4024` / `3036` | 輸出圖的尺寸。版面一律先照 4024 寬排好（高度依長寬比），再**縮放**到這個尺寸，所以改小只會讓字與檔案一起變小，版面不會擠壞；**不會讓渲染變快**（還多一次縮放）。要加快請改 `tiles_per_page`、`mode`、`jpeg_quality`。建議長寬比維持 4:3 | 檔案太大再縮小，例如 `2576` x `1944` |
| `jpeg_quality` | `85` | JPEG 品質（1~100）。**只影響檔案大小與編碼時間**，不影響渲染 | 磁碟吃緊先調這個，70 左右仍可讀 |
| `save_json` | `true` | 同名 `.json` 一併存（判定、比對分數、候選、事件，不含圖） | |
| `font_path` | `''` | 日文字型，留空 = `tasks/ocr/NotoSansJP-Bold.otf`。需要 `pillow`；沒裝或找不到字型時圖仍會出，但日文會變 `?`，開機 log 會有 `[debug_sheet] 沒有可用的 Pillow / 日文字型` | |
| `max_entries` | `80` | 單筆訂單最多保留幾筆 OCR 紀錄的裁切圖，超過就釋放最舊的貼紙裁切圖（文字紀錄保留）。傳票另外只留最新 4 張的圖 | 單筆訂單貼紙特別多時才需要調 |
| `crop_max_side` | `1600` | 出圖前把裁切圖縮到最長邊不超過這個值（縮圖在背景 thread，不佔辨識時間） | 調小可略為加快渲染 |
| `tiles_per_page` | `24` | 每張圖最多放幾格貼紙 / 醬料包（一格 = 一個追蹤物件，NG 後被 OK 蓋住會拆成兩格）。**放不下的往下一張圖放**，有問題的（NG / 疑似 / Unknown）排在最前面，所以 NG 一定在第 1 張。有傳票的頁每格約 432x517 px（24 格時文字約 20 px，要放大看），調小每格更大更好讀 | 常有超過 20 項的訂單可調小到 12~16 |
| `max_pages` | `4` | 一筆訂單最多輸出幾張圖，超過的格子只留在 `.json`，最後一張會註明數量。防止極端訂單吃光磁碟與 CPU。**範圍 1~99**（頁碼固定 2 碼），設超過 99 會在載入設定時直接報錯 `ValueError`，app 起不來 | |
| `slow_record_ms` | `50.0` | 主迴圈裡單次「記錄」超過幾毫秒就記 WARNING。這部分會直接拖慢辨識 | |
| `slow_render_s` | `5.0` | 背景一張圖（渲染 + 編碼 + 寫檔）超過幾秒就記 WARNING | |

這一段**改完要重啟 app 才會生效**，而且不會被 App 的指令改寫。

**會不會拖慢辨識**：收集（縮圖以外）都在主迴圈，每筆訂單累計約 6~13 ms；渲染、編碼、寫檔在獨立 thread。
每張圖輸出時 log 會有一段 `[debug_sheet][timing]`，主迴圈與背景分開列。要量這台機器的真實數字（不需要相機與模型）：

```sh
python3 tools/debug_sheet_bench.py                     # 預設：10 張貼紙、3 個醬料包、5 筆訂單
python3 tools/debug_sheet_bench.py --stickers 20 --orders 10
```

上面的 6~13 ms 與背景渲染 0.5~1.2 秒是開發機 x86 的數字，**邊緣機要自己量**。要先看版面長怎樣：
`python3 tools/debug_sheet_demo.py /tmp/sample.jpg`（用合成資料，不需要硬體）。

**分頁與檔名**：每一張圖的檔名格式都一樣，頁碼**固定 2 碼**（`_p01`~`_p99`），**單頁的訂單也帶 `_p01`**：

```
<時間>_<tray>_No<單號>_<狀態>_p01.jpg    # 第 1 張：傳票 + NG / 疑似 / Unknown 優先的格子
<時間>_<tray>_No<單號>_<狀態>_p02.jpg    # 第 2 張起：放不下的格子
<時間>_<tray>_No<單號>_<狀態>.json       # 整筆訂單共用一份，不帶頁碼；內有 pages 與 files
```

一筆訂單超過 `tiles_per_page` 格時才會有 `_p02` 以後。頁數上限 99（`max_pages` 超過 99 直接報錯，見上表）。
舊版（無分頁）產生的檔案沒有 `_pNN`，資料夾裡可能同時有兩種寫法。
每張的標頭都有完整的結果與「第 N / M 頁」，格子編號跨頁連續。第 1 張放最新 3 張傳票與最重要的格子；
更舊、還有裁切圖的傳票往後面的頁放，沒有圖的舊傳票只留在 `.json`。

**傳票與貼紙的框怎麼標**：
- **傳票**：圖上的 det 框只標**編號**（一張傳票常有幾十個框，把文字標在圖上會互相疊、擋住傳票本身的字）。
  文字在旁邊對照，有兩處：
  - 傳票圖**下方**的「框編號對照」：**全部**的框都列出 `編號:文字(信心)`，文字顏色 = 圖上那個框的顏色
    （歸了列 = 該列的顏色，沒歸列 = 依 rec 信心）。歸了列的框，右側那一列只有合併後的文字，各框自己的信心只有這裡看得到。
    放不下時會寫「…另 N 個框見 .json」。
  - 右側每一列結果最右邊的「框 2、3」：這一列用到哪幾個框。吉野家解析器有「列 → 框」對應；鈴茂的表格解析器沒有，
    所以鈴茂的列不會有「框 N」，所有的框都在下方對照。
- **貼紙**：維持 `編號:文字 信心` 直接標在框上（一張貼紙只有一兩個框，不會擋）。

注意：App 按 reset 作廢的訂單不會出圖。
crontab 的雲端上傳（`aibox_uploader/`）**預設不會上傳這個資料夾**（它的 `image.dirs` 預設只有 `full_screen`，設定在 `aibox_uploader/aibox_uploader.py`），這裡的 `cloud_upload.folders` 也只管 app 內建上傳。檔名已符合上傳器的規則（只含英數與 `._-`、以日期開頭），要上傳時把 `debug_sheet` 加進它的 `dirs` 即可；但這些圖在 Dashboard 上會被當成哪一類媒體、容量與流量是否可接受，我沒有驗證過。

### `cloud_upload` — 雲端上傳（app 內建，v3.3.0 起關閉）

v3.3.0 起上傳改由 crontab 執行 `aibox_uploader/aibox_uploader.sh`，這一段維持 `enabled: false`。
**crontab 那一行還在的時候不可以打開**，否則會上傳兩次。

```yaml
cloud_upload:
  enabled: false     # true = app 內建上傳：每天 at 上傳昨天的 CSV 與 folders 內的媒體
  at: '01:00'        # HH:MM；看到 623 這種數字是舊版留下的（= 10:23），改回時間寫法即可
  window_min: 120    # 到 at + window_min 強制截止（之後 crontab 的 rclone 會搬走同一批資料夾）
  store_code: ''     # 留空 = hostname 後 4 碼
  folders: [event]
```

整段可以不寫（= 關閉）。**API 金鑰與伺服器憑證不在這裡**，要人工放到機器的
`/etc/yoshinoya_ai/`，完整步驟、log 判讀與跟 crontab 的時間關係見 [`docs/cloud_upload.md`](../../../../docs/cloud_upload.md)。

### `ocr_routes` — 文字偵測後端

```yaml
ocr_routes:
  order_ticket: det    # det | cv_table | cv_ink
  order_sticker: det
```

`det` = 走 OCR 偵測模型；`cv_table` / `cv_ink` = 用傳統影像處理切文字區域（表格線 / 墨跡）。
key 也可以寫縮寫 `ticket` / `sticker`。值打錯會在載入設定時直接報錯。

### `ticket_parser` — 傳票解析器

`table` | `legacy`（**吉野家目前用 `legacy`**）。

`legacy` = `logic/yoshinoya/order_parser.py`（依吉野家傳票版面寫的版本）；
`table` = `logic/szm/order_parser.py`（Suzumo 的表格解析，會用到 `vocab_ticket`）。

---

## 改完之後

`config.yaml` 只是資料，改錯不會有語法檢查，所以改完先確認能載入再跑 app：

```sh
cd /home/root/ocr_mmr_dla
python3 -c "
from profile_config import load_profile_config
cfg = load_profile_config('.', 'yoshinoya')
print(cfg.mode, cfg.runtime.depth_cam.enable, cfg.preset_roi.ticket_roi.source, cfg.preset_roi.ticket_roi.points)"
```

沒有 traceback 就代表設定檔本身沒問題，再 `bash run_app.sh` 看實際畫面。
