# 吉野家 (yoshinoya) 設定說明

這個資料夾就是 `app.py` 實際跑的那一組設定（`runtime.yaml` 的 `profile: yoshinoya` 指到這裡）：

| 檔案 | 內容 |
| --- | --- |
| `config.yaml` | 本文件說明的主設定 |
| `classes.yaml` | 偵測器的 11 個類別與 id，**必須跟 `.dla` 模型訓練時一致** |
| `menus_mapping.yaml` | 菜單來源：每筆 `name` 展開成「傳票」品項、`sub_items` 展開成「貼紙」品項 |
| `menu_aliases.yaml` | 菜名身份的人工台帳（同菜異寫 `equivalents` / 確認不同 `distinct`） |
| `menu_modifiers.yaml` | 傳票修飾列的人工台帳（換湯、換飯…）、持帰り列（`takeout_rows`）、套餐 size 列（`set_size_rows`）與不印貼紙的套餐（`label_less_sets`）；搭配共用的 `../ja_ticket_grammar.yaml` |
| `stub_ticket.yaml` | 客戶聯小票（`small_order_ticket`）的開關、追蹤 / 重試、版面與 POS 代碼設定（見下面「客戶聯小票」） |
| `reference_img/<類別>/` | 要用固定版面 / 參考圖辨認的特殊物件，一類一個資料夾，設定寫在裡面的 `label.yaml`。目前只有 `salad_date_label/`（見下面「沙拉日期標籤」） |
| `japan_dict.txt` | OCR rec 的字典檔 |

四份菜單相關的 yaml 是**四種不同性質的知識**，改東西前先確定該動哪一份：

| 檔案 | 性質 | 會不會被工具重產 |
| --- | --- | --- |
| `../ja_readings.yaml` | 通用日文讀音規則（唐→から、盛→省略），跨品牌 | 不會 |
| `../ja_ticket_grammar.yaml` | 通用傳票修飾詞文法（`変更X` / `Xに変更` / `変)X`），跨品牌。**只做提案不做認定** | 不會 |
| `menu_aliases.yaml` | 本品牌的菜名身份判定 | 不會 |
| `menu_modifiers.yaml` | 本品牌的修飾列判定（哪一列是修改、換掉哪一軸）、持帰り列、套餐 size 列的主餐規則、不印貼紙的套餐（`label_less_sets`，以它為準）、客戶菜單缺的套餐（`extra_sets`，以客戶菜單為準） | 不會 |
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

同一支工具也會列出：

- **持帰り列**（`[持帰り列]`）：菜單裡名稱含 `持帰り` 的品項，以及各自能扣掉哪些套餐。
  外帶時 POS 會把套餐附的小菜另外印一列（`とじ丼 並(玉子)` + `玉子(持帰り)`），但貼紙只有一張，
  所以這一列照常核對，再從附同一張貼紙的套餐扣掉一份；內用不會多這一列，照常核對。
  「可扣的套餐 0 個」通常只是單點品項，但寫法跟套餐的 sub_items 不一致時也長這樣
  （2026-11 的 `ニンニクマシマシたれ` / `ﾆﾝﾆｸﾏｼﾏｼだれ`），要在 `menu_aliases.yaml` 補等價。
- **不印貼紙的套餐**（`[不印貼紙的套餐]`）：`label_less_sets` 的每一筆是客戶菜單原本就有（貼紙被清空）
  還是補上的，以及是否被認成套餐（是套餐，下面的 size 列才會歸給它）。

> **為什麼要有這份 README：**`config.yaml` 在執行期會被程式改寫（見下面「會被程式改寫的設定」），
> 改寫時用 `yaml.safe_dump` 整份重寫，**寫在 yaml 裡的註解會被抹掉**，所以參數說明統一放這裡。

---

## 會被程式改寫的設定

| 設定 | 誰改的 | 改到哪個檔案 |
| --- | --- | --- |
| `preset_roi.<roi>.points` / `.color` | App 的 `cmd/preset_roi`、或開機的自動校正 | **本檔案 `config.yaml`** |
| `camera_params.hik.*.value` | App 的 `cmd/hardware_ctrl`、`cmd/img_reverse_xy` | **本檔案 `config.yaml`**（`stream_vision/hikcam.py::_save_camera_config`） |

專案根目錄的 `runtime.local.yaml` 現在只負責選品牌（`profile`）。舊版曾把相機參數寫在那裡，
啟動時 `profile_config.migrate_legacy_runtime_local()` 會一次性搬回 `config.yaml`。

`auto_update/auto_update.sh` 換新版時只把 `ReverseX` / `ReverseY` / `ticket_roi.points` 從舊的
`config.yaml` 合併過去（`auto_update/merge_config.py`），其他現場調過的相機參數（曝光、增益…）會回到 repo 的值。

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
| `n_settle_frame` | 傳票要連續穩定幾幀才送 OCR（貼紙沒設 `sticker_settle_frame` 時也用這個） | 大 = 更穩但更慢；小 = 反應快但容易在手還沒離開時就送辨識。主迴圈約 85~90 ms 一幀，8 幀 ≈ 0.7 s |
| `sticker_settle_frame` | 貼紙（含醬料包）要連續穩定幾幀才送 OCR（選填，`0` / 不寫 = 跟 `n_settle_frame` 一樣） | 現場傳票通常早就解析完，貼紙卻常常出現不到一秒就被打包，所以兩者分開：傳票寧可多等幾幀拍清楚，貼紙要搶時間。每次重讀也要重新等這麼多幀。啟動 log 會印 `[穩定幀] 傳票 N 幀、貼紙 M 幀` |
| `tray_missing_frame` | 一筆訂單消失幾幀後執行收尾（截圖 + 寫 CSV） | — |
| `sticker_missing_frame` | 貼紙消失幾幀算「真的被拿走」，之後才從追蹤名單移除 | 太小 = 手掠過就被當拿走；太大 = 拿走又放回會被當成同一個 |
| `ticket_leave_frame` | 傳票離開判定幀數（`finalize_mode: ticket_leave` 時用；`order_guard` 超時收尾後，傳票也要連續消失這麼多幀才重新接單） | — |
| `k_container_fail` / `k_container_new` | 餐盤消失 / 新增的確認幀數（**只有 `tray` 模式用到**） | — |
| `still_shift_ratio` | 靜止判定（選填，預設 `0` = 關閉）。相鄰兩幀中心位移超過**框短邊的這個比例**就算還在動，`n_settle_frame` 的累積歸零 —— 避免紙被推動中就送 OCR 拍到糊字 | 例 `0.03` = 3%。太小：框本身的抖動就會一直歸零，OCR 遲遲不觸發；看 log 的 `[OCR#n ■ …]` 那行的「移動重置 N 次」與「出現→送OCR」秒數校正（逐次的 `[still] … 累積 N 幀後移動（位移 x%）` 改在 DEBUG） |
| `wrong_item_confirm` | 錯誤品項確認次數（選填，預設 `0` = 關閉，第一次判錯就成立）。同一張貼紙連續這麼多次 OCR 都沒比對成功才成立：發 `WRONG_ITEM_DETECTED`、CSV 記一筆、畫面標紅 NG。品名不要求每次相同，取出現最多的；中途比對成功或成立前就被拿走都不記 —— 擋畫面模糊、遮擋時一時看錯字造成的 NGtoOK。確認中讀到過不在訂單內品名的貼紙會先擋住訂單完成。全部都讀不出菜名時改發 `Unknown:<讀到的字>`，只提醒，不擋完成、不寫 CSV。醬料包不受影響 | 目前 `2`。每多 1 次，警報晚一次重讀（貼紙要重新靜止 `sticker_settle_frame` 幀、再排到 OCR；同時有很多張待讀時，已成立的錯誤品項排最後）；太小：連續兩次模糊誤讀就會誤判。log 關鍵字 `[wrong_confirm]` |
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

### `order_photo` — 每筆訂單的照片組（選填）

讓客戶不用看影片就能看到每一單。整段可以不寫，不寫 = 關閉。每筆訂單三張：

| 照片 | 什麼時候 | 存在哪裡 |
| --- | --- | --- |
| 開始（原始高清全幀） | 傳票確認（`TICKET_READY`）當下 | `full_screen/<時間>_No<單號>-start.jpg` |
| 最終（原始高清全幀） | 最終狀態那一刻（見下） | `full_screen/<時間>_No<單號>-final-<OK\|NG\|NGtoOK>.jpg` |
| 最終 UI（App 截圖） | 跟上一張同一刻，送 `ocr/v1/app/cmd/screenshot` 請 App 截 | App 存在 `<日期>/screen_capture/`，App 自己上傳 Dashboard |

最終狀態那一刻：

- **OK / NGtoOK**：訂單完成（`TRAY_COMPLETED`）後 `completed_delay_ms` 拍，那時餐點還在畫面上。延後是因為 App 收到狀態要一點時間重畫，馬上截會截到切換中的畫面（實測 0 ms 截到「OK」字樣但底色還是紅的）。完成後如果又退回未完成（例如第二張傳票加了品項），之後再完成會再拍一次。App 的 reset（a / b / all）或切換模式會當下取消還沒拍的最終照，下一筆訂單重新拍開始照與最終照。
- **NG**：要到收尾（護貝板露出 / 傳票離開 / 超時）才知道，收尾當下拍，護貝板模式下餐點多半已經收走。
- 完成後還沒到 `completed_delay_ms` 就收尾的，收尾時補拍。

高清圖跟其他 `full_screen` 圖一樣，凌晨由 aibox_uploader 上傳。沒有任何訂單內容（單號、品項都空）的收尾不拍。

| key | 預設 | 意義 |
| --- | --- | --- |
| `enabled` | `false` | 開關。repo 的 config 寫 `true` |
| `ui_screenshot` | `true` | 拍最終照時要不要請 App 截 UI。App 版本不支援時指令沒人收，不影響辨識 |
| `completed_delay_ms` | `500` | 訂單完成後等幾毫秒才拍（OK / NGtoOK） |
| `hold_events_ms` | `300` | NG 收尾時，送出截圖指令後重置事件（`TRAY_RESET` / `WAITING_TICKET` / `TRAY_REMOVED`）延後幾毫秒才發，App 才不會截到已清空的畫面（實測 0 ms 也還截得到最終畫面，這是保險）。`0` = 不延後 |

log 關鍵字 `[訂單照片]`：App 回報存檔後會印一行照片組，例如 `單號=494 OK 照片組：高清 …_No494-final-OK.jpg、UI …/screenshot_….png，UI 比高清晚 86 ms`。時間差 = App 截圖檔名的時間 − 高清圖的取像時間。App 收到指令約 40 ms 就截圖，但 UI 裡的串流畫面本身還比相機晚一點，這行看不出來。App 3 秒內只接受一次截圖，太密會回 `throttled`，log 會記下來，高清圖照存。

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

### `frame_label` — 全幀的 mmrotate 標籤檔（選填）

整段可以不寫，不寫就是下表的預設值。

每存一張 `full_screen` 全幀，就在**同一個資料夾**輸出**同檔名**的 `.json`：X-AnyLabeling 格式（格式見 `good_example`），
內容是**這一幀 mmrotate 偵測到的每一個物件**。座標都是整張全幀的座標，`imageWidth` / `imageHeight` 是全幀的大小，
所以直接用 X-AnyLabeling 開 `full_screen` 資料夾就能看到、修正。**不含 OCR 結果**，所以存圖的當下就輸出，不用等 OCR。

```
<日期>/full_screen/20261028_052109_217_sticker.jpg      全幀（原本就有）
<日期>/full_screen/20261028_052109_217_sticker.json     這一幀的偵測標籤（新）
<日期>/full_screen/20261028_052109_341_sauce.jpg        醬料包的全幀
<日期>/full_screen/20261028_052109_341_sauce.json       同樣是偵測標籤
```

四種全幀（`ticket`、`sticker`、`ticket-extra`、`sauce`）都有；這一幀偵測不到任何物件就不寫檔。

```yaml
frame_label:
  enabled: true
  image_wait_sec: 10.0
  max_queue: 200
```

| key | 預設 | 意義 | 調整建議 |
| --- | --- | --- | --- |
| `enabled` | `true` | `false` = 不輸出（不建背景執行緒，也不蒐集偵測結果） | 磁碟或 log 吃緊時先關 |
| `image_wait_sec` | `10.0` | 標籤要等全幀寫成檔才寫（沒有圖的 JSON 沒用）。全幀由另一條執行緒寫，最多等這麼久；等不到就不寫這份標籤並警告 | 寫檔很慢時加大 |
| `max_queue` | `200` | 背景佇列上限，滿了丟棄並用 loguru 警告（每 50 筆一則），不拖慢辨識 | |

**每個偵測框（shape）**

| 欄位 | 內容 |
| --- | --- |
| `label` | 偵測的類別名（`order_ticket`、`order_sticker`、醬料包、盤子…） |
| `shape_type` | `rotation`（X-AnyLabeling 的旋轉框） |
| `points` | 4 個角點（xyxyxyxy，即 `Detection.xyxy`），已經是全幀座標，順序是偵測器的 pt1~pt4 |
| `direction` | 偵測器的旋轉角 `r`（弧度），原樣放入，不取負、不換算 |
| `score` | 偵測信心 |
| `attributes` | `source: "mmrotate"` 與原始 `xywhr`（中心 x、y、寬、高、弧度） |

**`direction` 與角點順序**

- **單位與正負號都已確認**：X-AnyLabeling 的 `direction` 是弧度、順時針為正；偵測器的 `r` 也是弧度、順時針為正
  （影像座標 y 向下，`r` > 0 時寬軸往畫面下方轉，用 `xywhr2xyxyxyxy` 實際算過），兩邊一樣所以直接用。
- **還沒確認的**：0 度時是不是「寬邊沿 +x」（跟偵測器 `r` = 0 一樣）；另外角點順序是偵測器的 pt1~pt4
  （`r` = 0 時依序是右下、右上、左上、左下，視覺上逆時針），沒有重排成「左上起、順時針」。
  開發環境沒有 X-AnyLabeling 可以開來看，上機後請開一張標籤檔確認框的角度對不對；不對的話只要改
  `frame_labeler.detection_shapes` 一處。`points` 本身直接來自偵測器、已經是全幀座標，DOTA / YOLO-OBB 這類匯出用的就是這 4 個角點。

**限制（請看一下）**

- **全幀沒存成就沒有標籤**：寫檔佇列滿了丟掉的那一幀、或 `image_wait_sec` 內沒出現的，標籤跳過（log 有警告）。所以 JSON 一定有對應的圖。
- 偵測結果是 mmrotate **當下的輸出**（已過各類別信心門檻與 NMS），是複查 / 當訓練資料的起點，不是標準答案。
- 偵測結果是**存圖那一輪**的快照（主迴圈 `mmr.detect` 之後、`logic.update` 之前記下），不會被之後的幀蓋掉。
- **上傳器**：`aibox_uploader` 只上傳 `.mp4 / .jpg / .jpeg / .png`，`.json` 會被略過、不影響上傳，但它的 log 會每個 `.json` 記一行
  「略過非媒體檔」，量跟 `full_screen` 的圖一樣多。rclone 搬日期資料夾時 `.json` 會一起搬。
- 量：每存一張全幀就多一個 JSON（每個約幾 KB，偵測框越多越大）。一張 NG 貼紙被反覆重讀，就會有很多個。
- 舊版（曾經有 OCR 文字框的 `ocr_label:` 區段）如果還留在 config.yaml 裡，會被直接忽略，不會報錯，可以刪掉。

### `ocr_rec_save` — 每筆訂單的 rec 小圖（選填）

整段可以不寫，不寫就是下表的預設值（**預設關閉**：這是除錯用的，要用時寫 `enabled: true`。現場的 `config.yaml` 是狀態、部署不會覆蓋，預設開的話沒寫這一段的機台一更新程式就開始存）。

debug sheet 畫的是整張轉正後的圖，但 rec **讀的不是那張整圖**，而是每個偵測框各自裁出來、再經過 cls 單獨判 180 度
（必要時只翻那一個框）的小圖。整張圖看起來是正的、某個數字卻辨識錯誤時，要看的是 rec 當下吃進去的那張小圖。
開啟後，每次 OCR 的**每個框**都另存一張：小圖原樣（不縮放）放上面，底下一條白底寫**最終辨識結果**。

```
<save_dir>/<日期>/ocr_rec/<訂單>/<時分秒>_<類型><序號>_<框編號>.jpg
例：20261111/ocr_rec/No452-1234/143022_sticker57_01.jpg
```

| 部分 | 說明 |
| --- | --- |
| `<訂單>` | `No<單號>`（跟訂單照片組的標籤同寫法）。這次 OCR 套用完 tray 上還沒有單號就用 `tray<id>`，所以同一筆訂單最前面幾次讀取可能落在 `tray<id>` 資料夾 |
| `<類型>`、`<序號>` | OCR 的類型（`ticket`、`ticket_extra`、`sticker`…）與 log 上 `[OCR#n]` 的 n，跟 debug sheet JSON 的 `ocr_seq` 是同一個數字，可以對照 |
| `<框編號>` | 這次讀取最後採用的結果裡第幾個框，跟 debug sheet 畫的框同順序 |
| 圖上的文字 | **最終辨識結果**：OCR 引擎最後採用的那份 `rec_res`，就是 debug sheet 呈現的那份。過程（翻轉比對、轉 180 重讀、正方形四個方向）不存 |
| 小圖 | **rec 實際讀的那一張**：同一次讀取跑了好幾輪 rec 時，是最後採用那一輪的小圖；cls 單獨翻過的框，存的就是翻過的樣子 |

```yaml
ocr_rec_save:
  enabled: true
  max_per_order: 3000
  max_queue: 400
  font_path: ''
```

| key | 預設 | 意義 | 調整建議 |
| --- | --- | --- | --- |
| `enabled` | `false` | `false` = 不存，OCR 引擎也不會把小圖帶回主迴圈 | 除錯時才開，查完關掉 |
| `max_per_order` | `3000` | 每筆訂單最多存幾張，超過就不再存並警告一次 | 貼紙被反覆重讀時量會很大，可調小 |
| `max_queue` | `400` | 背景佇列上限（單位：一次 OCR），滿了整筆丟棄並警告，不拖慢辨識 | |
| `font_path` | `''` | 日文字型，留空 = `tasks/ocr/NotoSansJP-Bold.otf`（跟 `debug_sheet` 同一個） | |

量（沙箱 x86 量到的，ARM 會慢一些）：每張約 10 KB、畫圖加寫檔約 1 毫秒（在背景執行緒）；一次傳票 OCR（約 30 個框）約 300 KB，
一次貼紙 OCR（1 到 3 個框）約 20 KB。貼紙每被重讀一次就多存一次，所以一筆訂單可能有幾百張；`ocr_rec/` 在日期資料夾底下，
會跟著日期資料夾被搬走，**預設不會被上傳器上傳**（它只掃 `full_screen`）。

**這一版沒有存的**（它們的辨識結果不在 `rec_res` 的框裡，只出現在 `parsed` 的數量欄與菜名）：
傳票數量的「方法三」（數量欄另外裁圖重讀，`qty_src` 是 `數量欄crop` 的那種）、吉野家 CV 輔助補讀 det 漏框的格子，以及醬料包（不走 OCR）。
數字錯在 `qty_src = 數量欄crop` 的那一列時，這裡看不到那張圖。

---

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

### `ticket_cv_assist` — 大傳票 CV 輔助（實驗中）

```yaml
ticket_cv_assist: true     # 不寫 = false
```

只對 `ticket_parser: legacy` 有效。打開後，大傳票（`order_ticket`）改成「版面交給 CV、字交給 det」：

1. **det 之前先用表格框線決定方向**：数欄（最窄那欄）在左 = 正向。方向可信（表格高度剛好是
   整數個列距、表格左邊有方框）就只跑一次 det、一輪 rec，省掉「翻轉比對 / 轉180重讀」。
   log：`[OCR#n ■ ticket] 路徑=CV版面98ms(轉180)+CV定向`
2. **左邊有方框或数欄有字的列才是品項**；`■■追加■■`、`●テイクアウト●`、`▲▲取消▲▲` 是段落標題。
   取消段落裡的品項目前**只記 log、照常核對**。
3. **det 框依位置分到 CV 的列與欄**；det 漏框、只框到一部分、或字對菜單不完整的格子才用 CV 框
   補讀，log 標 `[CV補救]`。單號優先用 det 框，沒有才讀 CV 框。
4. **找不到表格**（紙被切到、疊在別張上）或方向不可信時，走原本的流程，結果跟關閉時一樣。

實作在 `logic/yoshinoya/ticket_cv_layout.py`（版面）與 `ticket_cv_assist.py`（分工與補救）；
在機器上離線比較兩條路的準確率與耗時：`python3 tools/ticket_cv_probe.py --out <資料夾> "<傳票 crop 或 --frames 整張畫面>"`。

---

## 客戶聯小票（`stub_ticket.yaml`，實驗中）

客戶聯（偵測類別 `small_order_ticket`）是跟大傳票一起出單的小票。它**只拿來跟主票交叉比對、
結果只記 log**：不改 tray 狀態、不影響 `TRAY_COMPLETED`，不發 MQTT，也不寫 CSV。
它跟 `config.yaml` 分開放，所以不會被執行期的改寫抹掉註解；app 啟動時才讀，改了要重啟。
檔案不存在時退回程式預設值 `enabled: false`。

```yaml
enabled: true                 # false = 完全不追蹤、不送 OCR（偵測到也忽略）
class_name: small_order_ticket
stable_frames: 2              # 靜止幾幀才送 OCR
max_attempts: 3               # 解析失敗最多送幾次，用完就放棄
mmr_cv_enhance:               # 偵測階段的 CV 補強，目前全關
  enable_candidate_fallback: false   # 只偵測到主票時，從主票頂部推算客戶聯候選框
  enable_full_image_fallback: false  # 整張畫面沒有偵測框時，用白紙邊界當候選框
  enable_roi_refinement: false       # 已偵測到的客戶聯框，用白紙邊界貼緊修正
layout: …                     # 版面比例、表頭/表尾雜訊字、21→1 這類數量校正、POS 易混碼、品名↔POS 對照
```

流程：

1. **追蹤**：每筆訂單最多一張。範圍跟主票相同：`preset_roi` 模式只收中心點在 `ticket_roi` 內的；
   `single` 模式排除 `ticket_exclude_roi`；`tray` 模式只收在餐盤內的。客戶聯不見超過
   `stability.sticker_missing_frame` 幀、而且還沒解析成功，就丟掉。
2. **OCR**：靜止滿 `stable_frames` 幀就送，**不用等主票**，送出也不會改 tray 狀態，所以不會卡住貼紙。
3. **解析**（`logic/yoshinoya/stub_parser.py`）：表頭讀時間、桌號、單號；品項讀 4 碼 POS 代碼和品名。
   POS 碼被遮住時用 `pos_lookup` 從品名反查，易混的碼依 `pos_confusion` 修正；數量欄 det 漏框時，
   對齊數量欄補一個框重讀（overlay 上的**橘框**）。
   log：`[客戶聯解析結果] tray=… 單號=… 桌號=… 時間=…｜品名(POS)×數量…`、`[客戶聯 JSON] …`
4. **交叉比對**（`logic/yoshinoya/stub_verifier.py`）：主票（含合併進來的第二張票）和客戶聯都讀到之後，
   比單號（只比數字）和品項數量。不論哪一張先讀完，都會在後讀完的那張讀完時比對。
   log：一致 `[stub_verify] … 客戶聯與主票一致`，不一致 `WARNING [stub_verify] … 客戶聯與主票不一致: stub少=… stub多=…`

實作：追蹤在三個 tracker 的 `_update_stub`，派發與分派在 `logic/tray_state_machine.py`，
結果套用在 `logic/readers/stub.py`，資料結構是 `logic/models.py::Stub`（`Tray.stub`）。

---

## 沙拉日期標籤（`reference_img/salad_date_label/label.yaml`，實驗中）

點サラダ時，除了一般的品項貼紙，旁邊常會多貼一張小標籤：小字「サラダ」、「保存方法 10℃以下 / 消費期限」，
底下是大字的日期（`26. 8. 8`）與時間（`16:27`）。偵測器一樣框成 `order_sticker`。現場不保證兩張都會貼。

沒打開時它照一般貼紙比對：只讀到日期時間（品名小字常被貼紙的字高過濾濾掉）就連續兩次發 Unknown；
讀到「サラダ」又會變成多一份サラダ（錯誤品項、擋完成）。

```yaml
enabled: false   # 檔案不存在也是 false；app 啟動時讀，改了要重啟
item: サラダ      # 這張標籤代表的品項（貼紙名稱，要在 menus_mapping.yaml 的 sub_items 裡）
require_template: true   # 一定要模板對到才算日期標籤（2026-11-16 起）
cv:              # 參考圖輔助（det 讀字，CV 確認身份、補救）
  enabled: true
  refs: "*.jpg"  # 參考圖 = 同資料夾裡轉正的標籤 crop
  template: {rows_above: 1, include: [date]}   # 模板區：日期正上方 1 列小字 + 日期列（跳過品名列）
  min_score: 0.6
  scales: [0.80, 1.25]
  fields: {date: -2, time: -1}   # 欄位是第幾列（負數從下面數）
```

打開後：

1. **判定與讀取**（`logic/readers/date_label.py`，det 優先、CV 輔助）：
   - **det**：rec 讀到合法的日期（`YY.M.D`）或時間（`HH:MM`），而且其他框沒有對到 `item` 以外的菜名，是日期標籤的候選。
   - **一定要模板對到**（`require_template`，2026-11-16）：只靠 det 讀到日期時間不算。現場店內備料罐的保存期限標籤
     （`72時間 / 26.11.16 / 15:58`）被 det 讀出 `26.11.16`，就被當成サラダ、報錯誤品項 —— 日期時間是所有保存期限標籤都有的。
     模板沒對到的照一般貼紙比對（讀不出菜名 = Unknown，不擋完成），log 記 `讀到日期/時間 …，但模板沒對到，不當日期標籤`。
     `false` = 舊行為（模板沒對到時只靠 det 也算）。
   - **CV**（`logic/readers/label_cv.py`）：在貼紙 crop 上找參考圖的模板區（`template`：日期正上方 1 列小字「消費期限」+ 日期列，
     跳過最上面的品名列；不寫 `template` = 欄位上方所有列），
     決定方向與尺度，再依版面切出日期、時間兩列給 rec 讀。版面不用手標：載入時每張參考圖自己用水平投影切列。
     det 已判定時，CV 補 det 缺的欄位（`[CV補救]`），兩邊不同時用 det、記 `[CV不一致]`；det 沒判定、而且 det 的字
     不是サラダ以外的菜名時，模板對到**而且** CV 讀出合法的日期或時間，才判定成日期標籤（CV 不能單獨判定）。
     模板分數不到 `min_score` 就不是日期標籤。
   - 2026-11-16 改成跳過品名列：現場 サラダ 日期標籤被另一張貼紙壓住上半部，含品名的模板只有 0.45 對不到。新模板區：
     沙拉日期標籤 0.82、被壓住的 0.81、沒拿來當參考圖的兩張 0.83 / 0.87、備料罐標籤 0.32、一般貼紙最高 0.43（528 張）。
     實驗機 full_screen 抽樣 684 張貼紙，新舊設定判定完全一樣（日期標籤 48 張）。日期數字每張不同，所以分數比舊模板低；
     品名跟小字黏成一列的參考圖切不出這段，載入時略過（目前 6 張用 4 張）。
   - 2026-11-06 實驗機（舊模板區）：沙拉日期標籤模板分數 0.93~0.99、一般貼紙最高 0.46（557 張，0 張誤對到）。
     7 張樣本 × 4 個方向，模擬 det 漏時間 / 漏日期 / 只讀到サラダ / 什麼都沒讀到，都讀對（28/28）。
     耗時：沙拉日期標籤約 76 ms（中位數）、一般貼紙在粗找就放棄約 31 ms，在主迴圈裡跑。
     歪 10 度時 CV 開始讀錯（det 不受影響），所以 CV 只當輔助。
   - **新增參考圖**：把轉正的標籤 crop（`.jpg`）丟進這個資料夾、重啟 app。參考圖要能切出足夠的列：
     切不出 `fields` 要的列數、或欄位上方沒有模板區的會被略過（log 有 WARNING；切不出 `template.rows_above` 要的列數是 INFO）。
2. **核對**（`logic/readers/sticker.py`）。沙拉品項分兩種（2026-11-16 現場說明）：
   - **原味 `サラダ`**（`item`）：日期標籤本身就是它的貼紙。一般貼紙與日期標籤任一張都能代表那份サラダ：先到的照一般
     品項核對（`ITEM_CHECKED`），另一種後到時跟它成組，不再計數、不再通知前端。
   - **口味沙拉**（`salad_items`：貼紙名稱含 `サラダ`、排除 `サラダ抜き`，由菜單推導，目前 `ポテトサラダ`、`ミニサラダ`）：
     一定有完整名稱的一般貼紙，日期標籤是附屬的。日期標籤不能代替它：先到時「收著」（不計數、不報錯），
     一般貼紙照常核對後跟它成組。
   - 日期標籤進來時依序：① 跟已核對、還沒成組的沙拉一般貼紙（原味或口味）成組 → ② 原味 サラダ 還沒到齊就以它核對
     サラダ → ③ 口味沙拉還有沒配到日期標籤的份數就收著 → ④ 都不是就是多的，照錯誤品項（`サラダ`）。
   - 同一筆同時點原味與口味沙拉時，日期標籤分不出是誰的。① 排在 ② 前面是刻意的：錯的時候是「サラダ 顯示沒到」
     （店員看得到），而不是「原味沒放、口味附的日期標籤被當成 サラダ」讓訂單靜默完成。
   - 成組只看「同一筆訂單裡缺另一半」，不看位置。多出來的照常判錯：點一份卻有兩張一般貼紙、日期標籤比沙拉份數多、
     或沒點沙拉卻有日期標籤，都走錯誤品項。成組 / 收著的日期標籤被移動時，跟一般貼紙一樣接回原本那張（`[貼紙移動]`）。
3. **離開畫面**：組裡還有一張在就不扣（日期標籤被手遮住，那份沙拉其實還在），兩張都離開才扣。收著的日期標籤離開什麼都不扣。
4. **輸出**：日期時間**只印 log**，MQTT 與 CSV 都不變：
   `[沙拉日期] サラダ(模板0.95):26.8.8-16:27：還沒有一般貼紙，先以日期標籤核對` / `…：跟已核對的一般貼紙成組（不重複計數）`。
   括號是身份的根據：`模板0.82` = 參考圖的模板區對到了；`設定` = 只有 det 讀到日期時間（只在 `require_template: false` 時出現）。
   品名一律照 `item` 填（品名小字常被貼紙的字高過濾濾掉，也可能被壓住，不靠它）。只讀到其中一個時另一個印 `?`。

已知限制：訂單完成（`COMPLETED`）後主迴圈不再讀貼紙，所以日期標籤比最後一個品項晚放時，日期不會出現在 log。
日期標籤先到、一般貼紙後到時，一般貼紙那次會先印一行 `[錯誤餐點]`（StickerMatcher 比對時サラダ已滿額），
緊接著的 `[沙拉日期] 一般貼紙 … 成組` 才是結果。

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
