# 執行
```
./run_app.sh
```

# 看 stream
```
ssh -L 9527:127.0.0.1:9527 root@192.168.1.90
```

# mqtt pub
- 觸發存圖
```
mosquitto_pub -h 127.0.0.1 -p 1883 -t "ocr/v1/cmd/reset" -m '{"tray_id": "single_order", "type": "a"}'
```

> 機器上沒有 `mosquitto_pub`，實機測試改用 python3 + paho：
> ```sh
> python3 -c "
> import paho.mqtt.client as mqtt, json, time
> c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
> c.connect('127.0.0.1', 1883, 10); c.loop_start()
> c.publish('ocr/v1/cmd/preset_roi', json.dumps({'action':'auto'}), qos=1)
> time.sleep(2)"
> ```

# 預設 ROI 設定 (cmd/preset_roi)

App 設定、AI 落地：App 只送指令不寫檔，現值一律由 AI 寫回 `tasks/ocr/<profile>/config.yaml`，
App 下次啟動再從那裡唯讀載入。設定完馬上生效，不用重啟程式。

```
ocr/v1/cmd/preset_roi
{"action": "set", "name": "ticket_roi",
 "points": [[400,1400],[1600,1400],[1600,2800],[400,2800]],
 "color": [0,255,255]}
```

- `name`：`config.yaml` 的 `preset_roi` 底下的 key（目前 `ticket_roi` / `object_roi` / `packing_roi`，
  只有 `ticket_roi` 會影響訂單生命週期判斷）
- `points`：左上/右上/右下/左下，座標系 = `runtime.camera.width/height` 的取像解析度，
  超出範圍會夾回邊界並留 warning
- `color`：顯示顏色 **RGB** 0~255（選填；畫面上會自動轉成 BGR）

套用成功會回一則 `events/system_info`：
```json
{"type": "PRESET_ROI_UPDATED", "msg": {"name": "...", "points": [...], "color": [...], "source": "mqtt"}}
```

App 只送 `name` / `points` / `color`，它沒送到的 key 會原地保留。

## ROI 自動校正

每個 ROI 可以各自決定座標來源（`preset_roi.<roi>.source`）：

```yaml
preset_roi:
  ticket_roi:
    source: auto          # manual = 用 points；auto = 開機自動框
    auto:
      class_name: ticket_mat   # 盯哪個偵測類別（classes.yaml 裡的名字）
      padding_ratio: 0.12
      confirm_frames: 15
      stable_iou: 0.9
      timeout_frames: 600
      persist: true
    points: [...]
```

`auto` 時開機後盯 `class_name` 那個類別的偵測框，連續 `confirm_frames` 幀穩定就取它旋轉框的
**外接正矩形**、外擴 `padding_ratio`，寫進 `points`（預設也寫回 config.yaml）。
一次性，定案或逾時就停，不會一直改 ROI。

隨時可以重跑一次（相機被搬動、現場重新擺位時很方便）：

```
ocr/v1/cmd/preset_roi  {"action": "auto", "name": "ticket_roi"}
```

`source: manual` 時不會自動跑，但仍可用上面這個指令臨時對一次。
反過來，收到 `action: "set"` 會把該 ROI 還在跑的自動校正停掉，以 App 給的座標為準。

注意 `source: auto` 時是**自動優先**：App 設的座標當下立刻生效，但下次開機會重新自動對位覆蓋它。
要讓前端設的座標固定下來，把 `source` 改成 `manual`。

# 深度相機開關

`config.yaml` 的 `depth_camera.enable`：`false` = 完全不啟動（不開 thread、不載入 Orbbec SDK，
畫面只有 RGB），不用拔插頭。改完要重啟 app。

# 只能跑一份 app.py（單例鎖）

`app.py` 啟動時對 `/run/yoshinoya_ai.lock` 上 `flock`。搶不到鎖的那份**不會**去碰
相機、DLA 與 MQTT，只 log 一行就以 exit code 3 結束；`yoshinoya_ai.service` 的
`RestartPreventExitStatus=3` 讓 systemd 不去重啟它。鎖綁在 fd 上，process 結束
（含當掉、被 kill）由 kernel 自動釋放，不會留殘鎖。

這是 2026-09-29 那次 MQTT 互踢的根因修正：service 起了一份、手動除錯或 App 的
AI 啟動鈕又帶起一份，兩份用同一個 MQTT client id，在 broker 上互踢：

```
# journalctl -u mosquitto
Client client already connected, closing old connection.   # 一個下午 181 次
# app 這邊
Unexpected disconnection with reason: 7                    # 7 = MQTT_ERR_CONN_LOST
```

**注意**：MQTT client id 刻意維持固定（`yoshinoya-ai-<hostname>`，可用
`config.yaml` 的 `mqtt.client_id` 覆寫），不加 PID 或亂數。加了之後兩份 app 會
各自連上、不再互踢，但仍然同時搶相機與 APU —— 症狀被蓋掉，問題更難查。

手動除錯的操作規則（程式未更新的機器上尤其重要）：

```sh
systemctl stop yoshinoya_ai
systemctl mask --runtime yoshinoya_ai   # 重開機後自動失效
./run_app.sh
# 除錯結束
systemctl unmask --runtime yoshinoya_ai && systemctl start yoshinoya_ai
```

# DLA 載入失敗自動重開機

現場遇過開機時 APU 沒正常起來、DLA 模型載不進去，而且**不重開機不會恢復**
（service 的 `Restart=always` 會每秒重試，但每次都一樣失敗）。
`utils/dla_guard.py` 會在載入失敗時先體檢模型檔，再決定怎麼處置：

| 狀況 | 處置 |
| --- | --- |
| 檔案不存在 / 不是檔案 / 讀不到 / 大小 0 | 只記 `ERROR`，**絕不重開** |
| 檔案正常卻載入失敗 | 判定 APU 沒起來 → 記 `CRITICAL`、等 10 秒、`reboot` |
| 檔案讀得到但與上次成功載入時不同 | 疑似壞模型 → 只記 `CRITICAL`，**不重開** |
| 已連續自動重開 2 次仍失敗 | 停止自動重開，留給人工（機器還能 SSH） |

原生模組對三種失敗丟的是完全一樣的 `RuntimeError: Load <path> failed !`，
分不出原因，所以判斷只能靠上面的檔案體檢 —— 這也是為什麼「路徑打錯」不會
害機器重開。任何一次全部模型載入成功就會把計數歸零（`mark_ok()`），偶發的
APU 問題永遠有完整額度。狀態存在 `.dla_guard_state.json`（計數 + 模型指紋）。

要驗證整條路徑而不想真的重開機：

```sh
DLA_GUARD_DRYRUN=1 ./run_app.sh     # 照常判斷與寫 log，只是不下 reboot
```

`config.yaml` 不寫 `dla_guard` 這一段也能運作（全部欄位都有預設值）。要調整：

```yaml
dla_guard:
  enabled: true        # false = 只記 log，完全不自動重開
  max_reboots: 2       # 連續重開上限
  delay_sec: 1        # 下 reboot 前等幾秒
  reset_after_sec: 21600
```

# 設定檔參數說明

各參數的完整說明放在 profile 資料夾內，因為 `config.yaml` 會被程式改寫、註解會被抹掉：

- `tasks/ocr/yoshinoya/README.md`