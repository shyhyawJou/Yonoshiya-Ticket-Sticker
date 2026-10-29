# DLA 品牌設定與本機切換

更新：2026-09-14。此調整針對 DLA 主程式；NUC 入口與計數行為沒有改動。

## 入口與優先順序

共用 runtime.yaml 預設 profile: yoshinoya。本機 runtime.local.yaml 已選 suzumo，受 .gitignore 排除。命令列 > 本機 profile > 共用 profile。

在 DLA_on_720 執行：

```sh
python3 app.py                     # 本機預設 Suzumo；沒有本機檔則吉野家
python3 app.py --profile yoshinoya  # 臨時指定吉野家
python3 app.py --profile suzumo
./run_app.sh --profile suzumo      # edge 的相機環境準備後啟動
```

入口需在安裝 DLA 與相機 SDK 的設備執行，Windows 測試只驗證純 Python 邏輯。不要為了切品牌同時開兩個實體相機服務。

## 檔案分工

- tasks/ocr/yoshinoya/config.yaml：遠端吉野家模型設定、DET、legacy 收據解析。
- tasks/ocr/suzumo/config.yaml：本機 Suzumo 模型設定、收據 cv_table、貼紙 det、table 收據解析。
- 各品牌 classes.yaml、menus_mapping.yaml、japan_dict.txt：配套資料。
- Suzumo 的 vocab_ticket.txt／vocab_sticker.txt：限制辨識詞彙；吉野家未指定時使用菜單候選。
- 模型暫時保留 tasks/ocr 原位置，各品牌以 ../模型檔名引用；不複製或新增提交模型。
- profile_config.py：選擇品牌；所有實際設定只從品牌 config.yaml 載入。
- config.py：載入選定 YAML 與品牌資料。
- manager/task_context.py：把設定傳給模型、OCR 與 LogicEngine。
- logic/szm/：Suzumo 專用。order_parser.py（表格／數量解析）、ticket_layout.py（表格版面）、cv_text_regions.py（純 CV 文字區域）、order_parser_legacy_reference.py（eval 工具的新舊比較基準）。
- logic/yoshinoya/order_parser.py：以 origin/main 的原版解析建立的獨立策略（原 logic/order_parser_yoshinoya.py）。
- logic/yoshinoya/ticket_cv_layout.py、ticket_cv_assist.py：吉野家大傳票的 CV 輔助（config 的 ticket_cv_assist，預設關閉）。表格框線決定方向與列欄、det 框分格、漏框補讀；找不到表格時退回 order_parser.py 原流程。跟 ocr_routes 無關（ocr_routes 仍是 det）。
- 兩品牌共用的 reader 流程與貼紙比對在 logic/readers/（sticker_matcher.py 也在這裡）。
- logic_engine.py：依 ticket_parser 選 parser，依品牌 object_class_items 分流配件。
- tracker、pending buffer、state machine：取得該任務自己的映射，沒有更改全域映射來切品牌。

## 本機品牌選擇

```yaml
profile: suzumo
```

runtime.local.yaml 只允許 profile，用來決定這台機器執行 yoshinoya 或 suzumo。沒有本機檔時使用 runtime.yaml 的共用預設；命令列 --profile 可暫時覆蓋品牌選擇。

模型、OCR 路由、MQTT、相機參數及其他實際設定全部維護在 tasks/ocr/<品牌>/config.yaml。Hik 相機經 UI／MQTT 調整後，也會寫回目前品牌的 config.yaml，不會建立 runtime.local.yaml。舊版 local 內若有 camera_params，首次載入會先備份 local、將參數搬回對應品牌 config，再把 local 正規化成只剩 profile；其他舊 override 會停止生效並保留在備份中。

## 映射與切換

menus_mapping.yaml 的 mapping 定義套餐子品項，object_class_items 定義不用 OCR 的配件身份。

Suzumo 例如 chopsticks: 箸；吉野家則保留 sesame_front/back 等既有對應。載入時驗證映射中的類別存在於 classes.yaml。

ocr_routes 控制文字框來源；ticket_parser 控制辨識完後的解析策略，兩者不同：改成 DET 不代表解析器自動變成另一品牌的版面算法。

DLA 核對數量目前仍採既有 confirmed_counts 歷史最高進度，沒有在此次加入新的計數模式；NUC 仍是可見數量。DLA 對 NUC 純 CV 檔案的引用也尚未抽離。

## 根層清理與工具

根層 config.yaml、classes.yaml、menus_mapping.yaml、japan_dict.txt、vocab_ticket.txt、vocab_sticker.txt 已移除。品牌資料只維護 tasks/ocr/yoshinoya 與 tasks/ocr/suzumo 兩處；根層保留模型、字型等共用資源。

移除前已比對：舊設定／類別對應 Suzumo，舊菜單 mapping 對應吉野家，字典及詞彙已有品牌副本。本機原始檔備份在 outputs/config-migration-backup-20260914，受 Git ignore 排除，不作為執行來源。

主程式與相機獨立入口共用 profile_config.load_profile_config，附帶目前品牌與設定檔路徑；相機儲存寫回該品牌 config.yaml。相機獨立測試請在 DLA_on_720 執行 `python3 -m stream_vision.hikcam --profile yoshinoya`（需 SDK）。

`tools/ocr_probe.py --profile suzumo --type ticket <裁切圖片> --backend onnx` 可指定品牌；未指定時依 runtime.yaml／runtime.local.yaml 選擇。DLA 模型讀 DLA 品牌設定，ONNX 替代測試讀 NUC 同品牌模型設定。解析器與候選詞使用 DLA 品牌設定。

注意：ocr_probe 與 eval 工具固定使用 DET 供離線比較，不是 app 的 cv_table 路由驗收；數量裁圖補辨識也未啟用。驗證正式 CV 路由請使用 app。

既有 eval_ticket_parsing、eval_sticker、eval_orientation 是 Suzumo 資料集評估，明確選 Suzumo 模型；新舊比較的舊菜單基準明確讀 yoshinoya/menus_mapping.yaml。eval_parser_ideal 使用標註文字，不執行模型。資料集真值仍取自 images。更換模型後請重建快取（支援的工具用 --refresh；orientation 刪除其 det_original.json 後重跑）。本次改用 _eval_cache_suzumo_profiles_v2，避免沿用舊模型結果。

部署請打包品牌資料夾；runtime.local.yaml 若存在只能保留 profile。排除本機 outputs 與 runtime.local.yaml 備份。舊版 local 相機覆寫由程式首次啟動時搬回品牌 config.yaml；舊設備若仍使用根層 config.yaml，升級前仍需人工確認設定歸屬。

## 驗證

本機 52 項測試通過（DLA 27、NUC 25），涵蓋品牌載入、無根層舊檔時的設定、工具模型選擇與原有解析／計數邏輯。OCR 工具亦成功以 auto 後端載入 Suzumo ONNX 模型並完成空白影像推論。實體 DLA 推論、相機控制與兩品牌完整現場流程仍需 edge 驗收。

```powershell
..\ONNX_on_nuc\.venv\Scripts\python.exe -m pytest unit_test/test_profiles.py unit_test/test_ticket_layout.py unit_test/test_order_parser_and_sticker_matcher.py -q
```
