# auto_update

把新版 `DLA_on_720.zip` 換上機器的腳本，給半夜自動更新用。

| 檔案 | 內容 |
| --- | --- |
| `auto_update.sh` | 解壓 → 舊版改名成 `ocr_mmr_dla_backup` → 新版改名成 `ocr_mmr_dla` → 合併現場設定 → 補 `aibox_uploader.sh` 執行權限 → 重啟服務；服務起不來就回滾 |
| `merge_config.py` | 把舊版 `tasks/ocr/yoshinoya/config.yaml` 的 `camera_params.hik.ReverseX/ReverseY.value`、`preset_roi.ticket_roi.points` 搬到新版（有 ruamel.yaml 保留格式，沒有就用 PyYAML） |

## 放哪裡

兩個檔案要一起放在機器的 **`/home/root/`**，不是 `ocr_mmr_dla` 裡面：這支腳本就是要把整個
`ocr_mmr_dla` 換掉，而且它用自己所在的資料夾去找 `merge_config.py`。新版 zip 也會帶著這個資料夾，
但不會自己覆蓋 `/home/root/` 那份，這裡改了要手動複製過去：

```sh
scp DLA_on_720/auto_update/auto_update.sh DLA_on_720/auto_update/merge_config.py root@<box-ip>:/home/root/
ssh root@<box-ip> 'chmod 755 /home/root/auto_update.sh'
```

## 使用

```sh
cd /home/root
chmod +x /home/root/auto_update.sh /home/root/merge_config.py
DRY_RUN=true ./auto_update.sh   # 只檢查 zip 結構，不動任何檔案
./auto_update.sh                # 需要 /home/root/DLA_on_720.zip
```

log 寫在 `/home/root/update_<時間>.log`。

## at 設定
```sh
# 先確認 atd 服務有跑
systemctl status atd

# 如果 atd 服務沒跑
systemctl start atd

# 排程
echo "/home/root/auto_update.sh" | at 02:00 tommorow

# 查看已排的工作
atq

# 取消某個工作 (job number 從 atq 來)
atrm <job_number>
```

## 注意

- **只合併上面三個值**。其他現場調過的相機參數（曝光、增益…）、`runtime.local.yaml`（只選品牌，不在 git 裡）、
  `.dla_guard_state.json` 都會回到新版 zip 的狀態。
- **`merge_config.py` 不見或合併失敗時**，腳本會停在「新版已換上、服務還沒重啟」：舊的 app 還在跑，
  下次重啟才會跑到新版，而且相機方向與 ROI 是 repo 的值。要人工檢查。
