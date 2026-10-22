#!/bin/bash
#
# install_crontab.sh
#
# 功能：
#   1. 檢查並安裝 rclone
#   2. 從專案內的 rclone_settings/rclone.conf 複製金鑰到 /root/.config/rclone/rclone.conf
#      （如果專案內沒有該檔案，會 fallback 檢查系統是否已有現成金鑰）
#   3. 建立 log 目錄
#   4. 寫入 crontab 排程（自動上傳 stream 影片到 onedrive）
#
# 排程時間：凌晨 03:05 起約兩小時（03:05 / 03:35 / 04:05 / 04:35 觸發，flock 防重疊；
#   主機當地時間，之後主機會實際跑在日本，故即為日本時間）。
#   01:00 起是雲端上傳（同一個排程區塊裡的 aibox_uploader.sh，見 docs/cloud_upload.md）。
#   它沒有截止時間：跑到 03:05 還沒傳完的部分會被 rclone 搬走。
#   兩者處理的是同一批「昨天」的資料夾，而 rclone move 會刪除本機檔，所以順序不能反過來。
#   選在午夜過後才跑，是因為此時「今天」(剛開始的新一天) 才是唯一還在持續寫入的
#   資料夾；「昨天」以前的影片/圖片、csv_logs、stream_log 都已經寫完關檔，
#   因此不再需要像之前那樣分兩段、排除 csv_logs/stream_log 再延一天搬——
#   全部資料型態都在同一次排程、同一個時間範圍 (seq 1 10，即昨天往前10天) 一起搬走即可。
#
# 使用方式：
#   chmod +x install_crontab.sh
#   sudo ./install_crontab.sh
#
# 重複執行是安全的：舊的排程區塊會先被移除，再寫入新的，不會疊加。

set -e

# ===== 基本設定 =====
RCLONE_CONF_DIR="/root/.config/rclone"
RCLONE_CONF_FILE="${RCLONE_CONF_DIR}/rclone.conf"

# 腳本所在目錄（讓這個 sh 檔可以放在專案任何位置都找得到 rclone_settings）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETTINGS_CONF_FILE="${SCRIPT_DIR}/rclone_settings/rclone.conf"

CRON_MARKER_START="# >>> onedrive-auto-upload start >>>"
CRON_MARKER_END="# <<< onedrive-auto-upload end <<<"

echo "===== 1. 檢查 rclone 是否已安裝 ====="
if ! command -v rclone >/dev/null 2>&1; then
    echo "未偵測到 rclone，開始安裝..."
    curl https://rclone.org/install.sh | bash
else
    echo "rclone 已安裝：$(command -v rclone)"
fi

echo ""
echo "===== 2. 設定 rclone 金鑰 ====="
mkdir -p "$RCLONE_CONF_DIR"

if [ -f "$SETTINGS_CONF_FILE" ]; then
    echo "找到專案內建金鑰檔案：$SETTINGS_CONF_FILE"
    cp "$SETTINGS_CONF_FILE" "$RCLONE_CONF_FILE"
    echo "已複製到 $RCLONE_CONF_FILE"
elif [ -f "$RCLONE_CONF_FILE" ]; then
    echo "專案內沒有 rclone_settings/rclone.conf，但系統已有現成金鑰，沿用：$RCLONE_CONF_FILE"
else
    echo "錯誤：找不到金鑰檔案"
    echo "  - 專案內：$SETTINGS_CONF_FILE（不存在）"
    echo "  - 系統：$RCLONE_CONF_FILE（不存在）"
    echo "請將 rclone.conf 放到 ${SCRIPT_DIR}/rclone_settings/ 之後再執行此腳本"
    exit 1
fi

if ! grep -q "^\[onedrive\]" "$RCLONE_CONF_FILE" 2>/dev/null; then
    echo "警告：$RCLONE_CONF_FILE 內找不到 [onedrive] 設定區塊，請確認金鑰內容是否正確"
fi

echo ""
echo "===== 3. 建立紀錄目錄 ====="
mkdir -p /mnt/reserved/restart_logs/
echo "已確認 /mnt/reserved/restart_logs/ 存在"

echo ""
echo "===== 4. 寫入 crontab ====="

# 用暫存檔組出這次要寫入的排程區塊，避免跳脫字元問題
CRON_BLOCK_FILE=$(mktemp)
cat > "$CRON_BLOCK_FILE" <<'CRONEOF'
# >>> onedrive-auto-upload start >>>
0 1 * * * /home/root/ocr_mmr_dla/aibox_uploader/aibox_uploader.sh
# 03:05 起才跑：01:00 起是雲端上傳（上面那一行，沒有截止時間）。
# 此時「今天」(新的一天) 才是唯一還在持續寫入的資料夾，
# 所以只搬 seq 1 10（昨天往前10天）：影片、圖片、csv_logs、stream_log
# 全部都已經寫完關檔，不需要再排除或延一天，一次搬完即可。
0 0 * * * mkdir -p /mnt/reserved/restart_logs/
5,35 3-4 * * * flock -n /tmp/rclone_stream_video.lock sh -lc 'D=$(hostname | awk -F- "{print \$NF}"); RC="/usr/bin/rclone --config /root/.config/rclone/rclone.conf"; OPT="--transfers 2 --timeout 30s --contimeout 30s --retries 5 --low-level-retries 10 --delete-empty-src-dirs --log-level INFO"; for i in $(seq 1 10); do Y=$(date -d "-$i day" +\%Y\%m\%d); $RC move "/mnt/reserved/record/stream/$Y" "onedrive:/upload/yoshinoya/$D/$Y/" $OPT --log-file="/mnt/reserved/restart_logs/rclone_stream_${D}_${Y}.log"; rmdir "/mnt/reserved/record/stream/$Y" 2>/dev/null; done'
# <<< onedrive-auto-upload end <<<
CRONEOF

# 取出現有 crontab（若沒有則視為空）
TMP_CRON=$(mktemp)
crontab -l 2>/dev/null > "$TMP_CRON" || true

# 移除舊的排程區塊（如果存在），避免重複執行時疊加
sed -i "/${CRON_MARKER_START}/,/${CRON_MARKER_END}/d" "$TMP_CRON"

# 加上新的排程區塊
cat "$CRON_BLOCK_FILE" >> "$TMP_CRON"

crontab "$TMP_CRON"
rm -f "$TMP_CRON" "$CRON_BLOCK_FILE"

echo "已更新 crontab，目前內容如下："
echo "-----------------------------------"
crontab -l
echo "-----------------------------------"

echo ""
echo "===== 完成 ====="
