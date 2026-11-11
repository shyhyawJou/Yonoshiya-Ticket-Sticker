#!/bin/bash
#
# auto_update.sh
# 自動更新 ocr_mmr_dla:解壓新版(DLA_on_720) -> 舊版變 backup ->
# 新版改名成 ocr_mmr_dla -> 搬移相機校正參數 -> 重啟服務
#
# 用法: ./auto_update.sh
# 建議先跑 DRY_RUN=true ./auto_update.sh 確認流程,不會真的動任何檔案
#
# 原始檔在 repo 的 DLA_on_720/auto_update/,機器上要跟 merge_config.py 一起放在
# /home/root/ 執行 (不能在 ocr_mmr_dla 裡面跑:這支腳本就是要把整個 ocr_mmr_dla 換掉)。
# 新版 zip 裡帶的這份不會自己覆蓋 /home/root/ 的那份,改了要手動複製過去。
#
set -euo pipefail

# ========== 基本設定,依實際情況調整 ==========
BASE_DIR="/home/root"
ZIP_FILE="${BASE_DIR}/DLA_on_720.zip"
EXTRACTED_DIR="${BASE_DIR}/DLA_on_720"      # zip 解壓後最上層資料夾名稱 (已用 unzip -l 確認)
OLD_DIR="${BASE_DIR}/ocr_mmr_dla"
BACKUP_DIR="${BASE_DIR}/ocr_mmr_dla_backup"
CONFIG_REL="tasks/ocr/yoshinoya/config.yaml"
SERVICE_NAME="yoshinoya_ai"
APP_SERVICE_NAME="yoshinoya"
MERGE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGFILE="${BASE_DIR}/update_$(date +%Y%m%d_%H%M%S).log"

DRY_RUN="${DRY_RUN:-false}"

exec > >(tee -a "$LOGFILE") 2>&1
echo "=== [auto_update] 開始於 $(date) ==="
echo "DRY_RUN=${DRY_RUN}"

fail() {
    echo "ERROR: $1"
    exit 1
}

# ========== 0. 前置檢查 ==========
[ -f "$ZIP_FILE" ] || fail "找不到 $ZIP_FILE"
[ -d "$OLD_DIR" ]  || fail "找不到目前執行中的 $OLD_DIR"
command -v unzip >/dev/null 2>&1 || fail "找不到 unzip 指令"
command -v systemctl >/dev/null 2>&1 || fail "找不到 systemctl"

cd "$BASE_DIR"

# 清掉上次可能殘留的解壓資料夾,避免 unzip 詢問覆蓋
rm -rf "$EXTRACTED_DIR"

# ========== 1. 解壓新版 (跟現有 ocr_mmr_dla 是不同資料夾,不會互相覆蓋) ==========
echo "解壓 $ZIP_FILE ..."
unzip -q "$ZIP_FILE" -d "$BASE_DIR"

[ -d "$EXTRACTED_DIR" ] || fail "解壓後找不到 $EXTRACTED_DIR,請確認 zip 內部結構是否有變"

NEW_CFG_STAGED="${EXTRACTED_DIR}/${CONFIG_REL}"
OLD_CFG_CHECK="${OLD_DIR}/${CONFIG_REL}"
[ -f "$NEW_CFG_STAGED" ]  || fail "新版套件裡找不到 $CONFIG_REL (路徑: $NEW_CFG_STAGED)"
[ -f "$OLD_CFG_CHECK" ]   || fail "目前執行中的版本裡找不到 $CONFIG_REL"

if [ "$DRY_RUN" = "true" ]; then
    echo "[dry-run] 檢查通過,以下動作將會被執行 (但不實際執行):"
    echo "  1. rm -rf $BACKUP_DIR"
    echo "  2. mv $OLD_DIR $BACKUP_DIR"
    echo "  3. mv $EXTRACTED_DIR $OLD_DIR"
    echo "  4. 合併 ReverseX/ReverseY/ticket_roi.points 到新 config"
    echo "  5. chmod 755 aibox_uploader.sh, systemctl restart $SERVICE_NAME"
    echo "[dry-run] 清理本次測試解壓出來的 $EXTRACTED_DIR"
    rm -rf "$EXTRACTED_DIR"
    exit 0
fi

# ========== 2. 舊版轉 backup ==========
echo "刪除舊的 $BACKUP_DIR ..."
rm -rf "$BACKUP_DIR"

echo "將目前執行中的版本 $OLD_DIR 改名為 $BACKUP_DIR ..."
mv "$OLD_DIR" "$BACKUP_DIR"

# ========== 3. 新版上線:把解壓出來的資料夾改名成 ocr_mmr_dla ==========
echo "將 $EXTRACTED_DIR 改名為 $OLD_DIR ..."
mv "$EXTRACTED_DIR" "$OLD_DIR"

# ========== 4. 合併相機校正參數 (ReverseX, ReverseY, ticket_roi.points) ==========
OLD_CFG="${BACKUP_DIR}/${CONFIG_REL}"
NEW_CFG="${OLD_DIR}/${CONFIG_REL}"

[ -f "$OLD_CFG" ] || fail "備份版找不到 config.yaml: $OLD_CFG"
[ -f "$NEW_CFG" ] || fail "新版找不到 config.yaml: $NEW_CFG"

echo "合併相機校正參數 ..."
python3 "${MERGE_SCRIPT_DIR}/merge_config.py" "$OLD_CFG" "$NEW_CFG" \
    || fail "config 合併失敗,新版 $OLD_DIR 內容可能不完整,請人工檢查後決定是否手動 rollback"

# ========== 5. 重啟服務並確認存活 ==========
# 雲端上傳腳本由 crontab 直接執行,zip 解開後可能沒有執行權限 (見 install_crontab.sh)。
# 失敗只警告不中止:這時新版已經換上,在這裡 exit 會停在「新版已上、服務沒重啟」的狀態。
chmod 755 /home/root/ocr_mmr_dla/aibox_uploader/aibox_uploader.sh \
    || echo "WARNING: chmod aibox_uploader.sh 失敗,01:00 的雲端上傳可能不會跑,請人工檢查"

echo "重啟服務 $SERVICE_NAME ..."
systemctl restart "$APP_SERVICE_NAME"
systemctl restart "$SERVICE_NAME"
sleep 3

if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "服務啟動成功。更新完成於 $(date)。舊版保留於 $BACKUP_DIR"
else
    echo "服務啟動失敗,開始回滾 ..."
    rm -rf "$OLD_DIR"
    mv "$BACKUP_DIR" "$OLD_DIR"
    systemctl restart "$SERVICE_NAME" || true
    fail "更新失敗已回滾到舊版,請檢查 $LOGFILE 與 journalctl -u $SERVICE_NAME"
fi

echo "=== [auto_update] 結束於 $(date) ==="