#!/usr/bin/env bash
# aibox_uploader.sh —— 給 crontab 每天執行一次的啟動腳本
#
#   1. 從 /etc/yoshinoya_ai/cloud_upload.env 取得金鑰，export YOSHINOYA_API_KEY
#   2. 取得店家代碼            → aibox_uploader.py --store
#      （STORE_CODE → cloud_upload.env 的 YOSHINOYA_STORE_CODE → hostname 最後一個「-」之後的部分）
#   3. 上傳檔案的根目錄（常數）→ --prefix-path
#   4. 計算昨天的日期          → --date（呼叫時已經給了 --date 就不算）
#   5. 設定檔（JSON）          → --config
#   然後執行 aibox_uploader.py，結束代碼原樣回傳（0 = 成功、1 = 有失敗、2 = 設定錯誤）。
#
# 用法：
#   aibox_uploader.sh [MODE] [其他 aibox_uploader.py 參數…]
#     MODE：all（預設）/ csv / log / video / image
#     其他參數放在後面，會蓋過這支腳本算出來的值；給了 --date 就用它、不再加昨天的日期，例如手動補傳指定日：
#       aibox_uploader.sh all --date 20261010
#       aibox_uploader.sh video --date 20261010 --dry-run
#
# crontab（crontab -e；每天 03:10 執行，輸出另外留一份）：
#   10 3 * * * /opt/yoshinoya_ai/aibox_uploader.sh all >> /var/log/yoshinoya_ai/aibox_uploader.cron.log 2>&1
#
# 下面的「常數」都可以用同名環境變數覆寫（測試時方便），例如 PREFIX_PATH=/mnt/data aibox_uploader.sh all
set -euo pipefail

# ── 常數（依門市環境修改）──────────────────────────────────────────────────
ENV_FILE="${ENV_FILE:-/etc/yoshinoya_ai/cloud_upload.env}"          # 機密設定：YOSHINOYA_API_KEY、YOSHINOYA_CA、YOSHINOYA_STORE_CODE
CONFIG_FILE="${CONFIG_FILE:-/etc/yoshinoya_ai/aibox_uploader.json}" # aibox_uploader.py 的設定檔
PREFIX_PATH="${PREFIX_PATH:-/mnt/reserved/record/stream}"           # 上傳檔案的根目錄：<PREFIX_PATH>/<YYYYMMDD>/csv_logs …
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPLOADER="${UPLOADER:-$SCRIPT_DIR/aibox_uploader.py}"
PYTHON="${PYTHON:-python3}"
LOCK_FILE="${LOCK_FILE:-/tmp/aibox_uploader.lock}"

log() { printf '%s [aibox_uploader.sh] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }

# 讀 KEY=VALUE 設定檔裡的一個值（最後一次出現的為準）。
# 不用 source：設定檔寫的是「KEY= value」（= 後面有空白），source 會把值當成指令執行。
# 規則與 aibox_upload.read_env_file() 相同：# 開頭為註解、可有 export、成對的引號會去掉、沒引號時「 #」之後是註解。
env_value() {
  local key="$1" file="$2" line value
  [ -r "$file" ] || return 1
  line="$(grep -E "^[[:space:]]*(export[[:space:]]+)?${key}[[:space:]]*=" "$file" | tail -n 1)" || return 1
  [ -n "$line" ] || return 1
  value="${line#*=}"
  value="${value//$'\r'/}"                                    # Windows 上編輯過的 CRLF
  value="$(printf '%s' "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  case "$value" in
    \"*\" | \'*\') value="${value:1:${#value}-2}" ;;
    *) value="${value%% #*}"; value="$(printf '%s' "$value" | sed -e 's/[[:space:]]*$//')" ;;
  esac
  printf '%s' "$value"
}

MODE="${1:-all}"
[ $# -gt 0 ] && shift
case "$MODE" in
  all|csv|log|video|image) ;;
  *) log "錯誤：上傳方式必須是 all / csv / log / video / image，實際為「$MODE」"; exit 2 ;;
esac

# ── 1. API 金鑰：從設定檔取得並 export（設定檔沒有時，沿用已經設定的環境變數）────────
KEY="$(env_value YOSHINOYA_API_KEY "$ENV_FILE" || true)"
if [ -n "$KEY" ]; then
  export YOSHINOYA_API_KEY="$KEY"
  log "API 金鑰：取自 $ENV_FILE"                              # 只記來源，不印出金鑰
elif [ -n "${YOSHINOYA_API_KEY:-}" ]; then
  log "API 金鑰：$ENV_FILE 裡沒有（或讀不到），沿用環境變數 YOSHINOYA_API_KEY"
else
  log "錯誤：找不到 API 金鑰（$ENV_FILE 不存在、讀不到，或沒有 YOSHINOYA_API_KEY）"
  exit 2
fi
unset KEY
# 憑證路徑也在同一個設定檔：有寫就一併 export（aibox_uploader.py 本身也會讀，這裡只是讓兩邊一致）
CA="$(env_value YOSHINOYA_CA "$ENV_FILE" || true)"
if [ -n "$CA" ] && [ -z "${YOSHINOYA_CA:-}" ]; then
  export YOSHINOYA_CA="$CA"
fi

# ── 2. 店家代碼：環境變數 STORE_CODE → 設定檔的 YOSHINOYA_STORE_CODE → hostname ─────
# hostname 的最後一個「-」之後就是店家代碼，例如 aibox-07c6 → 07c6
STORE="${STORE_CODE:-}"
STORE_FROM="環境變數 STORE_CODE"
if [ -z "$STORE" ]; then
  STORE="$(env_value YOSHINOYA_STORE_CODE "$ENV_FILE" || true)"
  STORE_FROM="$ENV_FILE 的 YOSHINOYA_STORE_CODE"
fi
if [ -z "$STORE" ]; then
  STORE="$(hostname | awk -F- "{print \$NF}" || true)"
  STORE_FROM="hostname（$(hostname 2>/dev/null || true)）"
fi
STORE_ARGS=()
if [ -n "$STORE" ]; then
  STORE_ARGS=(--store "$STORE")
  log "店家代碼：$STORE（取自 $STORE_FROM）"
else
  log "沒有店家代碼（STORE_CODE、$ENV_FILE 的 YOSHINOYA_STORE_CODE、hostname 都取不到），交給 $CONFIG_FILE 的 store"
fi

# ── 3. 上傳檔案的根目錄、4. 昨天的日期（本機時間）──────────────────────────
# 呼叫時已經給了 --date（--date 20261010 或 --date=20261010）就用它，不再另外加昨天的日期
DATE_ARGS=()
HAS_DATE=0
for arg in "$@"; do
  case "$arg" in
    --date|--date=*) HAS_DATE=1 ;;
  esac
done
if [ "$HAS_DATE" -eq 0 ]; then
  YESTERDAY="$(date -d yesterday +%Y%m%d 2>/dev/null || date -v-1d +%Y%m%d)"   # GNU date；BSD/macOS 用 -v-1d
  DATE_ARGS=(--date "$YESTERDAY")
fi

# ── 5. 設定檔 ──────────────────────────────────────────────────────────────
CONFIG_ARGS=()
if [ -f "$CONFIG_FILE" ]; then
  CONFIG_ARGS=(--config "$CONFIG_FILE")
else
  log "找不到設定檔 $CONFIG_FILE，aibox_uploader.py 改用內建預設值"
fi

# 腳本算出來的參數放前面、呼叫者另外給的放後面：同一個參數給兩次時，argparse 以最後一個為準，
# 所以呼叫者給的 --store 等會蓋過腳本算的值（--date 則上面已經處理，不會出現兩次）。
# 設定檔的值則一律被命令列參數蓋過。
# ${A[@]+"${A[@]}"}：陣列是空的時候展開成空，bash 4.4 以前在 set -u 下也不會報 unbound variable
CMD=("$PYTHON" "$UPLOADER" "$MODE" ${STORE_ARGS[@]+"${STORE_ARGS[@]}"} --prefix-path "$PREFIX_PATH"
     ${DATE_ARGS[@]+"${DATE_ARGS[@]}"} ${CONFIG_ARGS[@]+"${CONFIG_ARGS[@]}"} "$@")
log "執行：${CMD[*]}"

# 前一次還沒跑完（例如影片很多、網路很慢）就不要再開一個
if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    log "前一次的上傳還在執行（$LOCK_FILE 被鎖住），這次不執行"
    exit 1
  fi
fi

exec "${CMD[@]}"
