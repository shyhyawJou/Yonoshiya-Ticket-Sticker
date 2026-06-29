#!/bin/sh

# --- 1. GigE 相機網路環境設定 ---
echo "Configuring GigE camera network environment..."
sysctl -w net.ipv4.conf.all.rp_filter=0 >/dev/null
sysctl -w net.ipv4.conf.default.rp_filter=0 >/dev/null

# 偵測網路介面：排除 can0、lo、wlp1s0，找到實體有線網卡或 USB 網卡
# 可透過環境變數 GIGE_DEV 強制指定介面名稱（例如 enu1c2）
if [ ! -z "$GIGE_DEV" ]; then
    REAL_DEV="$GIGE_DEV"
    REAL_IP=$(ip -o addr show dev "$REAL_DEV" 2>/dev/null | grep "inet " | awk '{print $4}' | cut -d'/' -f1 | head -n 1)
else
    REAL_DEV=$(ip -o addr show | grep "inet " | grep "169.254." | grep -v -E "can[0-9]|\\blo\\b" | awk '{print $2}' | head -n 1)
    REAL_IP=$(ip -o addr show | grep "inet " | grep "169.254." | grep -v -E "can[0-9]|\\blo\\b" | awk '{print $4}' | cut -d'/' -f1 | head -n 1)
fi

if [ ! -z "$REAL_DEV" ] && [ ! -z "$REAL_IP" ]; then
    echo "Detected Camera Interface: $REAL_DEV, IP: $REAL_IP"
    sysctl -w net.ipv4.conf.${REAL_DEV}.rp_filter=0 >/dev/null 2>&1
    ip route del 169.254.0.0/16 2>/dev/null
    ip route add 169.254.0.0/16 dev ${REAL_DEV} proto kernel scope link src ${REAL_IP} 2>/dev/null
else
    echo "Warning: No valid 169.254.x.x ethernet interface detected (Skipping custom route)."
    echo "  Hint: Set GIGE_DEV=<interface> to force (e.g. export GIGE_DEV=enu1c2)"
fi

sysctl -w net.core.rmem_max=26214400 >/dev/null
sysctl -w net.core.rmem_default=26214400 >/dev/null

# --- 2. USB3 相機核心記憶體設定 ---
echo "Configuring USB3 camera memory..."
if [ -f /sys/module/usbcore/parameters/usbfs_memory_mb ]; then
    echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb
fi
chmod 666 /dev/bus/usb/*/* 2>/dev/null

# --- 3. 設定環境變數並啟動 ---
export MVCAM_COMMON_RUNENV="/usr/lib"
python3 app.py
