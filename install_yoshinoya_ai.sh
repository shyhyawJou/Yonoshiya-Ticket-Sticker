set -euo pipefail

unzip -o ocr_mmr_dla.zip -d /home/root

pip3 install \
    shapely \
    pyclipper \
    rapidfuzz \
    uvicorn \
    fastapi \
    loguru

cd /home/root/ocr_mmr_dla
cp -f yoshinoya_ai.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable yoshinoya_ai.service
systemctl restart yoshinoya_ai.service
systemctl status yoshinoya_ai.service --no-pager
#tail -n 30 logs/stream/$(date +%Y%m%d)-*.log

if systemctl is-active --quiet yoshinoya_ai.service; then
    rm -f ~/ocr_mmr_dla.zip
else
    echo "yoshinoya_ai.service failed to start, keeping ocr_mmr_dla.zip"
    exit 1
fi

ps aux | grep python