set -euo pipefail

unzip -o ocr_mmr_dla.zip -d /home/root
#rm ocr_mmr_dla.zip

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
tail -n 30 logs/stream/$(date +%Y%m%d)-*.log