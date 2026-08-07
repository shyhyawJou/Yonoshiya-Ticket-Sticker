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