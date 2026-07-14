# libMvCameraControl.so
目前可兼容版本是 4.6.1.3

# 執行程式
```
./run_app.sh
```

# 看 stream
```
ssh -L 9527:127.0.0.1:9527 root@{IP}
```

# 安裝程式 (快速非正式流程)
先把 ocr_mmr_dla 打包成 .zip, 上傳到一個 AI box 之後:
```
bash install_yoshinoya_ai.sh
```
這會把服務 enable 並且啟動 AI 端程式
