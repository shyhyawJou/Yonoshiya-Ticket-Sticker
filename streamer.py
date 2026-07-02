import cv2
import threading
import asyncio
import uvicorn
import queue
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from time import sleep, time
import numpy as np
from loguru import logger



class Mjpeg_Streamer:
    def __init__(self, 
                 host="0.0.0.0", 
                 port=9527, 
                 route="/meal", 
                 size=(640, 480), 
                 quality=60, 
                 enable=True):
        
        self.host = host
        self.port = port
        self.route = route
        self.stream_size = size
        self.quality = quality
        
        # 使用 Queue 來解耦主程序與處理程序
        self.frame_queue = queue.Queue(maxsize=1)
        self.processed_bytes = None  # 儲存處理完後的 JPEG bytes
        self._frame_lock = threading.Lock()
        self._frame_version = 0
        
        self.is_enable = enable
        self.is_running = False
        
        if not self.is_enable: 
            logger.info(f'Streamer is disabled.')
            return
            
        self.app = FastAPI()
        self._setup_routes()
        self.server_thread = None
        self.worker_thread = None
        self.server = None

    def _setup_routes(self):
        @self.app.get(self.route)
        async def video_feed():
            return StreamingResponse(self._generate(), 
                                     media_type="multipart/x-mixed-replace; boundary=frame")

    def _worker(self):
        """背景處理執行緒：負責消耗 Queue 中的影像並進行編碼"""
        while self.is_running:
            try:
                # 取得原始影格，設定 timeout 避免死鎖
                frame = self.frame_queue.get(timeout=1)
                
                # 在背景執行緒處理耗時的 resize 與編碼
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
                
                if ret:
                    # 封裝成標準的 MJPEG 影格格式
                    jpeg_bytes = (b'--frame\r\n'
                                  b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    with self._frame_lock:
                        self.processed_bytes = jpeg_bytes
                        self._frame_version += 1
                
                # 完成後標記任務
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker processing error: {e}")

    async def _generate(self):
        """ 發送執行緒：只在有『新』畫面時才送出，避免對慢速連線（如 ssh -L）
        重複灌入同一張舊畫面，塞爆 tunnel 造成延遲累積 """
        logger.info("Stream generator started.")
        last_sent_version = -1
        try:
            while self.is_running:
                with self._frame_lock:
                    current_version = self._frame_version
                    data = self.processed_bytes

                # 還沒有畫面，或畫面沒有更新 -> 不送，稍微等一下再檢查
                if data is None or current_version == last_sent_version:
                    await asyncio.sleep(0.01)
                    continue

                last_sent_version = current_version
                yield data

                # 讓 event loop 有機會處理其他任務，
                # 同時避免同一輪迴圈把 CPU 吃滿
                await asyncio.sleep(0)
        except Exception as e:
            logger.debug(f"Streaming connection closed: {e}")
        finally:
            logger.warning('Stream generator loop exited')

    def start(self):
        """啟動伺服器與背景處理執行緒"""
        if self.is_running:
            logger.info("[!] Streamer is already running.")
            return
        
        if not self.is_enable:
            return

        self.is_running = True
        
        # 啟動處理影像的 Worker
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
        # 設定 Uvicorn
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="error")
        self.server = uvicorn.Server(config)

        def run_server():
            self.server.run()

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        logger.success(f"[*] MJPEG Streamer started at http://{self.host}:{self.port}{self.route}")

    def stop(self):
        """釋放資源"""
        if not self.is_running:
            return

        logger.info("[*] Stopping MJPEG Streamer...")
        self.is_running = False 
        
        # 1. 停止 Uvicorn Server (這會主動切斷所有 FastAPI 的 StreamingResponse)
        if self.server:
            self.server.should_exit = True
        
        # 2. 清空 Queue 並放入一個 None 作為 Sentinel (哨兵值) 讓 Worker 退出
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        # 4. 等待執行緒結束
        if self.server_thread:
            self.server_thread.join(timeout=10)
        if self.worker_thread:
            self.worker_thread.join(timeout=10)
            
        logger.success("[*] MJPEG Streamer resources released.")

    def push_frame(self, frame):
        """更新影像：現在這對主程序來說非常快"""
        if self.is_running and self.is_enable:
            try:
                # 如果 Queue 滿了，取出最舊的幀（丟棄），確保即時性
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                # 放入新幀，非阻塞型態
                self.frame_queue.put_nowait(frame)
            except Exception:
                # 避免主程序因為推播錯誤而中斷
                pass