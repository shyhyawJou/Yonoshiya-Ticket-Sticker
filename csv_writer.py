import csv
import threading
import queue
import socket
import os
import json
import time
from datetime import datetime  
from loguru import logger

class CsvWriter:
    def __init__(self, log_dir: str):

        self.log_dir = log_dir
  
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.queue = queue.Queue()
        self.running = True
        
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _get_current_filename(self):
        """產生今天的檔名，例如: 20231027_{hostname}.csv"""
        today_str = datetime.now().strftime("%Y%m%d")
        full_hostname = socket.gethostname()
        host_suffix = full_hostname[-4:]
        filename = f"{today_str}_{host_suffix}.csv"
        return os.path.join(self.log_dir, filename)

    def _worker(self):
        fieldnames = [
            "tray_id", "start_time", "end_time", "expected_item", "expected_item_count",
            "checked_item", "ticket_capture", "final_tray_capture"
        ]

        while self.running:
            try:
                data = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                current_filename = self._get_current_filename()
                write_header = not os.path.exists(current_filename)
                
                with open(current_filename, mode='a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
   
                    row_to_write = {}
                    for k in fieldnames:
                        val = data.get(k)
                        if val is None:
                            row_to_write[k] = ""
                        elif isinstance(val, (dict, list)):
                            row_to_write[k] = json.dumps(val, ensure_ascii=False)
                        elif isinstance(val, float):
                            row_to_write[k] = round(val, 2)
                        else:
                            row_to_write[k] = val
                            
                    writer.writerow(row_to_write)
                    
                self.queue.task_done()
            except Exception as e:
                logger.error(f"CSV Write Error: {e}")

    def log(self, data: dict):
        self.queue.put(data)

    def stop(self):
        self.running = False
        self.thread.join()
        logger.info("Csv Writer stopped.")