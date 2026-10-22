from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Callable, List, Union
import paho.mqtt.client as mqtt
from loguru import logger


def default_client_id() -> str:
    """AI 這條連線的 client id，例如 `yoshinoya-ai-aibox-7d2f`。

    原本寫死成 `"client"`，太通用：任何測試腳本或工具沒指定 id 用了同名，就會
    在 broker 上把 AI 踢掉（MQTT 規範要求 broker 在「同一個 client id 又連進來」
    時必須斷開舊連線），而 `loop_start()` 底下的 `loop_forever()` 會自動重連
    （paho 2.0，`reconnect_on_failure` 預設 True）—— 兩邊就無限互踢。

    2026-09-29 現場實證（`journalctl -u mosquitto`）：

        Client client already connected, closing old connection.
        ...同一個下午 181 次

    app 這邊對應的就是連續的 `Unexpected disconnection with reason: 7`
    （7 = paho 的 `MQTT_ERR_CONN_LOST`）。當時互踢的是**兩份同時在跑的 app.py**。

    **id 刻意保持固定，不加 PID 也不加亂數。** 加了之後兩份 app 會各自連上、
    不再互踢，但它們仍然同時搶 HikRobot 相機、DLA/APU 與 /mnt/reserved 的寫入 ——
    症狀被蓋掉，問題更難發現。「只能跑一份」是由 `app.py` 的單例鎖
    （`acquire_single_instance_lock`）保證的，不是靠 id 去閃避。
    固定 id 的附帶好處是：broker log 一眼就看得出是誰連上來。
    """
    return f"yoshinoya-ai-{socket.gethostname()}"


# paho 錯誤碼的人話對照，只列這裡真的會看到的幾個
_PAHO_RC_HINT = {
    1: "（記憶體不足）",
    4: "（沒有連線）",
    5: "（連線被拒，認證問題？）",
    7: "（連線遺失：broker 重啟，或有人用同一個 client_id 把我們踢掉）",
    16: "（keepalive 逾時）",
}


@dataclass
class MqttSettings:
    host: str = "localhost"
    port: int = 1883
    # 固定且有辨識度的 id（見 default_client_id 的說明）。
    # 要換名字可以在 config.yaml 的 mqtt.client_id 指定。
    client_id: str = field(default_factory=default_client_id)
    base_topic: str = "demo"
    session: str = "default"
    keepalive: int = 60
    lwt_enabled: bool = True

class MqttBus:
    """
    Provides:
      - connect()/disconnect() with loop_start()
      - on_command(cb): subscribe and dispatch commands from bento/v1/<session>/cmd/#
    """
    def __init__(self, settings: Optional[MqttSettings] = None) -> None:
        self.s = settings or MqttSettings()
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=self.s.client_id)
        self._on_cmd_cb: Optional[Callable[[str, dict], None]] = None
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        if self.s.lwt_enabled:
            self.client.will_set(
                self._topic("events/record_status"),
                json.dumps({"recording": False, "reason": "ai_offline"}, ensure_ascii=False),
                qos=1,
                retain=True,
            )

        self._lwt_client: Optional[mqtt.Client] = None
        if self.s.lwt_enabled:
            self._lwt_client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION1,
                client_id=f"{self.s.client_id}-lwt",
            )
            self._lwt_client.will_set(
                self._topic("events/system_info"),
                json.dumps({"ts": self._now(), "type": "backend_offline"}, ensure_ascii=False),
                qos=1,
                retain=False,
            )

    # ---------- connection ----------
    def connect(self) -> None:
        # 重連間隔給個上限，萬一真的又發生互踢，不會變成每秒重連的忙迴圈
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)
        logger.info(
            f"[mqtt] 連線 {self.s.host}:{self.s.port} client_id={self.s.client_id} "
            f"topic={self._topic('...')}"
        )
        self.client.connect(self.s.host, self.s.port, keepalive=self.s.keepalive)
        self.client.loop_start()

        # 這條連線只為了持有 backend_offline 的遺囑。原本只建立、沒有連線，
        # 所以 AI 異常死掉時 App 永遠收不到 backend_offline —— 功能是死的。
        # id 是 `<主 id>-lwt`，跟主連線不衝突。
        if self._lwt_client is not None:
            self._lwt_client.reconnect_delay_set(min_delay=1, max_delay=30)
            self._lwt_client.connect(self.s.host, self.s.port, keepalive=self.s.keepalive)
            self._lwt_client.loop_start()

    def disconnect(self) -> None:
        # 順序很重要：要先 disconnect() 再 loop_stop()。
        # loop_start() 底下是 loop_forever()，DISCONNECT 封包得靠那個 thread 送出去；
        # 先 loop_stop() 會讓封包還沒送出 thread 就停了，broker 只能等 socket 關閉
        # （甚至等 keepalive 逾時）才知道我們走了 —— 重啟時新舊行程因此重疊，
        # 這也是原本會互踢的幫凶之一。paho 的 loop_forever 收到 disconnect() 後
        # 會自己結束，loop_stop() 只是補一刀確保 thread 收乾淨。
        try:
            self.client.disconnect()
        finally:
            self.client.loop_stop()

        if self._lwt_client is not None:
            try:
                self._lwt_client.disconnect()
            finally:
                self._lwt_client.loop_stop()

    # ---------- publishing helpers ----------
    def publish_json(self, topic_suffix: str, payload: Union[dict, list], retain: bool = False, qos: int = 1) -> None:
        topic = self._topic(topic_suffix)
        self.client.publish(topic, json.dumps(payload, ensure_ascii=False), qos=qos, retain=retain)

    def publish_system(self, payload: dict) -> None:
        """System-level events."""
        self.publish_json("events/system_info", payload, retain=False, qos=1)

    def publish_det_status(self, payload: dict) -> None:
        self.publish_json("events/detection_status", payload, retain=False, qos=1)
    
    def publish_record_status(
        self,
        recording: bool,
        start_time: Optional[float] = None,
        path: Optional[str] = None,
        reason: str = "",
    ) -> None:
        """
        發布目前的錄影狀態。務必用 retain=True，理由是：
        - 讓「之後才連上」的 subscriber (App 重開、重新訂閱) 也能馬上拿到
          目前是否在錄影，不用等下一次 start/stop 才更新。
        - 搭配上面的 will，AI 斷線時 broker 會自動把這個 topic 蓋成
          recording=false，App 不會卡在「上次看到的是 true」的狀態。

        只在狀態真的改變時呼叫 (收到 cmd/record start/stop，或 AI 自己
        因為某些原因中止錄影時)，不要週期性重複發送。
        """
        payload: dict = {"recording": recording, "reason": reason}
        if recording:
            if start_time is not None:
                payload["start_time"] = int(start_time)
            if path is not None:
                payload["path"] = path
        self.publish_json("events/record_status", payload, retain=True, qos=1)

    # ---------- command subscription ----------
    def on_command(self, cb: Callable[[str, dict], None]) -> None:
        """Register callback for commands: cb(cmd: str, payload: dict)."""
        self._on_cmd_cb = cb

    # ---------- callbacks ----------
    def _on_connect(self, client: mqtt.Client, userdata: Any, flags: dict, rc: int) -> None:
        """The callback for when the client receives a CONNACK response from the server."""
        if rc != 0:
            logger.error(f"[mqtt] 連線失敗 rc={rc} client_id={self.s.client_id}")
            return
        logger.info(f"[mqtt] 已連線 client_id={self.s.client_id}")
        self.client.subscribe(self._topic("cmd/#"), qos=1)
        self.publish_system({"ts": self._now(), "type": "backend_online"})

    def _on_disconnect(self, client: mqtt.Client, userdata: Any, rc: int) -> None:
        """The callback for when the client disconnects from the server.

        rc 是 paho 的錯誤碼，不是 MQTT v5 的 reason code：7 = MQTT_ERR_CONN_LOST，
        也就是「連線被切掉」。最常見的兩個原因是 broker 重啟，以及**有人用同一個
        client_id 連進來把我們踢掉**（後者請看 default_client_id 的說明）。
        loop_forever 會自己重連，所以這裡只記錄、不做事。
        """
        if rc != 0:
            reason = _PAHO_RC_HINT.get(rc, "")
            logger.warning(
                f"[mqtt] 非預期斷線 rc={rc}{reason} client_id={self.s.client_id}，"
                f"paho 會自動重連"
            )

    def _on_message(self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
        """The callback for when a PUBLISH message is received from the server."""

        cmd_base_topic = self._topic("cmd/")

        try:
            payload = json.loads(msg.payload.decode("utf-8")) if msg.payload else {}
        except json.JSONDecodeError:
            logger.warning(f"[mqtt] 無法解析來自主題 '{msg.topic}' 的 JSON 內容: {msg.payload}")
            return

        if msg.topic.startswith(cmd_base_topic):
            command_from_topic = msg.topic[len(cmd_base_topic):]
            cmd = command_from_topic.strip().lower()
            if self._on_cmd_cb and cmd:
                self._on_cmd_cb(cmd, payload)
        
    # ---------- utils ----------
    def _topic(self, suffix: str) -> str:
        return f"{self.s.base_topic}/{self.s.session}/{suffix}"

    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
