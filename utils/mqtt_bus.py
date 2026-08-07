from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Optional, Callable, List, Union
import paho.mqtt.client as mqtt

@dataclass
class MqttSettings:
    host: str = "localhost"
    port: int = 1883
    client_id: str = "client"
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

        # Last Will: backend offline (abnormal disconnect only)
        if self.s.lwt_enabled:
            self.client.will_set(
                self._topic("events/system_info"),
                json.dumps({"ts": self._now(), "type": "backend_offline"}),
                qos=1,
                retain=False,
            )

    # ---------- connection ----------
    def connect(self) -> None:
        self.client.connect(self.s.host, self.s.port, keepalive=self.s.keepalive)
        self.client.loop_start()

    def disconnect(self) -> None:
        try:
            self.client.loop_stop()
        finally:
            self.client.disconnect()

    # ---------- publishing helpers ----------
    def publish_json(self, topic_suffix: str, payload: Union[dict, list], retain: bool = False, qos: int = 1) -> None:
        topic = self._topic(topic_suffix)
        self.client.publish(topic, json.dumps(payload, ensure_ascii=False), qos=qos, retain=retain)

    def publish_system(self, payload: dict) -> None:
        """System-level events."""
        self.publish_json("events/system_info", payload, retain=False, qos=1)

    def publish_det_status(self, payload: dict) -> None:
        self.publish_json("events/detection_status", payload, retain=False, qos=1)

    # ---------- command subscription ----------
    def on_command(self, cb: Callable[[str, dict], None]) -> None:
        """Register callback for commands: cb(cmd: str, payload: dict)."""
        self._on_cmd_cb = cb

    # ---------- callbacks ----------
    def _on_connect(self, client: mqtt.Client, userdata: Any, flags: dict, rc: int) -> None:
        """The callback for when the client receives a CONNACK response from the server."""
        if rc != 0:
            print(f"Failed to connect: {rc}")
            return
        self.client.subscribe(self._topic("cmd/#"), qos=1)
        self.publish_system({"ts": self._now(), "type": "backend_online"})

    def _on_disconnect(self, client: mqtt.Client, userdata: Any, rc: int) -> None:
        """The callback for when the client disconnects from the server."""
        if rc != 0:
            print(f"Unexpected disconnection with reason: {rc}")

    def _on_message(self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
        """The callback for when a PUBLISH message is received from the server."""

        cmd_base_topic = self._topic("cmd/")

        try:
            payload = json.loads(msg.payload.decode("utf-8")) if msg.payload else {}
        except json.JSONDecodeError:
            print(f"無法解析來自主題 '{msg.topic}' 的 JSON 內容: {msg.payload}")
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
