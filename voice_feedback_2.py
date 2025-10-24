import os
import time
from typing import Optional, Dict

_last_alert_event: Dict[str, object] = {"label": None, "ts": 0.0, "seq": 0}

class VoiceAlertManager:
    def __init__(self, temp_audio_dir: str = "voice_cache", cooldown_seconds: float = 0.25):
        self.last_state: Optional[str] = None  # "Move" or "Stop"
        self.last_time: float = 0.0            # last time an alert was emitted
        self.cooldown: float = cooldown_seconds

        self.audio_folder = temp_audio_dir
        os.makedirs(self.audio_folder, exist_ok=True)

    def _resolve_prerecorded(self, message: str) -> Optional[str]:
        candidates = [
            os.path.join(self.audio_folder, f"{message}.mp3"),
            os.path.join(self.audio_folder, f"{message.lower()}.mp3"),
            os.path.join(self.audio_folder, f"{message.title()}.mp3"),
            os.path.join(self.audio_folder, f"{message.upper()}.mp3"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def _emit_event(self, label: str):
        global _last_alert_event
        _last_alert_event = {
            "label": label,
            "ts": time.time(),
            "seq": int(_last_alert_event.get("seq", 0)) + 1,
        }
        print(f"[VoiceEvent] Emitted alert: {_last_alert_event}")

    def update_and_speak(self, is_safe: bool, timestamp: float):
        now = time.time()
        label = "Move" if is_safe else "Stop"
        print(f"[VoiceDebug] is_safe={is_safe}, label={label}, t={timestamp:.2f}s")

        if self.last_state != label or now - self.last_time > self.cooldown:
            path = self._resolve_prerecorded(label)
            if not path:
                print(f"[VoiceWarn] Missing prerecorded audio for '{label}' in {self.audio_folder}.")
            self._emit_event(label)
            self.last_time = now
            self.last_state = label


def get_last_alert_event() -> Dict[str, object]:
    """Return the latest alert event dict: {label:str|None, ts:float, seq:int}."""
    return dict(_last_alert_event)
