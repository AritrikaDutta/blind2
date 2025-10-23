import os
import time
from gtts import gTTS
from pydub import AudioSegment

class VoiceAlertManager:
    def __init__(self, temp_audio_dir="voice_cache"):
        self.last_state = None   # "Move" or "Stop"
        self.last_time = 0       # last time a message was spoken
        self.cooldown = 8        # repeat after 8 seconds
        self.audio_folder = temp_audio_dir
        os.makedirs(self.audio_folder, exist_ok=True)

    def generate_audio(self, message):
        """Generate and cache audio for the given message."""
        filepath = os.path.join(self.audio_folder, f"{message}.mp3")
        if not os.path.exists(filepath):
            try:
                tts = gTTS(text=message, lang='en')
                tts.save(filepath)
                print(f"[VoiceInfo] Generated new audio: {filepath}")
            except Exception as e:
                print(f"[VoiceError] Failed to generate audio for '{message}': {e}")
        return filepath

    def get_audio_path(self, message):
        """Return the path to the audio file for browser playback."""
        return self.generate_audio(message)

    def update_and_speak(self, is_safe, timestamp):
        """Decide whether to play 'Move' or 'Stop' based on state changes."""
        now = time.time()
        label = "Move" if is_safe else "Stop"
        print(f"[VoiceDebug] is_safe={is_safe}, label={label}, now={timestamp:.2f}s")
        if self.last_state != label or now - self.last_time > self.cooldown:
            self.generate_audio(label)
            self.last_time = now
            self.last_state = label
