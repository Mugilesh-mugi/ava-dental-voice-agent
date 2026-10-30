"""Text-to-speech ("the mouth"): Meta's MMS English voice through Hugging Face Transformers."""
from __future__ import annotations

import threading

import numpy as np

from .stt import model_load_error


class TextToSpeech:
    def __init__(self, model_name: str):
        self.model_name = model_name
        try:
            from transformers import pipeline

            self._tts = pipeline("text-to-speech", model=model_name, device="cpu")
        except Exception as e:
            raise model_load_error("text-to-speech", model_name, e) from e
        self._lock = threading.Lock()

    def speak(self, text: str) -> tuple[int, np.ndarray]:
        """Return (sample_rate, audio as 16-bit whole numbers), the format Gradio plays directly."""
        with self._lock:
            out = self._tts(text)
        audio = np.asarray(out["audio"], dtype=np.float32).squeeze()
        return int(out["sampling_rate"]), (np.clip(audio, -1, 1) * 32767).astype(np.int16)
