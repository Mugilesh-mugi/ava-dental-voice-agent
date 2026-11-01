"""Text-to-speech ("the mouth"): Kokoro (the default) or Meta's MMS, chosen by the TTS_MODEL setting.

    TTS_MODEL=kokoro                 Kokoro-82M through ONNX Runtime: more natural, has a speed setting, says every digit.
    TTS_MODEL=facebook/mms-tts-eng   MMS through Hugging Face Transformers: faster on a CPU, but one fixed quick pace and it
                                     cannot say the digits 7, 8 or 9 (speech.for_voice spells numbers out for it).
"""
from __future__ import annotations

import logging
import threading
import urllib.request
from pathlib import Path

import numpy as np

from .stt import model_load_error

log = logging.getLogger("voice_agent.tts")

KOKORO_MODEL_FILE = "kokoro-v1.0.int8.onnx"     # the small (about 90 MB) version of the model
KOKORO_VOICES_FILE = "voices-v1.0.bin"
KOKORO_DOWNLOAD = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"


def is_kokoro(model_name: str) -> bool:
    return model_name.lower().startswith("kokoro")


def _download_if_missing(folder: Path, name: str) -> Path:
    path = folder / name
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        log.info("tts_downloading_kokoro_file name=%s", name)
        part = path.with_suffix(path.suffix + ".part")      # so a half-finished download is never mistaken for the file
        urllib.request.urlretrieve(KOKORO_DOWNLOAD + name, part)
        part.replace(path)
    return path


class TextToSpeech:
    def __init__(self, model_name: str, voice: str = "bf_emma", speed: float = 0.95, model_dir: Path | str = "data/models/kokoro"):
        self.model_name = model_name
        self._voice, self._speed = voice, speed
        self._lock = threading.Lock()
        self._kokoro = self._pipeline = None
        try:
            if is_kokoro(model_name):
                from kokoro_onnx import Kokoro

                folder = Path(model_dir)
                self._kokoro = Kokoro(str(_download_if_missing(folder, KOKORO_MODEL_FILE)),
                                      str(_download_if_missing(folder, KOKORO_VOICES_FILE)))
            else:
                from transformers import pipeline

                self._pipeline = pipeline("text-to-speech", model=model_name, device="cpu")
        except Exception as e:
            raise model_load_error("text-to-speech", model_name, e) from e

    def speak(self, text: str) -> tuple[int, np.ndarray]:
        """Return (sample_rate, audio as 16-bit whole numbers), the format Gradio plays directly."""
        with self._lock:
            if self._kokoro is not None:
                # "bf_" voices are British English, "af_"/"am_" American: the language code must match the voice.
                audio, rate = self._kokoro.create(text, voice=self._voice, speed=self._speed,
                                                  lang="en-gb" if self._voice.startswith("b") else "en-us")
            else:
                out = self._pipeline(text)
                audio, rate = out["audio"], out["sampling_rate"]
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        return int(rate), (np.clip(audio, -1, 1) * 32767).astype(np.int16)
