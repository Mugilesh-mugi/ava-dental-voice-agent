"""Text-to-speech ("the mouth"): Piper (the default), Kokoro or Meta's MMS, chosen by the TTS_MODEL setting.

    TTS_MODEL=piper                  Piper: by far the fastest on a CPU (a reply in well under a second), natural enough.
    TTS_MODEL=kokoro                 Kokoro-82M through ONNX Runtime: the most natural voice here, but about as slow as real time.
    TTS_MODEL=facebook/mms-tts-eng   MMS through Hugging Face Transformers: one fixed quick pace and it cannot say the
                                     digits 7, 8 or 9 (speech.for_voice spells numbers out for it).
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
KOKORO_DEFAULT_VOICE = "bf_emma"                # British woman

PIPER_DOWNLOAD = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
PIPER_DEFAULT_VOICE = "en_GB-jenny_dioco-medium"    # British woman
PIPER_PAUSE_SECONDS = 0.2                       # silence between sentences


def is_kokoro(model_name: str) -> bool:
    return model_name.lower().startswith("kokoro")


def is_piper(model_name: str) -> bool:
    return model_name.lower().startswith("piper")


def _download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("tts_downloading name=%s", path.name)
        part = path.with_suffix(path.suffix + ".part")      # so a half-finished download is never mistaken for the file
        urllib.request.urlretrieve(url, part)
        part.replace(path)
    return path


def _piper_files(folder: Path, voice: str) -> tuple[Path, Path]:
    """A Piper voice is two files. "en_GB-jenny_dioco-medium" lives at en/en_GB/jenny_dioco/medium/ in the voice library."""
    region, name, quality = voice.split("-")
    base = f"{PIPER_DOWNLOAD}{region.split('_')[0]}/{region}/{name}/{quality}/{voice}"
    return (_download(base + ".onnx", folder / f"{voice}.onnx"),
            _download(base + ".onnx.json", folder / f"{voice}.onnx.json"))


class TextToSpeech:
    def __init__(self, model_name: str, voice: str = "", speed: float = 1.0, model_dir: Path | str = "data/models"):
        self.model_name = model_name
        self._speed = speed
        self._lock = threading.Lock()
        self._kokoro = self._piper = self._pipeline = None
        folder = Path(model_dir)
        try:
            if is_piper(model_name):
                from piper import PiperVoice, SynthesisConfig

                self._voice = voice or PIPER_DEFAULT_VOICE
                model_file, _ = _piper_files(folder / "piper", self._voice)
                self._piper = PiperVoice.load(model_file)
                self._piper_config = SynthesisConfig(length_scale=1.0 / speed)     # length_scale above 1 is slower
            elif is_kokoro(model_name):
                from kokoro_onnx import Kokoro

                self._voice = voice or KOKORO_DEFAULT_VOICE
                self._kokoro = Kokoro(str(_download(KOKORO_DOWNLOAD + KOKORO_MODEL_FILE, folder / "kokoro" / KOKORO_MODEL_FILE)),
                                      str(_download(KOKORO_DOWNLOAD + KOKORO_VOICES_FILE, folder / "kokoro" / KOKORO_VOICES_FILE)))
            else:
                from transformers import pipeline

                self._pipeline = pipeline("text-to-speech", model=model_name, device="cpu")
        except Exception as e:
            raise model_load_error("text-to-speech", model_name, e) from e

    def speak(self, text: str) -> tuple[int, np.ndarray]:
        """Return (sample_rate, audio as 16-bit whole numbers), the format Gradio plays directly."""
        with self._lock:
            if self._piper is not None:
                chunks = list(self._piper.synthesize(text, self._piper_config))      # one chunk per sentence
                rate = chunks[0].sample_rate
                pause = np.zeros(int(rate * PIPER_PAUSE_SECONDS), dtype=np.int16)
                parts = [part for c in chunks for part in (c.audio_int16_array, pause)][:-1]
                return int(rate), np.concatenate(parts).astype(np.int16)
            if self._kokoro is not None:
                # "bf_" voices are British English, "af_"/"am_" American: the language code must match the voice.
                audio, rate = self._kokoro.create(text, voice=self._voice, speed=self._speed,
                                                  lang="en-gb" if self._voice.startswith("b") else "en-us")
            else:
                out = self._pipeline(text)
                audio, rate = out["audio"], out["sampling_rate"]
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        return int(rate), (np.clip(audio, -1, 1) * 32767).astype(np.int16)
