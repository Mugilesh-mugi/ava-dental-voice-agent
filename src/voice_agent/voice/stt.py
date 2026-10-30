"""Speech-to-text ("the ears"): Whisper, run by one of two engines (STT_ENGINE):

- "faster_whisper": the faster-whisper library (CTranslate2). Several times faster on a CPU and
  uses less memory, e.g. STT_MODEL=distil-small.en.
- "transformers": Hugging Face Transformers, e.g. STT_MODEL=openai/whisper-small.

Gradio gives us the recording as (sample_rate, numbers). We turn it into what
Whisper expects (one channel, values between -1 and 1, 16,000 samples per second)
ourselves, so we do not need ffmpeg to read audio files.
"""
from __future__ import annotations

import logging
import math
import threading

import numpy as np

log = logging.getLogger(__name__)

WHISPER_RATE = 16_000


class ModelLoadError(RuntimeError):
    """A model could not be loaded. The message says which one and what to check."""


def model_load_error(kind: str, model_name: str, error: Exception) -> ModelLoadError:
    return ModelLoadError(
        f"Could not load the {kind} model '{model_name}'.\n"
        f"  Check that it exists: https://huggingface.co/{model_name}\n"
        f"  Check your internet connection (the first run downloads it), and that you have enough free memory.\n"
        f"  Original error: {type(error).__name__}: {error}"
    )


def to_whisper_audio(sample_rate: int, audio: np.ndarray) -> np.ndarray:
    """Convert any recording to mono float32 at 16 kHz with values between -1 and 1."""
    audio = np.asarray(audio)
    if np.issubdtype(audio.dtype, np.integer):        # whole numbers -> -1..1 (do this BEFORE averaging)
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    audio = audio.astype(np.float32)
    if audio.ndim == 2:                               # stereo -> mono (average the channels)
        audio = audio.mean(axis=1 if audio.shape[1] <= 8 else 0)
    if sample_rate != WHISPER_RATE and len(audio):    # change the speed of sampling (resample)
        # resample_poly filters out frequencies 16 kHz audio cannot hold before shrinking, so
        # the sound is not distorted ("aliasing"), which plain interpolation would do.
        from scipy.signal import resample_poly

        g = math.gcd(WHISPER_RATE, int(sample_rate))
        audio = resample_poly(audio, WHISPER_RATE // g, int(sample_rate) // g).astype(np.float32)
    return audio


class SpeechToText:
    def __init__(self, model_name: str, prompt: str | None = None):
        """prompt: words the caller is likely to say (clinic name, services). Whisper hears them more reliably."""
        self.model_name = model_name
        try:
            from transformers import pipeline

            # ignore_warning: chunk_length_s (below) prints an "experimental" warning on every call.
            self._asr = pipeline("automatic-speech-recognition", model=model_name, device="cpu",
                                 ignore_warning=True)
        except Exception as e:
            raise model_load_error("speech-to-text", model_name, e) from e
        # English-only models (names ending ".en") already know the language and refuse these settings.
        self._generate_kwargs: dict = {} if model_name.endswith(".en") else {"language": "english",
                                                                             "task": "transcribe"}
        if prompt:
            self._generate_kwargs["prompt_ids"] = self._asr.tokenizer.get_prompt_ids(prompt, return_tensors="pt")
        self._lock = threading.Lock()   # one transcription at a time keeps memory use predictable

    def transcribe(self, sample_rate: int, audio: np.ndarray) -> str:
        samples = to_whisper_audio(sample_rate, audio)
        if len(samples) < WHISPER_RATE * 0.2:          # under 0.2 seconds: nothing to hear
            return ""
        with self._lock:
            result = self._asr(
                {"raw": samples, "sampling_rate": WHISPER_RATE},
                generate_kwargs=self._generate_kwargs,
                chunk_length_s=30,
            )
        return result["text"].strip()


class FasterWhisperSpeechToText:
    """Same job as SpeechToText, using faster-whisper. int8 numbers make it fast on a CPU."""

    def __init__(self, model_name: str, prompt: str | None = None):
        self.model_name = model_name
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(model_name, device="cpu", compute_type="int8")
        except Exception as e:
            raise model_load_error("speech-to-text", model_name, e) from e
        self._prompt = prompt
        self._lock = threading.Lock()

    def transcribe(self, sample_rate: int, audio: np.ndarray) -> str:
        samples = to_whisper_audio(sample_rate, audio)
        if len(samples) < WHISPER_RATE * 0.2:          # under 0.2 seconds: nothing to hear
            return ""
        with self._lock:
            segments, _ = self._model.transcribe(samples, language="en", initial_prompt=self._prompt,
                                                 beam_size=5, condition_on_previous_text=False)
            return " ".join(s.text.strip() for s in segments).strip()


def build_speech_to_text(engine: str, model_name: str, prompt: str | None = None):
    if engine == "faster_whisper":
        return FasterWhisperSpeechToText(model_name, prompt)
    return SpeechToText(model_name, prompt)
