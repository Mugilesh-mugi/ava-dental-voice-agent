"""Measure speech-to-text accuracy as Word Error Rate (WER).

WER = (words wrongly changed + words missed + words added) / words in the correct sentence.
0% is perfect. 10% means about one word in ten is wrong.

How to use:
1. Record yourself saying some sentences. Save each as a WAV file in evals/recordings/,
   for example  q01.wav
2. Next to each one, put a text file with the same name holding the correct sentence:
   q01.txt  ->  I can't breathe
3. Run:  python evals/run_wer.py

Only plain WAV files are read (no ffmpeg needed). If there are no recordings, the script says so and stops.
"""
from __future__ import annotations

import os
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

RECORDINGS = ROOT / "evals" / "recordings"


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as w:
        sample_width, channels, rate = w.getsampwidth(), w.getnchannels(), w.getframerate()
        frames = w.readframes(w.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sample_width]
    audio = np.frombuffer(frames, dtype=dtype)
    if dtype is np.uint8:                         # 8-bit WAV is stored 0..255; centre it on 0
        audio = (audio.astype(np.int16) - 128) * 256
    return rate, audio.reshape(-1, channels) if channels > 1 else audio


def main() -> int:
    pairs = [(wav, wav.with_suffix(".txt")) for wav in sorted(RECORDINGS.glob("*.wav"))] if RECORDINGS.exists() else []
    pairs = [(w, t) for w, t in pairs if t.exists()]
    if not pairs:
        print(f"No recordings found in {RECORDINGS} (each .wav needs a matching .txt). Skipping WER check.")
        return 0

    import jiwer

    from voice_agent.agent.rules import normalise
    from voice_agent.config import get_settings
    from voice_agent.voice.stt import SpeechToText

    settings = get_settings()
    print(f"Loading speech-to-text model {settings.stt_model} ...")
    stt = SpeechToText(settings.stt_model)

    references, hypotheses = [], []
    for wav_path, txt_path in pairs:
        reference = normalise(txt_path.read_text(encoding="utf-8"))
        hypothesis = normalise(stt.transcribe(*read_wav(wav_path)))
        references.append(reference)
        hypotheses.append(hypothesis)
        sentence_wer = jiwer.wer(reference, hypothesis) if reference else 0.0
        flag = "" if sentence_wer == 0 else "   <-- check"
        print(f"{wav_path.name:16} WER {sentence_wer:6.1%}   said: \"{reference}\"   heard: \"{hypothesis}\"{flag}")

    print(f"\nOverall WER on {len(pairs)} recording(s): {jiwer.wer(references, hypotheses):.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
