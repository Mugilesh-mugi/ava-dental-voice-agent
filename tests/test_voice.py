"""The parts of the speech code that don't need a model."""
import numpy as np

from voice_agent.voice.stt import WHISPER_RATE, to_whisper_audio


def test_stereo_int16_at_48k_becomes_mono_float_at_16k():
    stereo = (np.ones((48_000, 2)) * 16_384).astype(np.int16)
    out = to_whisper_audio(48_000, stereo)
    assert out.dtype == np.float32 and out.ndim == 1
    assert len(out) == WHISPER_RATE
    # The resampling filter ripples briefly where the signal jumps on and off; check away from the edges.
    assert np.allclose(out[200:-200], 0.5, atol=0.01)


def test_resampling_keeps_a_voice_tone_clean():
    t = np.arange(44_100) / 44_100
    tone = (np.sin(2 * np.pi * 440 * t) * 16_000).astype(np.int16)      # 1 second of a 440 Hz tone
    out = to_whisper_audio(44_100, tone)
    assert len(out) == WHISPER_RATE
    spectrum = np.abs(np.fft.rfft(out))
    assert abs(np.argmax(spectrum) - 440) <= 1                           # still a 440 Hz tone


def test_unknown_engine_falls_back_to_transformers(monkeypatch):
    import voice_agent.voice.stt as stt
    monkeypatch.setattr(stt, "SpeechToText", lambda name, prompt=None: ("transformers", name, prompt))
    monkeypatch.setattr(stt, "FasterWhisperSpeechToText", lambda name, prompt=None: ("faster_whisper", name, prompt))
    assert stt.build_speech_to_text("faster_whisper", "distil-small.en", "hi")[0] == "faster_whisper"
    assert stt.build_speech_to_text("transformers", "openai/whisper-small")[0] == "transformers"
