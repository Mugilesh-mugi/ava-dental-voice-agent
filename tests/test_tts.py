"""The voice: Piper is the default, a voice is picked by name, and (if their files are present) the voices really produce audio."""
import numpy as np
import pytest

from voice_agent.config import PROJECT_ROOT, Settings
from voice_agent.voice.tts import KOKORO_MODEL_FILE, PIPER_DEFAULT_VOICE, TextToSpeech, _piper_files, is_kokoro, is_piper

MODELS = PROJECT_ROOT / "data" / "models"


def test_piper_is_the_default_voice():
    assert is_piper(Settings(_env_file=None).tts_model)


@pytest.mark.parametrize("name,piper,kokoro", [("piper", True, False), ("kokoro", False, True), ("Kokoro-82M", False, True),
                                               ("facebook/mms-tts-eng", False, False)])
def test_the_voice_is_chosen_by_name(name, piper, kokoro):
    assert is_piper(name) is piper and is_kokoro(name) is kokoro


def test_a_piper_voice_is_found_in_the_voice_library_by_its_name(tmp_path, monkeypatch):
    urls = []
    monkeypatch.setattr("voice_agent.voice.tts._download", lambda url, path: urls.append(url) or path)
    _piper_files(tmp_path, "en_GB-jenny_dioco-medium")
    assert urls == ["https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx",
                    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx.json"]


def test_piper_makes_audio():
    if not (MODELS / "piper" / f"{PIPER_DEFAULT_VOICE}.onnx").exists():
        pytest.skip("Piper's voice is not downloaded yet (it downloads on the first start)")
    rate, audio = TextToSpeech("piper", model_dir=MODELS).speak("Hello, you're through to Sunrise Dental. How can I help?")
    assert rate > 0 and audio.dtype == np.int16 and len(audio) > rate // 2


def test_kokoro_makes_audio():
    if not (MODELS / "kokoro" / KOKORO_MODEL_FILE).exists():
        pytest.skip("Kokoro's files are not downloaded yet (they download when TTS_MODEL=kokoro is first used)")
    rate, audio = TextToSpeech("kokoro", model_dir=MODELS).speak("Hello, you're through to Sunrise Dental.")
    assert rate > 0 and audio.dtype == np.int16 and len(audio) > rate // 2
