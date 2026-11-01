"""The voice: Kokoro is picked by name, and (if its files are present) really produces audio."""
import numpy as np
import pytest

from voice_agent.config import PROJECT_ROOT, Settings
from voice_agent.voice.tts import KOKORO_MODEL_FILE, TextToSpeech, is_kokoro


def test_kokoro_is_the_default_voice():
    assert is_kokoro(Settings(_env_file=None).tts_model)


@pytest.mark.parametrize("name,expected", [("kokoro", True), ("Kokoro-82M", True), ("facebook/mms-tts-eng", False)])
def test_the_voice_is_chosen_by_name(name, expected):
    assert is_kokoro(name) is expected


def test_kokoro_makes_audio():
    folder = PROJECT_ROOT / "data" / "models" / "kokoro"
    if not (folder / KOKORO_MODEL_FILE).exists():
        pytest.skip("Kokoro's files are not downloaded yet (they download on the first start)")
    rate, audio = TextToSpeech("kokoro", model_dir=folder).speak("Hello, you're through to Sunrise Dental.")
    assert rate > 0 and audio.dtype == np.int16 and len(audio) > rate // 2
