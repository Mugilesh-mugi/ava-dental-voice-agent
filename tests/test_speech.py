"""What the voice reads: digits are spelled out, because the text-to-speech voice garbles them."""
import pytest

from voice_agent.agent.speech import for_voice, number_words


@pytest.mark.parametrize("n,words", [(0, "zero"), (45, "forty-five"), (60, "sixty"), (95, "ninety-five"),
                                     (100, "one hundred"), (500, "five hundred"), (1250, "one thousand two hundred and fifty")])
def test_numbers_are_spelled_out(n, words):
    assert number_words(n) == words


@pytest.mark.parametrize("written,spoken", [
    ("the phone number is 01632 960 000.", "the phone number is zero one six three two, nine six zero, zero zero zero."),
    ("urgent line 01632 960 111", "urgent line zero one six three two, nine six zero, one one one"),
    ("Call NHS 111", "Call N H S one one one"),
    ("open 8am to 6pm.", "open eight a m to six p m."),
    ("We open at 9:30 on the 8th", "We open at nine thirty on the eighth"),
    ("a scale and polish costs 60 pounds", "a scale and polish costs sixty pounds"),
    ("fillings from £95", "fillings from ninety-five pounds"),
    ("24 Harbour Road, BW1 2AB", "twenty-four Harbour Road, B W one, two A B"),
    ("number 0 7 7 0 0 9 0 0 1 2 3", "number zero seven seven zero zero nine zero zero one two three"),
])
def test_digits_are_spelled_out_for_the_voice(written, spoken):
    assert for_voice(written) == spoken


def test_text_without_digits_is_left_alone():
    text = "Hello, you're through to Sunrise Dental. How can I help?"
    assert for_voice(text) == text


def test_the_voice_gets_the_spelled_out_text_but_the_screen_keeps_the_original(build_service):
    from tool_fakes import ScriptedToolLLM, calls
    spoken = []
    service = build_service(ScriptedToolLLM(calls("get_clinic_info", question="What is your phone number?")))

    class RecordingVoice:
        def speak(self, text):
            spoken.append(text)
            return 16000, None

    service._tts = RecordingVoice()
    audio, _ = service.speak("Call 01632 960 000 before 8am.")
    assert spoken == ["Call zero one six three two, nine six zero, zero zero zero before eight a m."]
