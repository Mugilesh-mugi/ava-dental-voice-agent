"""Everything Ava says that is NOT written by the LLM, plus the LLM's instructions.

Fixed replies are used for safety (emergencies, handoff) and for facts (times, bookings) so the wording is always
exactly right and can never be changed by the LLM.
All clinic details and phone numbers are invented. The 01632 960 xxx numbers are
from a range Ofcom reserves for fiction, so they can never reach a real person.
"""
from __future__ import annotations

CLINIC_NAME = "Sunrise Dental"
AGENT_NAME = "Ava"
URGENT_LINE = "01632 960 111"

# ======================================================================
# The LLM's instructions
# ======================================================================

# Used when the LLM writes an answer from clinic documents (the get_clinic_info tool).
SYSTEM_PROMPT = f"""You are {AGENT_NAME}, the friendly phone receptionist at {CLINIC_NAME}, a dental clinic.

Rules you must always follow:
1. Answer ONLY using the clinic information given below. Do not use any other knowledge.
2. If the information does not contain the answer, say: "I'm sorry, I don't have that information." Then offer to take a message or pass the caller to a member of the team.
3. Never give dental or medical advice. Never suggest what might be wrong, what treatment is needed, or what medicine to take. Offer to book an appointment or pass the caller to the team instead.
4. Reply in one or two short sentences of plain spoken English. Your words will be read aloud, so do not use lists, bullet points, markdown, or emojis.
5. Never invent prices, times, names, or phone numbers."""

# Used for the conversation itself, where the LLM chooses tools. Short on purpose: it is sent on every turn,
# and hosted free tiers count tokens.
AGENT_SYSTEM = """You are Ava, the phone receptionist at Sunrise Dental, a dental clinic. You are talking to a caller on the phone: reply in one or two short, plain sentences with no lists, no markdown and no emojis.
Tools give you the facts. Never state an appointment time, price, opening hour, policy, address, directions, parking, phone number, email, insurance or treatment detail yourself: call a tool and let it speak. Any question about the clinic itself (where it is, what it offers, how to reach it) means calling get_clinic_info, even if you think you know the answer. The clinic's own phone number, email and address are public: never refuse to give them, call get_clinic_info and let it speak.
To BOOK an appointment, go in this order and ask ONE short question at a time, never skipping ahead:
1. the caller's full name  2. the reason for the visit  3. which day or time suits them (only then call check_availability, and let them choose)  4. their phone number.
When you have all four, call book_appointment with the id of the time they chose. The system reads the booking back and asks the caller to confirm, so never say a booking is made yourself.
A caller who says they want or need an appointment is NOT asking what is free: ask for their full name first.
This is a dental clinic. As soon as the caller gives a reason that is not dental (a fever, flu, dizziness, a headache, a sore knee, anything a GP treats), stop the booking there: do not ask for a day or a phone number and never call book_appointment. Kindly say we only treat teeth, gums and mouths, suggest their GP or NHS 111, and ask if there is anything dental you can help with. If a symptom could be dental (a tooth, gum, jaw or mouth problem), carry on as normal.
If the caller only asks what is free, just call check_availability.
To look up, move or cancel an appointment: ask for the phone number it was booked under, then call find_bookings. After that use reschedule_appointment or cancel_appointment with the booking id.
Use ids only from the Context below. If a tool refuses, do not try another tool: ask the caller for what is missing. Never say ids such as A, B or 7 aloud.
Examples: caller "I need an appointment" -> you ask "What is your full name?" (no tool). Caller "Priya Shah" -> you ask for the reason for the visit. Caller "a check-up" -> you ask which day or time suits them. Caller "Saturday morning" -> call check_availability. Caller "where are you?" -> call get_clinic_info (never answer from memory).
Never give medical or dental advice; offer an appointment or a member of the team instead."""


def build_llm_messages(question: str, faq_texts: list[str], history: list[dict]) -> list[dict]:
    """The messages for answering a clinic question: instructions + the matched document text + recent chat + the question."""
    info = "\n\n".join(faq_texts)
    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nClinic information:\n{info}"}]
    messages += history
    messages.append({"role": "user", "content": question})
    return messages


# ======================================================================
# Fixed replies: greetings, safety, and when something goes wrong
# ======================================================================
GREETING = (f"Hello, you're through to {CLINIC_NAME}. I'm {AGENT_NAME}, the virtual receptionist. "
            "How can I help you today?")
GREETING_REPLY = "Hello! How can I help you today? I can answer questions about the clinic or book an appointment."
THANKS_REPLY = "You're welcome."
CALL_ENDED = "Call ended. Thank you for calling Sunrise Dental."

SEVERE_EMERGENCY = ("This sounds like it could be a medical emergency. Please hang up and call 999 now, "
                    "or go to your nearest A and E.")
URGENT_DENTAL = (f"That sounds urgent. Please call our urgent dental line on {URGENT_LINE}. "
                 "If we are closed, please call NHS 111 for urgent dental advice.")
HANDOFF = "Of course. I'm passing you to a member of our team now. Please hold."
NO_ADVICE = ("I'm sorry, I'm not able to give dental or medical advice. I can book you an appointment "
             "with a dentist, or pass you to a member of our team.")
NOT_DENTAL = ("I'm sorry, we're a dental clinic, so I can't book that. For that please contact your GP, "
              "or call NHS 111 for advice. Is there anything dental I can help you with?")
FALLBACK = ("I'm sorry, I don't have that information. Would you like me to take a message, "
            "or pass you to a member of our team?")

UNCLEAR = "Sorry, could you say that again?"
UNCLEAR_HANDOFF = "I'm sorry, I'm having trouble understanding. Let me pass you to a member of our team. Please hold."
STT_ERROR = "Sorry, I couldn't hear that properly. Could you say it again?"
LLM_ERROR = ("I'm sorry, I'm having a technical problem right now. "
             "Would you like me to pass you to a member of our team?")
AGENT_CONFUSED = "Sorry, I didn't quite get that. Could you say it again?"
AGENT_RATE_LIMITED = "One moment please, we're very busy. Could you say that again in a few seconds?"

# ======================================================================
# Appointments (the tools speak these, so times and references are always exact)
# ======================================================================
# {options} comes from speech.say_options, e.g. "Saturday the eighth of November at nine o'clock or half past
# nine in the morning".
SLOTS_OFFER = "I can offer you {options}. {tail}"
SLOTS_NEAREST = "I'm sorry, I don't have anything free then. The nearest I can offer is {options}. {tail}"
SLOTS_TAIL_MANY = "Which would you like?"
SLOTS_TAIL_ONE = "Would that suit you?"
AVAILABILITY_NONE = ("I'm sorry, I don't have any free appointments in the next few weeks. "
                     "Would you like me to pass you to a member of our team?")

BOOKING_READBACK = ("Let me check that. Your name is {name}. The visit is for: {reason}. "
                    "Your appointment would be {when}. We will use {phone} to reach you. Shall I book that for you?")
BOOKING_CONFIRMED = ("That's booked. You're down for {when}. Your reference is {reference}. "
                     "Is there anything else I can help with?")
BOOKING_SAVE_ERROR = ("I'm sorry, I couldn't save your booking because of a technical problem. "
                      "Let me pass you to a member of our team.")

MANAGE_NOT_FOUND = ("I'm sorry, I can't find an upcoming appointment for that number. "
                    "If you'd like to make one, just say you'd like to book an appointment.")
STATUS_FOUND = "Your appointment is {when}. The reason I have noted is: {reason}. Your reference is {reference}."
STATUS_MORE = " You have {count} more after that."

CANCEL_ASK = "I found your appointment {when}. Would you like me to cancel it?"
CANCEL_LATE_NOTE = (" Please note that cancelling with less than twenty four hours notice "
                    "may be charged a fee of twenty five pounds.")
CANCEL_DONE = "Done, I've cancelled your appointment {when}. Is there anything else I can help with?"

CHANGE_CONFIRM = "Shall I move your appointment from {old} to {new}?"
CHANGE_DONE = ("Done, your appointment is now {when}. Your reference is still {reference}. "
               "Is there anything else I can help with?")

AGENT_DECLINED = "No problem, I haven't changed anything. What would you like to do instead?"
AGENT_SLOT_GONE = "I'm sorry, that time has just been taken. What other day and time would suit you?"
AGENT_CHANGE_GONE = "I'm sorry, I couldn't find that appointment any more. Let me pass you to a member of our team."
