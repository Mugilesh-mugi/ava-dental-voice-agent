# Ava: interview guide

Everything you need to present and defend this project. Read sections 1–4 for the story, section 5 for the demo, sections 6–9 for the "how does it work" questions, and section 10 for likely questions with answers.

---

## 1. The 30-second pitch

> "Ava is an AI voice receptionist for a dental clinic. A caller speaks; Ava understands, answers clinic questions from the clinic's own documents, and books, moves or cancels real appointments in a database. The LLM chooses *what to do* by calling tools, but plain code decides *whether it's allowed* and *what is said* for anything that matters: times, prices, bookings and emergencies. So it's useful like an LLM but safe like normal software."

Say this first. Everything else is detail on top of it.

## 2. The problem it solves

A dental front desk spends a lot of time on repetitive calls: opening hours, prices, "can I book Saturday?", "I need to move my appointment". Calls are missed when staff are busy or the clinic is closed, and a missed call is a lost patient.

Why this is hard to automate safely:

| Risk | What could go wrong | How Ava handles it |
|---|---|---|
| LLM makes things up | Wrong price, invented opening time | Facts come from documents (RAG) or from the database. If there's no source, Ava says "I don't have that information" |
| LLM books wrongly | Double booking, wrong person, wrong time | The LLM can only *stage* a change. The **code** commits it after the caller says yes. The database refuses two bookings at the same time |
| Medical emergency | Someone says "I can't breathe" and gets a chatbot reply | Emergency detection runs **before** the LLM, with fixed replies ("call 999") |
| Medical advice | "What painkiller should I take?" | Fixed refusal; Ava never gives advice |
| Speech recognition errors | Name or time misheard | Ava reads the booking back and asks for confirmation before saving |

The design principle: **the LLM is allowed to be smart, but not allowed to be in charge of anything irreversible.**

> Be upfront: the clinic (Sunrise Dental), its prices and phone numbers are invented. No real patient data is used.

## 3. Tech stack, and why each piece

| Layer | Technology | Why this, in one line |
|---|---|---|
| Language | Python 3.13 | Where all the AI libraries live |
| Web server / API | FastAPI + Uvicorn | Fast, simple, gives `/health` and `/metrics` endpoints |
| Demo UI | Gradio | A microphone and chat page in a few lines; good for demos |
| Agent flow | **LangGraph** | The turn is a graph of small steps (route → node). Easy to read, test and extend |
| LLM client | **LangChain** (`ChatOpenAI`) | One client that works with Groq, Gemini, Grok or local Ollama, because they share the OpenAI API format. Gives tool-calling for free |
| LLM | **qwen/qwen3.8-27b** on **Groq** | Good at choosing tools, fast, free tier for a demo |
| Knowledge search (RAG) | LangChain text splitter + HuggingFace embeddings + in-memory vector store | Small FAQ → no need for a vector database |
| Embeddings | `all-MiniLM-L6-v2` | Small (fast on CPU), good quality for short English text |
| Speech-to-text | `faster-whisper` `distil-small.en` (Whisper) | Runs locally on CPU, no audio leaves the machine, several times faster than plain Whisper |
| Text-to-speech | `facebook/mms-tts-eng` | Runs locally, free |
| Database | PostgreSQL + SQLAlchemy | Real bookings need a real database with constraints and concurrency safety |
| Monitoring | Prometheus client, JSON logs | Production habits: metrics and traceable logs per call |
| Testing | pytest (398 tests) | Fake LLM, so tests are free, fast and repeatable |
| Packaging | Dockerfile | Written but not yet built or tested; say so if asked |

### "Why LangChain/LangGraph, and what do they NOT do?"

- **LangGraph** runs the turn as a graph (classify → one node). **LangChain** talks to the LLM and handles documents, splitting, embeddings and the vector store.
- **They don't do the hard part.** The safety rules, tool checks, calendar, double-booking protection and date understanding are our own code, because they must be exactly right and testable.
- An earlier version was hand-written without frameworks. Moving to LangChain/LangGraph and deleting an old "step-by-step" mode cut the source from about 4,700 to about 3,700 lines. LangChain alone saved only about 150 lines; the rest came from removing the old mode. Be honest about that.

## 4. Models used

| Job | Model | Where it runs | Notes |
|---|---|---|---|
| Decide what to do, phrase answers | `qwen/qwen3.8-27b` (reasoning off) | Groq cloud (hosted) | Chosen after a 40-case tool-choice benchmark. `gpt-oss` models scored similarly; a local 3B model scored lower and made unsafe calls (like inventing a phone number) |
| Speech-to-text | `distil-small.en` via faster-whisper | Local CPU | English only; names are the weak spot |
| Text-to-speech | `facebook/mms-tts-eng` | Local CPU | Robotic but free and offline |
| Embeddings | `all-MiniLM-L6-v2` | Local CPU | 384-dimension vectors for the FAQ |

**Why a hosted LLM?** Tool-calling needs a capable model. Small local models (3B–4B) were tested and were not reliable. The cost: conversation text goes to the provider, and the free tier can be used by the provider for training, so only invented data is used. For real patients you'd use a paid plan with a data-processing agreement, or host your own model.

## 5. Demo script (about 5 minutes)

Start the app (`python app.py`, wait for `Ready.`, open http://127.0.0.1:7860). Use the text box if the microphone misbehaves. The free Groq tier allows about 5 turns a minute, so don't rush.

1. **Knowledge question:** "How much is a check-up?" Shows RAG: the answer comes from `data/raw/faq.md`.
2. **Out-of-scope:** "What's the weather tomorrow?" Shows Ava refusing to guess.
3. **Safety:** "My face has swollen up." Shows the urgent route, with no LLM involved. Mention "I can't breathe" gives 999.
4. **Booking:** "I'd like to book a check-up Saturday morning." Shows availability from the database, then picking "the second one", giving name and phone, the read-back, and "yes".
5. **Database:** show the booking row, and the `tool_calls` table (every tool the LLM tried, accepted or refused, with personal details masked). The page itself has no dashboard tab.
6. **Reschedule or cancel:** "I need to move my appointment" using the same phone number.
7. Optional: open `/metrics` and `/health`.

Tell them what to watch: *"Notice the LLM never writes the times or the reference number; the code does."*

## 6. Architecture

```
 Browser (Gradio)  ── mic / text ──►  FastAPI app  (app.py)
                                            │
                                  CallService  (one turn)
                                            │
   ┌───────────┬────────────────────────────┼───────────────────────────┐
   ▼           ▼                            ▼                           ▼
  EARS        BRAIN (LangGraph)            MOUTH                      NOTEBOOK
 Whisper     classify → one node          MMS TTS             PostgreSQL: calls, messages,
 (STT)          │                                              bookings, tool_calls,
                ├─ fixed replies (no LLM): 999 / urgent /      unanswered questions
                │   handoff / advice / thanks / hello / unclear  + JSON logs + /metrics
                ├─ agent_confirm: code commits on "yes"
                └─ tool_agent: LLM + 6 tools
                        │
        ┌───────────────┼────────────────────────────────────┐
        ▼               ▼                                    ▼
   check_availability  get_clinic_info (RAG)    find / book / reschedule / cancel
        │               │                                    │
   Scheduling +     Retriever → vector store       BookingService → PostgreSQL
   BookingService   (threshold 0.35)               (unique index prevents double booking)
```

Four ideas to remember:

1. **Layered safety.** Emergency/advice/handoff rules run first (no LLM). Only normal requests reach the LLM.
2. **Stage, then commit.** Book/move/cancel tools only *stage* the change. The code commits after a spoken yes.
3. **Code speaks the facts.** Tools return the exact sentence to say. The LLM can't garble a time, price or reference.
4. **The database is the last line of defence.** Even if everything above failed, a unique index stops two confirmed bookings at the same time.

## 7. How one turn flows through the code

Example: caller says *"I'd like to book a check-up Saturday morning."*

1. **`voice/stt.py`**: Whisper converts audio to text.
2. **`services/call_service.py`**: runs the turn, times each stage, and calls the agent.
3. **`agent/graph.py`**: `classify` (using `agent/rules.py` and `agent/guardrails.py`) picks a route. Emergencies, handoff, yes/no on a pending change, thanks/hello, advice and noise are all handled here with no LLM. Anything else goes to `tool_agent`.
4. **`agent/nodes.py` → `tool_agent`**: sends the LLM the instructions (`prompts.py`), the last 6 turns, and the tool list.
5. The LLM replies with a **tool request** such as `check_availability(when="Saturday morning")`.
6. **`tools/registry.py`** runs it: logs it to the `tool_calls` table, counts metrics, and catches errors.
7. **`tools/check_availability.py`** parses "Saturday morning" (`agent/timeparse.py`), asks `services/scheduling.py` for free slots, and returns ready-made words ("I have 9:00, 9:30 or 10:00"). The slots are stored as A/B/C in call state (`agent/state.py`).
8. Caller: "the second one." The code works out B (`timeparse.pick_offered`) and tells the LLM.
9. LLM requests `book_appointment(name, reason, slot_id="B", phone)`. **Checks in `tools/base.py`:** the slot must have been offered, and the phone number must have been said by the caller. If not, the call is refused with a reason, and the LLM must ask the caller.
10. If valid, the booking is **staged** (`agent["pending"]`) and read back.
11. Caller says "yes". `rules.is_yes` → route `agent_confirm` → `booking_service.create()` writes to PostgreSQL.
12. Ava says the reference number. **`voice/tts.py`** turns it into audio.
13. The turn is saved: route, timings, similarity score. Logs and metrics are updated.

### Knowledge question (RAG) flow

"How much is a check-up?" → LLM calls `get_clinic_info` → `rag/retriever.py` embeds the question and compares it with FAQ section vectors → if the best score is below **0.35**, return "no information"; otherwise the matching section goes to the LLM to phrase an answer. If the LLM writes a price or time in its own words, the reply is rejected and it must use a tool (`guardrails.states_unverified_fact`).

### Where the FAQ comes from (offline step)

`data/raw/faq.md` → `rag/loader.py` (read) → `rag/chunker.py` (split at each `## heading`) → `rag/embedder.py` (text → vector) → `rag/store.py` (save to `data/processed/`). Run by `scripts/ingest.py`. At startup the app loads the saved index; if the documents changed it notices (a fingerprint is stored) and re-embeds.

## 8. Every file: what it is and why it exists

### Root

| File | Purpose / why |
|---|---|
| `app.py` | Wiring only: builds each component once, creates the FastAPI app, mounts the Gradio page. Kept thin so it's easy to read |
| `.env` / `.env.example` | Secrets and settings. `.env` is git-ignored; the example is the template |
| `requirements.txt` | Pinned-range dependencies |
| `Dockerfile`, `.dockerignore` | Container build (not yet verified) |
| `pytest.ini` | Test paths |
| `README.md` | Setup and usage |

### `data/` and `scripts/`

| File | Purpose / why |
|---|---|
| `data/raw/faq.md` | The clinic's knowledge. Edit this to change what Ava knows |
| `data/processed/` | Saved search index, so startup is fast |
| `scripts/ingest.py` | Builds the index (`--check` tells you if it's stale; usable in CI) |

### `src/voice_agent/`

| File | Purpose / why |
|---|---|
| `config.py` | All settings from environment variables, with validation and secret hiding. Fails at startup on bad values |
| `logging_setup.py` | JSON logs with the call ID on every line, so one call can be traced |
| `monitoring.py` | Prometheus counters/histograms for `/metrics` |
| **`llm/client.py`** | The only file that talks to an LLM. Handles rate limits (429), malformed tool calls (retry once), reasoning parameters, token metrics. Isolating it makes the provider swappable |
| **`rag/loader.py`** | Reads the markdown documents |
| `rag/chunker.py` | Splits by heading so each chunk is one topic |
| `rag/embedder.py` | Text → vectors with MiniLM |
| `rag/store.py` | Saves/loads the vector index with a fingerprint and model check |
| `rag/retriever.py` | Search: meaning score + small keyword bonus, similarity threshold (0.35), optional reranker |
| **`tools/base.py`** | `Tool`, `ToolResult`, `ToolContext` and the **safety checks**: slot ids must be offered, booking ids must be known, phones must be heard, arguments masked for audit |
| `tools/registry.py` | Runs tools, writes the audit row, counts metrics, contains errors |
| `tools/check_availability.py` | Free slots |
| `tools/search_knowledge.py` | RAG tool (`get_clinic_info`) |
| `tools/check_status.py` | `find_bookings` by phone number |
| `tools/create_booking.py` | Stage a booking |
| `tools/reschedule_booking.py` | Stage a move |
| `tools/cancel_booking.py` | Stage a cancellation |
| **`agent/graph.py`** | LangGraph: the route map and `VoiceAgent.run_turn` |
| `agent/nodes.py` | The nodes: emergency, urgent, handoff, advice, greeting, thanks, unclear, `tool_agent`, `agent_confirm` |
| `agent/state.py` | What a call remembers: offered slots, pending change, known bookings, heard phones, failures |
| `agent/prompts.py` | Instructions to the LLM |
| `agent/guardrails.py` | Emergency and advice word lists (typo-tolerant), and the "unverified fact" check |
| `agent/rules.py` | Intent routing, yes/no, thanks, noise, phone extraction, name correction |
| `agent/timeparse.py` | Dates and times from speech ("the 12th", "half past ten"); exact rules, not an LLM |
| `agent/speech.py` | Spells times/digits so the voice reads them well |
| `memory/database.py` | SQLAlchemy engine, table creation, safe in-place upgrade |
| `memory/models.py` | Tables: Call, Message, Booking, ToolCall, UnansweredQuestion |
| `api/routes.py` | `/health`, `/metrics`, bookings API, Basic-auth login |
| `api/schemas.py` | Response shapes |
| `voice/stt.py` | Speech-to-text |
| `voice/tts.py` | Text-to-speech |
| `services/call_service.py` | Orchestrates one turn and records it |
| `services/booking_service.py` | Create/reschedule/cancel/suggest/lookup with an injectable clock (testable) |
| `services/scheduling.py` | Computes slots from opening hours (not stored) |
| `services/mock_clinic.py` | Pretend clinic system that receives booking events (stands in for a real integration) |
| `ui/demo_page.py` | The call page |
| `ui/dashboard.py` | Calls, bookings, tool calls, unanswered questions (written, but not shown on the page) |

### `evals/` and `tests/`

| File | Purpose / why |
|---|---|
| `evals/run_evals.py` | Offline quality check: emergency catch rate (must be 100%), routing, RAG section accuracy, out-of-scope refusal. Exits non-zero if an emergency is missed, so it can gate a release. Latest: 11/11, 36/36, 15/15, 3/3 |
| `evals/test_cases.json` | The 54 cases |
| `evals/bench_tool_calling.py` | Measures how well a real LLM picks tools on 40 sentences. Use before switching models |
| `evals/run_wer.py` | Speech-to-text word error rate on your recordings |
| `tests/` | 398 tests: tools and refusals, whole conversations with a scripted LLM, calendar, double-booking, dates, guardrails, RAG, LLM client, login, endpoints |

## 9. Design decisions to be ready to defend

**Why not let the LLM do everything?** It is right most of the time, but "most" isn't enough for bookings. The best models scored about 36–39/40 on the tool benchmark. The remaining mistakes must not become wrong bookings, so code enforces the rules.

**Why stage-then-confirm with code?** A spoken "yes" is detected by plain code (`rules.is_yes`), so the LLM can't decide for the caller. It also gives the caller a chance to catch a misheard name or time.

**Why compute the calendar instead of storing slots?** Slots follow opening hours, so storing them means keeping a table in sync. Computing means the only stored fact is a confirmed booking.

**Why a unique index for double booking?** Application checks can race (two callers at the same moment). The database guarantees it. Tested with 8 simultaneous attempts: 1 booking, 7 refusals.

**Why timezone handling?** Stored in UTC, shown in Europe/London, so 9am stays 9am when clocks change.

**Why not pgvector?** The FAQ has 10 chunks; an in-memory store is simpler and instant. The retriever sits behind `rag/store.py`, so it can be swapped for LangChain's PGVector when the documents grow.

**Why a similarity threshold (0.35)?** It separates real clinic questions (scores about 0.39 and up) from out-of-scope ones (about 0.30 and below) on this data. Without it, the nearest chunk would be used even when irrelevant, which causes confident wrong answers.

**Why rules for dates and emergencies, not an LLM?** They must be deterministic, fast and testable. An emergency detector that is occasionally wrong is not acceptable; ours is deliberately over-cautious.

**Why audit every tool call?** Accountability: you can see what the AI tried, what was refused and why. Personal data in the arguments is masked.

## 10. Questions an interviewer may ask

**Q: What happens if the LLM hallucinates?**
For clinic facts, the answer must come from retrieved text; with no match above the threshold Ava says she doesn't know. For times, prices and booking details, the code speaks them, not the LLM. If the LLM states a time or price in its own words, the reply is rejected and it is forced to call a tool.

**Q: What if the LLM calls a tool with made-up arguments?**
Checked in `tools/base.py`: a slot id must be one that was offered, a booking id must come from a lookup or this call, a phone number must have been said by the caller. Otherwise the call is refused, the LLM is told why, and it can't call another tool straight after a refusal.

**Q: How do you prevent double booking?**
A unique partial index on the start time of confirmed bookings in PostgreSQL. A second attempt fails, and the caller hears that the time was just taken. Cancelled bookings free the time.

**Q: How do you handle emergencies?**
Keyword and stem lists with typo tolerance run before the LLM. Severe → "call 999". Urgent dental → urgent line (NHS 111 if closed). All fixed text. It's not a clinically validated triage tool; say so.

**Q: How do you handle speech recognition errors?**
Read-back before saving; a spoken name correction is understood; "I can't brite" (a real misrecognition of "can't breathe") is in the emergency tests. Names remain the weak spot; a larger hosted Whisper would help.

**Q: How did you choose the LLM?**
A benchmark of 40 realistic caller sentences measuring whether the right tool and arguments were chosen. Hosted models like `qwen3.8-27b` and `gpt-oss` scored 36–39 out of 40; the best local 3B model scored about 30 and sometimes invented a phone number. I also had to fix model-specific problems (for example one model got confused by its own earlier turns, and Groq sometimes returns a malformed tool call, so the client retries once).

**Q: Why does it need the internet / a key?**
Only the LLM is hosted. Speech and search run locally. A local Ollama model works through the same client, but with lower reliability.

**Q: How would you scale this?**
Today call state is in memory, so it's one process. For scale: Redis for call state, a streaming telephony layer (Twilio/SIP) instead of record-and-send, a paid LLM tier with higher limits, Alembic migrations, and Prometheus/Grafana on `/metrics`.

**Q: What about privacy and compliance (GDPR, patient data)?**
Logs omit transcripts by default (`LOG_TRANSCRIPTS=false`); tool audit arguments are masked; phone numbers are partly hidden on the dashboard; the dashboard has a login. The demo uses invented data. Real use needs a provider agreement, data retention rules, and consent to record.

**Q: How do you test an LLM application?**
Three levels. (1) Unit and conversation tests with a scripted fake LLM, so they're deterministic and free: 398 tests. (2) Offline evals for the non-LLM parts, with a hard gate on emergencies. (3) A benchmark that does use a real LLM, run before changing models.

**Q: What's the weakest part?**
Be honest: it is turn-based (record, wait, reply), not streaming, so callers can't interrupt; one dentist only; no bank holidays; appointments are found by phone number without a second identity check; names are mis-transcribed sometimes; the free LLM tier limits throughput and may let the provider use the prompts; and there's no real phone line or real clinic system yet (`mock_clinic.py` stands in).

**Q: What would you do next?**
Real telephony with streaming audio and barge-in; integrate the clinic's actual booking system; date-of-birth check before revealing a booking; multi-dentist support and holidays; SMS reminders; tracing with LangSmith/Langfuse; a clinically validated emergency classifier.

**Q: Why LangGraph instead of a plain function with if/else?**
The graph makes each route an independent, testable node, and the map of routes is visible in one file. It also gives room to add nodes (for example a human-handoff step) without untangling conditionals.

**Q: Did you use AI to write this?**
Answer honestly. Whatever your real answer is, be prepared to explain any file in this guide, since they will likely ask you to open one. Practise on `agent/graph.py`, `agent/nodes.py` and `tools/base.py`.

## 11. Numbers to remember

| Fact | Value |
|---|---|
| Tests | 398 passing |
| Offline eval cases | 54 (emergencies 11/11, routing 36/36, RAG section 15/15, out-of-scope refusals 3/3) |
| Tool-choice benchmark | 40 cases; best hosted models 36–39; best local 3B about 30 |
| Tools | 6 |
| Similarity threshold | 0.35 |
| Opening hours | Mon–Fri 8–18, Sat 9–13, 30-minute slots, 28-day horizon, 60-minute notice |
| Conversation memory | last 6 turns |
| Tool rounds per turn | max 3 |
| Free Groq limit | about 1,000 requests a day, 8,000 tokens a minute per model |
| Source size | about 3,700 lines (was about 4,700) |

## 12. If something breaks during the demo

- **"Rate limited" reply:** wait 10 seconds; the free tier allows about 5 turns a minute.
- **Microphone problem:** use the text box.
- **Name misheard:** this is a good moment to show the read-back and the correction.
- **App won't start:** check PostgreSQL is running and `.env` has `LLM_API_KEY` and `DATABASE_URL`. Run `python -m pytest -q` to show the tests pass even without the LLM.
- **Port 7860 busy:** another copy is running; close it first.
