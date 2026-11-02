# Ava: AI voice receptionist for Sunrise Dental

Ava answers the phone for **Sunrise Dental**, an invented dental clinic. A caller speaks, and Ava:

- answers questions (opening hours, prices, insurance, cancellations, location, children, emergency slots) **only from the clinic's own documents**, and says "I don't have that information" instead of guessing
- **books real appointments**: she checks the clinic calendar, offers free times, reads the booking back, and gives a reference number
- **finds, moves and cancels** an existing appointment (by phone number), and answers "do you have anything on Saturday?"
- spots emergencies **before** the LLM is involved: severe ones get "call 999", urgent dental problems get the urgent line (and NHS 111 if closed)
- hands over to a person when asked, or after not understanding twice; never gives dental or medical advice
- logs every call, turn and tool call with timings to a database, and saves bookings

All clinic details, names, prices and phone numbers are invented. No real patient data is used.

---

## How it works

```
 Browser (Gradio page)                                      Monitoring
   │ microphone / text                                     /health  /metrics  logs/voice_agent.log
   ▼
 ┌────────────────────────── CallService: one turn ─────────────────────────────┐
 │  1. EARS    Whisper (speech-to-text)  ── caller text                          │
 │                                                                              │
 │  2. BRAIN   LangGraph flow, checked in this order (agent/graph.py):           │
 │       severe emergency ──► "call 999"                    fixed reply, no LLM  │
 │       urgent dental ─────► urgent line / NHS 111         fixed reply, no LLM  │
 │       asks for a person ─► hand over                     fixed reply, no LLM  │
 │       a change is waiting for "yes" ► the CODE says yes/no (agent_confirm)    │
 │       thanks / hello / asks for advice / noise ► fixed replies, no LLM        │
 │       anything else ─────► tool_agent: the LLM answers, or chooses a tool     │
 │                                                                              │
 │  3. MOUTH   Piper (text-to-speech) ── reply audio                               │
 │  4. RECORD  database (calls, messages, bookings, tool calls), JSON log, metrics│
 └──────────────────────────────────────────────────────────────────────────────┘
```

### The tool agent

The LLM does not follow a script. It reads what the caller said and **chooses a tool**:

| Tool | What it does |
|---|---|
| `check_availability(when)` | Free times from the calendar (database) |
| `get_clinic_info(question)` | Answers from the clinic's documents (RAG, below) |
| `find_bookings(phone)` | The caller's upcoming appointments |
| `book_appointment(name, reason, slot_id, phone)` | **Stages** a booking and reads it back |
| `reschedule_appointment(booking_id, slot_id)` | **Stages** a move |
| `cancel_appointment(booking_id)` | **Stages** a cancellation |

The LLM decides *what to do*; the code decides *whether it may* ([tools/base.py](src/voice_agent/tools/base.py)):

- **A change needs a spoken yes, and the code makes it, not the LLM.** The book/move/cancel tools only stage the change. Ava reads it back ("Your name is..., your appointment would be..., shall I book that?"). When the caller says yes, `agent_confirm` makes the change. If the caller says no, or something else, nothing is changed.
- **No invented values.** Times are slot ids (A, B, C) from the last `check_availability`; booking ids must come from `find_bookings` (or a booking made in this call); a phone number is only accepted if the caller actually said it. A refused call tells the LLM why, and the LLM must then ask the caller (no other tool is allowed straight after a refusal).
- **The code speaks the facts.** A successful tool returns the exact words, so the LLM cannot garble a time, price or reference number. If the LLM writes a time or a price in its own words, the reply is rejected and it is forced to use a tool ([guardrails.py](src/voice_agent/agent/guardrails.py)). Medical advice in its words is blocked.
- **Audit trail.** Every tool call is stored in the `tool_calls` table (tool, accepted or refused, time taken, arguments with names, reasons and phone numbers masked) and shown on the dashboard. `/metrics` counts tool calls, LLM requests and tokens.
- **If the LLM is down or rate limited:** Ava says so, and after two failures in a row she hands over to a person.

To add a tool: write a class in [src/voice_agent/tools/](src/voice_agent/tools/) (name, description, parameters, `run`) and add it to `default_tools()` in `registry.py`.

**Which LLM?** Any service that speaks the OpenAI chat API and can choose tools: Groq (the default), Gemini, xAI Grok, or a local Ollama (`http://127.0.0.1:11434/v1`). On 40 realistic caller sentences ([evals/bench_tool_calling.py](evals/bench_tool_calling.py)): `qwen3.8-27b` and the two `gpt-oss` models were right 36-39 times in 40; the best local 3B model, 30 (and it made unsafe calls such as inventing a phone number). That is why the code-side checks exist: a model that is right 90% of the time must still not be able to book on its own. Free tiers (for example Groq: 1,000 requests a day and 8,000 tokens a minute per model) are enough for a demo, not for real call volume, and **free tiers may let the provider use your prompts**: use invented data only.

### The knowledge search (RAG)

`data/raw/*.md` → **loader** reads them → **chunker** (LangChain's markdown splitter) cuts at each `## heading` → **embedder** (LangChain's HuggingFaceEmbeddings, runs on this computer) turns each chunk into a vector → **store** (LangChain's in-memory vector store) saves and searches the vectors → **retriever** scores each chunk by meaning plus a small bonus for shared key words (so short questions like "location" or "what is your email" still find their section), then applies the similarity limit (`MATCH_THRESHOLD`) and an optional reranker. When nothing scores above the limit, the answer is "I don't have that information" without asking the LLM at all.

Run `python scripts/ingest.py` after changing a document (or `EMBED_MODEL`). It saves the index in `data/processed`, and the app loads that instead of embedding on every start. If the index no longer matches the documents, the app notices (a fingerprint of the text is stored with it), logs a warning, and embeds them itself: slower to start, never wrong. `python scripts/ingest.py --check` tells you whether the saved index is up to date.

### Appointments

The clinic calendar is **worked out** from the opening hours (Mon-Fri 8am-6pm, Sat 9am-1pm, one dentist, 30-minute slots, up to 4 weeks ahead, at least 1 hour's notice), not stored. A time is free when no confirmed booking has that start time.

- **No double booking, ever.** The `bookings` table has a unique index on the start time of *confirmed* bookings. Two callers choosing the same time at the same moment: one wins and the other hears "that time has just been taken". (Checked on PostgreSQL: 8 simultaneous attempts gave 1 booking and 7 refusals.) A cancelled booking frees its time.
- **Dates and times** ("Saturday morning", "tomorrow at 3", "the 12th", "half past ten") are understood by plain rules in [timeparse.py](src/voice_agent/agent/timeparse.py), because they must be exact. Times are stored in UTC and shown in the clinic's time zone, so 9am stays 9am when the clocks change. [speech.py](src/voice_agent/agent/speech.py) spells dates, times and digits out, because the voice reads "9:30" badly.
- **If the clinic's own system is down**, the booking is still saved (`sync_status = pending`) and the dashboard shows what has not been sent.
- **Upgrading an existing database:** on start, the app adds the new columns to an older `bookings` table and keeps the old rows. It is safe to run repeatedly.

### Project layout

```
voice-agent/
├── app.py                         starts the app (FastAPI + the Gradio page): only wires things together
├── requirements.txt  Dockerfile  .env.example  pytest.ini
│
├── data/
│   ├── raw/faq.md                 the clinic's own documents: edit these to change what Ava knows
│   └── processed/                 the saved search index, made by scripts/ingest.py
├── scripts/
│   └── ingest.py                  one-off job: load the documents, split, embed, save the index
│
├── src/voice_agent/
│   ├── config.py                  every setting, read from environment variables
│   ├── logging_setup.py  monitoring.py     JSON logs tagged with the call ID, and /metrics
│   │
│   ├── llm/client.py              the ONLY file that talks to an LLM (LangChain's ChatOpenAI behind a thin wrapper)
│   ├── rag/                       loader.py, chunker.py, embedder.py, store.py, retriever.py
│   ├── tools/                     base.py, registry.py + one file per tool (check_availability.py, search_knowledge.py,
│   │                              check_status.py, create_booking.py, reschedule_booking.py, cancel_booking.py)
│   ├── agent/                     state.py, prompts.py, nodes.py, graph.py, guardrails.py   (the safety rules)
│   │                              + rules.py (routing, yes/no, phone numbers), timeparse.py, speech.py
│   ├── memory/                    database.py, models.py   (calls, messages, bookings, tool calls)
│   ├── api/                       routes.py, schemas.py    (/health, /metrics, the bookings API)
│   │
│   ├── voice/                     stt.py, tts.py           (extra: this is a VOICE agent)
│   ├── services/                  call_service.py, booking_service.py, scheduling.py, mock_clinic.py
│   │                              (extra: one phone call, the appointment calendar, the pretend clinic system)
│   └── ui/                        demo_page.py, dashboard.py   (extra: the web page)
│
├── evals/                         test_cases.json, run_evals.py (+ run_wer.py, bench_tool_calling.py)
└── tests/                         test_tools.py, test_retriever.py, test_guardrails.py (+ the conversation tests)
```

What LangChain and LangGraph do here: **LangGraph** runs the turn as a graph of small steps (`agent/graph.py`). **LangChain** talks to the LLM (`llm/client.py`) and handles documents, splitting, embeddings and the vector store (`rag/`). Our own code is everything that has to be exactly right: the safety rules, the tool checks, the calendar and the double-booking protection, date understanding, and the audit trail.

---

## Setup (Windows PowerShell)

You need **Python 3.11 or newer** (tested on 3.13) and about **3 GB of free memory** while it runs, plus a **PostgreSQL** database and an **LLM API key** (a free Groq key is enough for a demo: https://console.groq.com).

```powershell
cd "C:\path\to\voice-agent"

# 1. (Recommended) create a virtual environment: a private folder of packages just for this project
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install the packages (PyTorch is large; this can take a few minutes)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Settings: copy the example and fill in the LLM key and the database address
copy .env.example .env
notepad .env          # LLM_API_KEY=...   and   DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/voice_agent

# 4. Build the search index from the documents in data/raw (optional: the app builds it itself if you skip this)
python scripts/ingest.py
```

**ffmpeg:** not needed by this app. The page sends raw audio, and the code converts it itself.

The first start downloads the speech and search models (about 1.2 GB) into your Hugging Face cache.

## Run

```powershell
python app.py
```

Wait for `Ready.`, then open **http://127.0.0.1:7860**.

- **Call Ava** tab: press **Start call** to hear the greeting, then record a question with the microphone. There is also a text box, as a backup if the microphone misbehaves. The right-hand panel shows the route taken, the search similarity score, and the time for each stage. **End call** saves the call's final outcome.
- **Dashboard** tab: press **Refresh** to see calls by outcome, average stage times, handoff rate, emergencies caught, the **upcoming appointments** (phone numbers are partly hidden), every **tool call** the LLM made, unanswered questions, and any call's full conversation.

| Address | What it is |
|---|---|
| http://127.0.0.1:7860/health | Is the app OK? (database reachable, which models) |
| http://127.0.0.1:7860/metrics | Live numbers for monitoring tools (Prometheus format) |
| http://127.0.0.1:7860/api/mock-consultation/bookings | Booking events (created / rescheduled / cancelled) the pretend clinic system has received |

`python -m voice_agent.services.mock_clinic` (run from the `src` folder) prints the received bookings in the terminal.

### Settings

Every setting can be changed with an environment variable or in `.env`. See [.env.example](.env.example). The most useful:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | Groq, (none), `qwen/qwen3.8-27b` | The LLM. Any OpenAI-compatible service that can choose tools; for a local Ollama use `http://127.0.0.1:11434/v1` |
| `DATABASE_URL` | local PostgreSQL | e.g. `postgresql+psycopg://postgres:postgres@localhost:5432/voice_agent` |
| `DASHBOARD_USER`, `DASHBOARD_PASSWORD` | `admin`, (none) | Login for the web page, dashboard and bookings API. **Set a password before anyone else can reach the app** (a warning is printed if you listen on another address without one) |
| `DOCS_DIR`, `PROCESSED_DIR`, `USE_PREBUILT_INDEX` | `data/raw`, `data/processed`, `true` | Where the documents and the saved search index are |
| `MATCH_THRESHOLD`, `TOP_K` | `0.35`, `2` | Minimum similarity to answer, and how many chunks the LLM sees. 0.35 separates real clinic questions (scores 0.39 and up) from out-of-scope ones (0.30 and below) |
| `STT_ENGINE`, `STT_MODEL` | `transformers`, `openai/whisper-small` | `faster_whisper` with `distil-small.en` is several times faster on a CPU |
| `CLINIC_TIMEZONE`, `SLOT_MINUTES` | `Europe/London`, `30` | Calendar time zone and appointment length |
| `BOOKING_HORIZON_DAYS`, `MIN_NOTICE_MINUTES` | `28`, `60` | How far ahead callers can book, and the shortest notice |
| `AGENT_HISTORY_TURNS`, `AGENT_MAX_TOOL_ROUNDS` | `6`, `3` | How much of the call the LLM sees, and tool calls allowed in one turn |
| `LOG_LEVEL`, `LOG_TRANSCRIPTS` | `INFO`, `false` | `DEBUG` shows more detail; transcripts are personal data and are kept out of logs unless you turn this on |

### Logs and monitoring

- **Screen:** short readable lines. Each one shows the call ID in brackets.
- **`logs/voice_agent.log`:** one JSON object per line, for example `{"event": "turn_finished", "call_id": "...", "route": "faq_answered", "stt_ms": 1840, "llm_ms": 912, "tts_ms": 1530, ...}`. The file starts again at 5 MB and keeps 5 old files, so it never fills the disk. Caller words are left out unless `LOG_TRANSCRIPTS=true`.
- **`/metrics`:** calls, turns by route, emergencies by tier, unanswered questions, bookings (saved, rescheduled, cancelled, slot conflicts, sent), tool calls by tool and result, LLM requests and tokens, errors by component, plus timing histograms for each stage. Point Prometheus/Grafana or Azure Monitor at it to draw charts and set alerts.
- **Database:** every turn is stored with its route, similarity score and three timings. Here `llm_ms` is the whole "brain" step. The LLM-only time is in the JSON log as `llm_call_ms`.

---

## Tests and evaluation

```powershell
python -m pytest                       # about 400 tests, under a minute, no LLM key needed (the LLM is faked)
python evals/run_evals.py              # safety, routing and knowledge-search checks: offline, uses no LLM requests
python evals/bench_tool_calling.py --hosted qwen/qwen3.8-27b   # how well a model chooses tools (about 40 LLM requests)
python evals/run_wer.py                # speech-to-text accuracy on your own recordings
```

- **pytest** covers the tools and the checks that refuse unsafe calls, the whole conversation (booking, moving, cancelling, availability, with a scripted LLM and a frozen clock), the calendar and double-booking protection, date understanding, the safety rules, the knowledge search (with fakes, and with the real embedding model), the LLM client (against a fake server), the login, and the web endpoints.
- **run_evals.py** prints the emergency catch rate (goal 100%), routing accuracy, whether the knowledge search finds the right section, and whether it refuses out-of-scope questions. It exits with an error if any emergency is missed, so it can block a release automatically. Latest: emergencies 11/11, routing 36/36, right section 15/15, out-of-scope refusals 3/3.
- **bench_tool_calling.py** gives a model 40 caller sentences and checks it picks the right tool with the right arguments. Use it before switching `LLM_MODEL`.
- **run_wer.py** measures Word Error Rate. Put `name.wav` + `name.txt` (the correct sentence) pairs in `evals/recordings/`.

## Docker

```powershell
docker build -t ava-voice-agent .
docker run -p 7860:7860 --env-file .env -v ava-models:/models -v ava-data:/app/data ava-voice-agent
```

The image will be large: an **estimated** 2-3 GB, mostly PyTorch, even with the CPU-only build. The models (about 1.2 GB) are downloaded on first start into the `ava-models` volume rather than baked into the image.
_Not yet verified: the image has not been built on the development machine (Docker Desktop was not running). Build it once before relying on it._

---

## Limits and next steps

**Limits (this is a demo):**
- Turn-based: record, stop, wait. No live streaming, and you can't interrupt Ava.
- The tool agent needs a capable hosted LLM. Small local models (3B-4B) were tried and are not reliable at choosing tools. The conversation goes to the LLM provider: for real patients use a paid tier with a data-processing agreement.
- The emergency check is keyword-based. It is deliberately over-cautious (false alarms are acceptable), but it is **not** a clinically validated triage tool.
- Speech-to-text mishears, names most of all ("Mugilesh" became "Mukilives"). That is why Ava reads a booking back before saving it. Groq also hosts `whisper-large-v3-turbo`, which would be far better than the local model, at the cost of sending audio out.
- One dentist (one appointment per time slot). Opening hours are a constant in [scheduling.py](src/voice_agent/services/scheduling.py) and must be kept in step with `data/raw/faq.md`. Bank holidays and clinic closures are not modelled.
- Appointments are found by phone number only (and only a number the caller says in the call is accepted), and there are no reminder texts or e-mails.
- Call state lives in memory. If the app restarts, calls in progress are lost (they are closed as "abandoned" on the next start).
- English only. Only tested with a few voices and accents.

**Next steps for production:**
- A real phone line (Twilio or SIP) with streaming audio so callers can interrupt; the clinic's real booking system in place of the mock file
- pgvector (LangChain's PGVector store) once the documents grow; Redis for live call state so several servers can share calls; Alembic migrations in place of the small built-in upgrade step
- Several dentists, holidays and closures, SMS/e-mail reminders, a second identity check (such as date of birth) before showing a booking
- Tracing with LangSmith or Langfuse; Grafana dashboards and alerts on `/metrics`
- A clinically validated emergency classifier alongside the keyword rules
