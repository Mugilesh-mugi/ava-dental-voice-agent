"""Live numbers for monitoring, published at /metrics in the Prometheus format.

Prometheus is a common monitoring tool: it visits /metrics every few seconds and
stores the numbers, and Grafana (or Azure Monitor) draws charts and sends alerts,
for example "alert if any emergency was routed to the LLM" or "alert if the LLM
error rate goes above 5%".

Counter   = a number that only goes up (calls started, emergencies caught).
Histogram = records how long things take, so we can see averages and slow outliers.
"""
from prometheus_client import Counter, Gauge, Histogram

CALLS_STARTED = Counter("voice_calls_started_total", "Calls started")
CALLS_ENDED = Counter("voice_calls_ended_total", "Calls ended, by final outcome", ["outcome"])
ACTIVE_CALLS = Gauge("voice_active_calls", "Calls in progress")
TURNS = Counter("voice_turns_total", "Caller turns, by route taken", ["route"])
EMERGENCIES = Counter("voice_emergencies_total", "Emergencies detected", ["tier"])
UNANSWERED = Counter("voice_unanswered_total", "Questions the FAQ could not answer")
BOOKINGS = Counter("voice_bookings_total", "Bookings, by result", ["result"])
ERRORS = Counter("voice_errors_total", "Errors, by component", ["component"])
TOOL_CALLS = Counter("voice_tool_calls_total", "Tool calls chosen by the LLM, by tool and result", ["tool", "result"])
LLM_REQUESTS = Counter("voice_llm_requests_total", "Requests to a hosted LLM, by outcome", ["outcome"])
LLM_TOKENS = Counter("voice_llm_tokens_total", "Tokens used on a hosted LLM (watch this against the free-tier limit)",
                     ["kind"])
STAGE_SECONDS = Histogram(
    "voice_stage_seconds", "Time per stage of a turn", ["stage"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 20, 30),
)
SIMILARITY = Histogram(
    "voice_faq_similarity", "Best FAQ similarity score per question",
    buckets=(0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
