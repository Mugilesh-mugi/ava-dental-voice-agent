"""Measure the parts of Ava that do not need an LLM, and print a report. Nothing here makes an LLM request.

    python evals/run_evals.py

Each case in test_cases.json checks one of three things:

    expect_intent      the safety and routing rules send this message to the right place (an emergency to "call 999",
                       a request for a person to a handoff, a normal request on to the agent, ...)
    expect_topic       the knowledge search finds the right section of the clinic documents for this question
    expect_answerable  false: the documents have no answer, so the search must return nothing and Ava must say
                       "I don't have that information" instead of guessing

How well an LLM CHOOSES TOOLS is measured separately by evals/bench_tool_calling.py (it does use LLM requests).

Exit code is 1 if any emergency is missed, so this can run in CI (automatic checks before release).
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from voice_agent.agent import rules  # noqa: E402
from voice_agent.config import get_settings  # noqa: E402
from voice_agent.rag import load_chunks  # noqa: E402
from voice_agent.rag.embedder import build_embedder  # noqa: E402
from voice_agent.rag.retriever import Retriever  # noqa: E402

EMERGENCY_INTENTS = {"severe", "urgent"}


def check(case: dict, retriever: Retriever) -> tuple[bool, str]:
    """Did this case pass, and what actually happened (for the report)."""
    if "expect_intent" in case:
        got = rules.decide_intent(case["text"])
        return got == case["expect_intent"], got
    result = retriever.search(case["text"])
    top = result.matches[0][0].metadata["title"] if result.matches else None
    detail = f"{top or 'nothing found'} (best score {result.best_score:.3f})"
    if "expect_topic" in case:
        return top == case["expect_topic"], detail
    return not result.answerable, detail


def run() -> int:
    settings = get_settings()
    cases = json.loads((ROOT / "evals" / "test_cases.json").read_text(encoding="utf-8"))

    print(f"Loading embedding model {settings.embed_model} ...")
    retriever = Retriever(load_chunks(settings.docs_dir), build_embedder(settings.embed_model),
                          settings.match_threshold, settings.top_k)
    print(f"threshold: {settings.match_threshold}   |   cases: {len(cases)}\n")

    results, by_category = [], defaultdict(lambda: [0, 0])
    for case in cases:
        start = time.perf_counter()
        passed, got = check(case, retriever)
        results.append({**case, "passed": passed, "got": got, "ms": round((time.perf_counter() - start) * 1000)})
        by_category[case["category"]][0] += passed
        by_category[case["category"]][1] += 1

    def share(part, whole):
        return f"{len(part)}/{len(whole)} = {len(part) / len(whole):.0%}" if whole else "n/a"

    emergencies = [r for r in results if r["category"].startswith("emergency")]
    caught = [r for r in emergencies if r["got"] in EMERGENCY_INTENTS]
    right_tier = [r for r in emergencies if r["passed"]]
    routing = [r for r in results if "expect_intent" in r]
    topics = [r for r in results if "expect_topic" in r]
    refusals = [r for r in results if r.get("expect_answerable") is False]

    print("=" * 64)
    print(f"Emergency catch rate (any emergency reply): {share(caught, emergencies)}   (goal: 100%)")
    print(f"Emergency correct tier (999 vs urgent):     {share(right_tier, emergencies)}")
    print(f"Routing rules, all cases:                   {share([r for r in routing if r['passed']], routing)}")
    print(f"Knowledge search finds the right section:   {share([r for r in topics if r['passed']], topics)}")
    print(f"Knowledge search refuses out-of-scope:      {share([r for r in refusals if r['passed']], refusals)}")
    print("-" * 64)
    for category, (passed, total) in by_category.items():
        print(f"  {category:18} {passed}/{total}")
    print("=" * 64)

    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for r in failures:
            expected = r.get("expect_intent") or r.get("expect_topic") or "nothing (out of scope)"
            print(f"  [{r['id']}] \"{r['text']}\"\n      expected {expected}, got {r['got']}")
    else:
        print("\nNo failures.")

    out_dir = ROOT / "evals" / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull results saved to {out_file.relative_to(ROOT)}")
    return 0 if len(caught) == len(emergencies) else 1


if __name__ == "__main__":
    sys.exit(run())
