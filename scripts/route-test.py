#!/usr/bin/env python3
"""End-to-end routing demonstration for ProjectLodestar.

Verifies that task classification correctly routes cheap/free tasks to local
Ollama models and only escalates to paid APIs for complex work.

Run from the repo root with the venv active:

    cd ~/Documents/github-repos/ProjectLodestar
    source .venv/bin/activate
    python scripts/route-test.py

No API calls are made to paid providers — the script only tests T1 and T2
routing (Ollama) and uses model_override/task_override to verify routing
decisions without live cloud calls.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)

from modules.routing.proxy import LodestarProxy
from modules.routing.router import SemanticRouter

RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"


def check(label: str, condition: bool, detail: str = "") -> bool:
    mark = f"{GREEN}✅{RESET}" if condition else f"{RED}❌{RESET}"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {mark}  {label}{suffix}")
    return condition


def main() -> int:
    print(f"\n{BOLD}Lodestar Routing End-to-End Test{RESET}")
    print("=" * 50)

    failures = 0

    # ------------------------------------------------------------------
    # 1. Router classification (no LLM calls)
    # ------------------------------------------------------------------
    print(f"\n{BOLD}1. Task classification (no LLM calls){RESET}")
    router = SemanticRouter({})
    router.start()

    cases = [
        ("code_generation", "gpt-3.5-turbo", "T1 → Ollama (free)"),
        ("bug_fix",         "gpt-3.5-turbo", "T1 → Ollama (free)"),
        ("documentation",   "gpt-3.5-turbo", "T1 → Ollama (free)"),
        ("general",         "gpt-3.5-turbo", "T1 → Ollama (free)"),
        ("refactor",        "local-reasoning", "T2 → Ollama reasoning (free)"),
        ("debug_analysis",  "local-reasoning", "T2 → Ollama reasoning (free)"),
        ("code_review",     "claude-sonnet",  "T3 → Anthropic (paid)"),
        ("architecture",    "claude-sonnet",  "T3 → Anthropic (paid)"),
        ("security_audit",  "claude-sonnet",  "T3 → Anthropic (paid)"),
    ]

    for task, expected_alias, label in cases:
        got = router.route(task)
        ok = check(f"{task:<20} → {expected_alias:<20}", got == expected_alias, label)
        if not ok:
            failures += 1
            print(f"         got: {got!r}")

    # ------------------------------------------------------------------
    # 2. Live T1 call — qwen2.5-coder:3b via Ollama (free)
    # ------------------------------------------------------------------
    print(f"\n{BOLD}2. Live T1 call — qwen2.5-coder:3b via Ollama{RESET}")
    print(f"  {YELLOW}(This makes a real Ollama call — no cloud cost){RESET}")

    proxy = LodestarProxy()

    try:
        result = proxy.complete(
            prompt="Write a one-line Python function that returns True if a number is even.",
            task_override="code_generation",
        )
        model_used = result.get("model", "?")
        content = result.get("content", "").strip()
        is_t1 = model_used == "gpt-3.5-turbo"
        ok = check(
            f"Routed to T1 alias (gpt-3.5-turbo)",
            is_t1,
            f"model alias used: {model_used!r}",
        )
        if not ok:
            failures += 1
        print(f"\n  Response preview:\n  {content[:200]}")
    except Exception as exc:
        print(f"  {RED}❌  T1 live call failed: {exc}{RESET}")
        failures += 1

    # ------------------------------------------------------------------
    # 3. Live T2 call — deepseek-r1:7b via Ollama (free)
    # ------------------------------------------------------------------
    print(f"\n{BOLD}3. Live T2 call — deepseek-r1:7b via Ollama{RESET}")
    print(f"  {YELLOW}(This makes a real Ollama call — no cloud cost){RESET}")

    try:
        result = proxy.complete(
            prompt="In one sentence: what is memoisation?",
            task_override="refactor",
            max_tokens=60,
        )
        model_used = result.get("model", "?")
        content = result.get("content", "").strip()
        is_t2 = model_used == "local-reasoning"
        ok = check(
            "Routed to T2 alias (local-reasoning)",
            is_t2,
            f"model alias used: {model_used!r}",
        )
        if not ok:
            failures += 1
        print(f"\n  Response preview:\n  {content[:200]}")
    except Exception as exc:
        print(f"  {RED}❌  T2 live call failed: {exc}{RESET}")
        failures += 1

    # ------------------------------------------------------------------
    # 4. Routing decision for T3 (no live cloud call — just verify alias)
    # ------------------------------------------------------------------
    print(f"\n{BOLD}4. T3 routing decision (no cloud call made){RESET}")
    t3_alias = router.route("code_review")
    ok = check(
        "code_review routes to claude-sonnet (T3)",
        t3_alias == "claude-sonnet",
        "paid — not called in this test",
    )
    if not ok:
        failures += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 50}")
    if failures == 0:
        print(f"{GREEN}{BOLD}All checks passed.{RESET}")
        print("T1 and T2 tasks route to free Ollama — zero cloud spend.")
        print("T3 tasks route to claude-sonnet only when escalation is needed.")
    else:
        print(f"{RED}{BOLD}{failures} check(s) failed.{RESET}")

    proxy.stop()
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
