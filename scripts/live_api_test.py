#!/usr/bin/env python3
"""
Live API connectivity tests for all Lodestar model tiers.
Tests T1 (Ollama), T2 (deepseek-r1), T3 (claude-sonnet, grok-beta, perplexity).

Usage:
    cd /path/to/ProjectLodestar
    source ~/.env && .venv/bin/python scripts/live_api_test.py
"""

import os
import sys
import time
import traceback

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.routing.proxy import LodestarProxy

TESTS = [
    {
        "name":           "T1 · deepseek-coder:6.7b (Ollama)",
        "model_override": "gpt-3.5-turbo",
        "prompt":         "Reply with exactly: TIER1_OK",
        "max_tokens":     32,
        "tier":           1,
    },
    {
        "name":           "T2 · deepseek-r1:7b (Ollama reasoning)",
        "model_override": "local-reasoning",
        # R1 needs headroom to finish its <think> block before writing the answer.
        # 256 tokens is enough for a trivial prompt; real tasks need more.
        "prompt":         "Reply with exactly: TIER2_OK",
        "max_tokens":     256,
        "tier":           2,
    },
    {
        "name":           "T3 · claude-sonnet (Anthropic)",
        "model_override": "claude-sonnet",
        "prompt":         "Reply with exactly: TIER3_CLAUDE_OK",
        "max_tokens":     32,
        "tier":           3,
    },
    {
        "name":           "T3 · grok-beta (xAI)",
        "model_override": "grok-beta",
        "prompt":         "Reply with exactly: TIER3_GROK_OK",
        "max_tokens":     32,
        "tier":           3,
    },
    {
        "name":           "T3 · perplexity-sonar (Perplexity AI)",
        "model_override": "perplexity-sonar",
        "prompt":         "Reply with exactly: TIER3_PERPLEXITY_OK",
        "max_tokens":     32,
        "tier":           3,
    },
]

WIDTH = 50

def run_test(proxy, test):
    name   = test["name"]
    model  = test["model_override"]
    prompt = test["prompt"]

    print(f"  {'─' * WIDTH}")
    print(f"  TEST : {name}")
    print(f"  MODEL: {model}")
    start = time.time()
    try:
        result = proxy.complete(
            prompt=prompt,
            model_override=model,
            max_tokens=test.get("max_tokens", 32),
        )
        elapsed = time.time() - start
        if result.get("response"):
            reply = result["response"].strip()
            cost  = result.get("cost_entry", {})
            tok_in  = cost.get("tokens_in",  "?")
            tok_out = cost.get("tokens_out", "?")
            print(f"  ✅ PASS  ({elapsed:.1f}s)  reply='{reply}'  in={tok_in} out={tok_out}")
            return True
        else:
            exc = result.get("result")
            print(f"  ❌ FAIL  ({elapsed:.1f}s)  no response  exc={exc}")
            return False
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ❌ ERROR ({elapsed:.1f}s)  {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def main():
    print()
    print("=" * (WIDTH + 4))
    print("  Lodestar Live API Tests")
    print("=" * (WIDTH + 4))

    # Check required env vars
    missing = []
    for key in ("ANTHROPIC_API_KEY", "XAI_API_KEY", "PERPLEXITY_API_KEY"):
        if not os.environ.get(key):
            missing.append(key)
    if missing:
        print(f"\n⚠️  Missing env vars: {', '.join(missing)}")
        print("   Source ~/.env before running this script.\n")
        sys.exit(1)

    proxy = LodestarProxy()
    passed, failed = 0, 0

    for test in TESTS:
        ok = run_test(proxy, test)
        if ok:
            passed += 1
        else:
            failed += 1
        # Brief pause between Ollama calls to let the T600 unload/reload models
        # cleanly — prevents SIGKILL from dual-loading two large models.
        if test.get("tier", 3) < 3:
            time.sleep(3)

    print(f"  {'─' * WIDTH}")
    print(f"\n  Results: {passed} passed, {failed} failed out of {len(TESTS)} tests")
    print()

    # Print cost summary if available
    try:
        summary = proxy.cost_summary()
        if summary:
            print("  Cost summary (this run):")
            for k, v in summary.items():
                print(f"    {k}: {v}")
            print()
    except Exception:
        pass

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
