"""CLI cost report for Lodestar.

Reads directly from the SQLite cost database and prints a summary.
No running Lodestar instance required.

Usage
-----
From the repo root with the venv active::

    # All-time summary
    python -m modules.costs

    # This month only
    python -m modules.costs --period month

    # This week
    python -m modules.costs --period week

    # Today
    python -m modules.costs --period today

    # Per-model breakdown
    python -m modules.costs --by-model

    # Custom database path
    python -m modules.costs --db /path/to/costs.db
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.costs.storage import CostStorage

DEFAULT_DB = Path.home() / ".lodestar" / "costs.db"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_usd(value: float) -> str:
    """Format a USD value — pence-level precision for small amounts."""
    if value == 0.0:
        return "$0.00"
    if value < 0.01:
        return f"${value:.6f}"
    return f"${value:.4f}"


def _fmt_tokens(n: int) -> str:
    """Format a token count with thousands separator."""
    return f"{n:,}"


def _period_bounds(period: str) -> tuple[datetime, datetime]:
    """Return (start, end) datetimes for a named period."""
    now = datetime.now()
    if period == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "week":
        start = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    elif period == "month":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        raise ValueError(f"Unknown period: {period!r}")
    return start, now


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a list of cost records into a summary dict."""
    total_cost = 0.0
    total_savings = 0.0
    total_tokens_in = 0
    total_tokens_out = 0
    by_model: Dict[str, Dict[str, Any]] = {}

    for r in records:
        total_cost += r["cost_usd"]
        total_savings += r["savings_usd"]
        total_tokens_in += r["tokens_in"]
        total_tokens_out += r["tokens_out"]

        model = r["model"]
        if model not in by_model:
            by_model[model] = {
                "requests": 0,
                "cost": 0.0,
                "tokens_in": 0,
                "tokens_out": 0,
            }
        by_model[model]["requests"] += 1
        by_model[model]["cost"] += r["cost_usd"]
        by_model[model]["tokens_in"] += r["tokens_in"]
        by_model[model]["tokens_out"] += r["tokens_out"]

    baseline = total_cost + total_savings
    pct = (total_savings / baseline * 100) if baseline > 0 else 0.0

    return {
        "total_requests": len(records),
        "total_cost": total_cost,
        "total_savings": total_savings,
        "savings_pct": pct,
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "by_model": by_model,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render(
    summary: Dict[str, Any],
    period_label: str,
    show_by_model: bool,
) -> str:
    """Render a cost summary to a string for terminal output."""
    lines: List[str] = []
    sep = "─" * 44

    lines.append("")
    lines.append("  Lodestar Cost Report")
    lines.append(f"  Period : {period_label}")
    lines.append(f"  {sep}")

    if summary["total_requests"] == 0:
        lines.append("  No records found for this period.")
        lines.append("")
        return "\n".join(lines)

    lines.append(
        f"  Requests    : {summary['total_requests']}"
    )
    lines.append(
        f"  Tokens in   : {_fmt_tokens(summary['total_tokens_in'])}"
    )
    lines.append(
        f"  Tokens out  : {_fmt_tokens(summary['total_tokens_out'])}"
    )
    lines.append(
        f"  Total tokens: {_fmt_tokens(summary['total_tokens_in'] + summary['total_tokens_out'])}"
    )
    lines.append(f"  {sep}")
    lines.append(
        f"  Actual cost : {_fmt_usd(summary['total_cost'])}"
    )
    lines.append(
        f"  Saved vs all-claude-sonnet: {_fmt_usd(summary['total_savings'])}  "
        f"({summary['savings_pct']:.1f}%)"
    )

    if show_by_model and summary["by_model"]:
        lines.append(f"  {sep}")
        lines.append("  By model:")
        # Sort by cost descending
        for model, data in sorted(
            summary["by_model"].items(), key=lambda x: x[1]["cost"], reverse=True
        ):
            total_tok = data["tokens_in"] + data["tokens_out"]
            lines.append(
                f"    {model:<20s}  "
                f"{data['requests']:>4} reqs  "
                f"{_fmt_tokens(total_tok):>10} tok  "
                f"{_fmt_usd(data['cost'])}"
            )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m modules.costs",
        description="Lodestar cost report — reads from local SQLite database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m modules.costs\n"
            "  python -m modules.costs --period month --by-model\n"
            "  python -m modules.costs --period today\n"
        ),
    )
    p.add_argument(
        "--period",
        choices=["today", "week", "month", "all"],
        default="all",
        help="Time period to report on (default: all)",
    )
    p.add_argument(
        "--by-model",
        action="store_true",
        default=False,
        help="Show per-model breakdown",
    )
    p.add_argument(
        "--db",
        default=str(DEFAULT_DB),
        metavar="PATH",
        help=f"Path to SQLite database (default: {DEFAULT_DB})",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    db_path = Path(args.db).expanduser()

    if not db_path.exists():
        print(
            f"\n  No cost database found at {db_path}\n"
            "  Lodestar hasn't recorded any requests yet.\n",
            file=sys.stderr,
        )
        return 1

    storage = CostStorage(str(db_path))
    try:
        storage.connect()

        if args.period == "all":
            records = storage.query_all()
            period_label = "all time"
        else:
            start, end = _period_bounds(args.period)
            records = storage.query_by_date_range(start, end)
            if args.period == "today":
                period_label = f"today ({start.strftime('%Y-%m-%d')})"
            elif args.period == "week":
                period_label = f"this week (since {start.strftime('%Y-%m-%d')})"
            else:
                period_label = f"this month ({start.strftime('%B %Y')})"

        summary = _aggregate(records)
        print(_render(summary, period_label, show_by_model=args.by_model))

    finally:
        storage.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
