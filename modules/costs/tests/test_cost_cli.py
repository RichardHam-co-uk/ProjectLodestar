"""Tests for modules/costs/__main__.py (cost report CLI)."""

import json
from pathlib import Path

import pytest

from modules.costs.__main__ import (
    _aggregate,
    _fmt_usd,
    _fmt_tokens,
    _period_bounds,
    _render,
    main,
)


# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------

def _make_record(
    model="claude-sonnet",
    tokens_in=100,
    tokens_out=50,
    cost_usd=0.001,
    savings_usd=0.002,
    task="code_generation",
):
    """Build a record dict using _aggregate-compatible keys (from query results)."""
    return {
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": cost_usd,
        "baseline_cost_usd": cost_usd + savings_usd,
        "savings_usd": savings_usd,
        "task": task,
    }


def _make_storage_record(
    model="claude-sonnet",
    tokens_in=100,
    tokens_out=50,
    cost_usd=0.001,
    savings_usd=0.002,
    task="code_generation",
):
    """Build a record dict matching CostStorage.insert() key expectations."""
    return {
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost": cost_usd,
        "baseline_cost": cost_usd + savings_usd,
        "savings": savings_usd,
        "task": task,
    }


# ---------------------------------------------------------------------------
# _fmt_usd
# ---------------------------------------------------------------------------

class TestFmtUsd:
    def test_zero(self):
        assert _fmt_usd(0.0) == "$0.00"

    def test_normal_amount(self):
        assert _fmt_usd(1.2345) == "$1.2345"

    def test_small_amount_uses_high_precision(self):
        result = _fmt_usd(0.000001)
        assert result.startswith("$")
        assert "0.000001" in result

    def test_rounds_to_four_decimal_places(self):
        assert _fmt_usd(0.1) == "$0.1000"


# ---------------------------------------------------------------------------
# _fmt_tokens
# ---------------------------------------------------------------------------

class TestFmtTokens:
    def test_small_number(self):
        assert _fmt_tokens(100) == "100"

    def test_thousands_separator(self):
        assert _fmt_tokens(1000) == "1,000"

    def test_large_number(self):
        assert _fmt_tokens(1_234_567) == "1,234,567"


# ---------------------------------------------------------------------------
# _period_bounds
# ---------------------------------------------------------------------------

class TestPeriodBounds:
    def test_today_starts_at_midnight(self):
        start, end = _period_bounds("today")
        assert start.hour == 0
        assert start.minute == 0
        assert start.second == 0
        assert end > start

    def test_week_starts_on_monday(self):
        start, _ = _period_bounds("week")
        assert start.weekday() == 0  # Monday

    def test_month_starts_on_first(self):
        start, _ = _period_bounds("month")
        assert start.day == 1
        assert start.hour == 0

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError, match="Unknown period"):
            _period_bounds("forever")

    def test_end_is_after_start(self):
        for period in ("today", "week", "month"):
            start, end = _period_bounds(period)
            assert end > start


# ---------------------------------------------------------------------------
# _aggregate
# ---------------------------------------------------------------------------

class TestAggregate:
    def test_empty_records(self):
        result = _aggregate([])
        assert result["total_requests"] == 0
        assert result["total_cost"] == 0.0
        assert result["total_savings"] == 0.0
        assert result["savings_pct"] == 0.0
        assert result["by_model"] == {}

    def test_single_record(self):
        records = [_make_record(cost_usd=0.01, savings_usd=0.02)]
        result = _aggregate(records)
        assert result["total_requests"] == 1
        assert result["total_cost"] == pytest.approx(0.01)
        assert result["total_savings"] == pytest.approx(0.02)

    def test_token_totals(self):
        records = [
            _make_record(tokens_in=100, tokens_out=50),
            _make_record(tokens_in=200, tokens_out=75),
        ]
        result = _aggregate(records)
        assert result["total_tokens_in"] == 300
        assert result["total_tokens_out"] == 125

    def test_by_model_groups_correctly(self):
        records = [
            _make_record(model="local-code", cost_usd=0.0),
            _make_record(model="local-code", cost_usd=0.0),
            _make_record(model="claude-sonnet", cost_usd=0.01),
        ]
        result = _aggregate(records)
        assert result["by_model"]["local-code"]["requests"] == 2
        assert result["by_model"]["claude-sonnet"]["requests"] == 1

    def test_savings_percentage(self):
        # cost=1, savings=3 → baseline=4 → 75% savings
        records = [_make_record(cost_usd=1.0, savings_usd=3.0)]
        result = _aggregate(records)
        assert result["savings_pct"] == pytest.approx(75.0)

    def test_zero_baseline_gives_zero_pct(self):
        records = [_make_record(cost_usd=0.0, savings_usd=0.0)]
        result = _aggregate(records)
        assert result["savings_pct"] == 0.0


# ---------------------------------------------------------------------------
# _render
# ---------------------------------------------------------------------------

class TestRender:
    def test_no_records_message(self):
        summary = _aggregate([])
        output = _render(summary, "all time", show_by_model=False)
        assert "No records found" in output

    def test_contains_period_label(self):
        summary = _aggregate([_make_record()])
        output = _render(summary, "this month (March 2026)", show_by_model=False)
        assert "March 2026" in output

    def test_contains_token_counts(self):
        records = [_make_record(tokens_in=1000, tokens_out=500)]
        summary = _aggregate(records)
        output = _render(summary, "all time", show_by_model=False)
        assert "1,000" in output
        assert "500" in output

    def test_contains_cost(self):
        records = [_make_record(cost_usd=0.0042)]
        summary = _aggregate(records)
        output = _render(summary, "all time", show_by_model=False)
        assert "0.0042" in output

    def test_by_model_hidden_when_not_requested(self):
        records = [_make_record(model="local-code")]
        summary = _aggregate(records)
        output = _render(summary, "all time", show_by_model=False)
        assert "local-code" not in output

    def test_by_model_shown_when_requested(self):
        records = [_make_record(model="local-code")]
        summary = _aggregate(records)
        output = _render(summary, "all time", show_by_model=True)
        assert "local-code" in output

    def test_by_model_sorted_by_cost_descending(self):
        records = [
            _make_record(model="cheap-model", cost_usd=0.001),
            _make_record(model="expensive-model", cost_usd=0.1),
        ]
        summary = _aggregate(records)
        output = _render(summary, "all time", show_by_model=True)
        assert output.index("expensive-model") < output.index("cheap-model")


# ---------------------------------------------------------------------------
# main() — CLI entry point
# ---------------------------------------------------------------------------

class TestMain:
    def test_missing_db_returns_1(self, tmp_path):
        rc = main(["--db", str(tmp_path / "nonexistent.db")])
        assert rc == 1

    def test_empty_db_returns_0(self, tmp_path):
        from modules.costs.storage import CostStorage
        db = tmp_path / "costs.db"
        s = CostStorage(str(db))
        s.connect()
        s.close()
        rc = main(["--db", str(db)])
        assert rc == 0

    def test_all_period_default(self, tmp_path):
        from modules.costs.storage import CostStorage
        db = tmp_path / "costs.db"
        s = CostStorage(str(db))
        s.connect()
        s.insert(_make_storage_record(cost_usd=0.01, savings_usd=0.02))
        s.close()
        rc = main(["--db", str(db)])
        assert rc == 0

    def test_period_today(self, tmp_path):
        from modules.costs.storage import CostStorage
        db = tmp_path / "costs.db"
        s = CostStorage(str(db))
        s.connect()
        s.insert(_make_storage_record())
        s.close()
        rc = main(["--db", str(db), "--period", "today"])
        assert rc == 0

    def test_period_week(self, tmp_path):
        from modules.costs.storage import CostStorage
        db = tmp_path / "costs.db"
        s = CostStorage(str(db))
        s.connect()
        s.insert(_make_storage_record())
        s.close()
        rc = main(["--db", str(db), "--period", "week"])
        assert rc == 0

    def test_period_month(self, tmp_path):
        from modules.costs.storage import CostStorage
        db = tmp_path / "costs.db"
        s = CostStorage(str(db))
        s.connect()
        s.insert(_make_storage_record())
        s.close()
        rc = main(["--db", str(db), "--period", "month"])
        assert rc == 0

    def test_by_model_flag(self, tmp_path):
        from modules.costs.storage import CostStorage
        db = tmp_path / "costs.db"
        s = CostStorage(str(db))
        s.connect()
        s.insert(_make_storage_record(model="local-code", cost_usd=0.0, savings_usd=0.0))
        s.close()
        rc = main(["--db", str(db), "--by-model"])
        assert rc == 0

    def test_parser_defaults(self):
        from modules.costs.__main__ import _build_parser
        args = _build_parser().parse_args([])
        assert args.period == "all"
        assert args.by_model is False

    def test_parser_period_choices(self):
        from modules.costs.__main__ import _build_parser
        for period in ("today", "week", "month", "all"):
            args = _build_parser().parse_args(["--period", period])
            assert args.period == period
