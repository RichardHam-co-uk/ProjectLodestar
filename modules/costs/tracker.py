"""Cost tracking for LLM requests.

Records token usage and cost per request, persists to SQLite,
and provides summary reporting and budget alert capabilities.
"""

from typing import Any, Dict, Optional
import logging

from modules.base import LodestarPlugin
from modules.costs.storage import CostStorage

logger = logging.getLogger(__name__)

# Cost per 1M tokens (input/output) for each model alias
MODEL_COSTS: Dict[str, Dict[str, float]] = {
    # Tier 1 — free local models (T600 Ollama)
    "gpt-3.5-turbo":   {"input": 0.0,   "output": 0.0},
    "local-llama":     {"input": 0.0,   "output": 0.0},
    "local-reasoning": {"input": 0.0,   "output": 0.0},
    # Tier 3 — paid cloud models
    "claude-sonnet":   {"input": 3.0,   "output": 15.0},
    "claude-opus":     {"input": 15.0,  "output": 75.0},
    "gpt-4o-mini":     {"input": 0.15,  "output": 0.60},
    "gpt-4o":          {"input": 2.50,  "output": 10.0},
    # xAI Grok 3 pricing (per million tokens, as of 2026-03 — verify at console.x.ai)
    "grok-beta":            {"input": 5.0,   "output": 25.0},   # grok-3-fast
    "grok-3-mini":          {"input": 0.30,  "output": 0.50},   # grok-3-mini
    "gemini-pro":           {"input": 0.075, "output": 0.30},
    # Perplexity — sonar pricing (per million tokens, as of 2026-03)
    "perplexity-sonar":     {"input": 1.0,   "output": 1.0},
    "perplexity-sonar-pro": {"input": 3.0,   "output": 15.0},
}

# Baseline model for savings calculation (what you'd pay without Lodestar)
BASELINE_MODEL = "claude-sonnet"


class CostTracker(LodestarPlugin):
    """Tracks LLM request costs and calculates savings.

    Maintains an in-memory ledger of all tracked requests. Persistence
    to SQLite will be added via the storage module.

    Args:
        config: Tracker configuration with budget_limit, baseline_model, etc.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.model_costs: Dict[str, Dict[str, float]] = config.get(
            "model_costs", MODEL_COSTS
        )
        self.baseline_model: str = config.get("baseline_model", BASELINE_MODEL)
        self.budget_limit: Optional[float] = config.get("budget_limit")
        self._records: list = []
        self._started = False
        self._storage: Optional[CostStorage] = None

        db_path = config.get("database_path")
        if db_path:
            self._storage = CostStorage(db_path)

    def start(self) -> None:
        """Start the cost tracker and connect to storage if configured."""
        if self._started:
            return  # idempotent — safe to call multiple times
        if not self.enabled:
            logger.info("Cost tracker disabled, skipping start")
            return
        if self._storage:
            self._storage.connect()
            logger.info("Cost storage connected at %s", self._storage.db_path)
        self._started = True
        logger.info("Cost tracker started")

    def stop(self) -> None:
        """Stop the cost tracker and close storage."""
        if self._storage:
            self._storage.close()
        self._started = False
        logger.info("Cost tracker stopped")

    def health_check(self) -> Dict[str, Any]:
        """Return tracker health status."""
        return {
            "status": "healthy" if self._started else "down",
            "enabled": self.enabled,
            "records_count": len(self._records),
            "total_cost": self.total_cost(),
        }

    def calculate_cost(
        self, model: str, tokens_in: int, tokens_out: int
    ) -> float:
        """Calculate the cost of a single request.

        Args:
            model: Model alias used for the request.
            tokens_in: Number of input tokens.
            tokens_out: Number of output tokens.

        Returns:
            Cost in USD.
        """
        costs = self.model_costs.get(model, {"input": 0.0, "output": 0.0})
        cost = (tokens_in * costs["input"] + tokens_out * costs["output"]) / 1_000_000
        return round(cost, 6)

    def record(
        self, model: str, tokens_in: int, tokens_out: int, task: str = ""
    ) -> Dict[str, Any]:
        """Record a completed request.

        Args:
            model: Model alias used.
            tokens_in: Input token count.
            tokens_out: Output token count.
            task: Optional task classification label.

        Returns:
            The recorded entry dict.
        """
        cost = self.calculate_cost(model, tokens_in, tokens_out)
        baseline_cost = self.calculate_cost(
            self.baseline_model, tokens_in, tokens_out
        )
        entry = {
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost": cost,
            "baseline_cost": baseline_cost,
            "savings": round(baseline_cost - cost, 6),
            "task": task,
        }
        self._records.append(entry)
        if self._storage and self._storage._conn:
            try:
                self._storage.insert(entry)
            except Exception:
                logger.exception("Failed to persist cost record to storage")
        return entry

    def total_cost(self) -> float:
        """Total actual cost across all recorded requests."""
        return round(sum(r["cost"] for r in self._records), 6)

    def total_savings(self) -> float:
        """Total savings vs baseline across all recorded requests."""
        return round(sum(r["savings"] for r in self._records), 6)

    def savings_percentage(self) -> float:
        """Savings as a percentage of baseline cost.

        Returns:
            Percentage (0-100), or 0.0 if no baseline cost.
        """
        baseline = sum(r["baseline_cost"] for r in self._records)
        if baseline == 0:
            return 0.0
        return round((1 - self.total_cost() / baseline) * 100, 1)

    def is_over_budget(self) -> bool:
        """Check if total cost exceeds the configured budget limit.

        Returns:
            True if over budget, False if under or no budget set.
        """
        if self.budget_limit is None:
            return False
        return self.total_cost() > self.budget_limit

    def summary(self) -> Dict[str, Any]:
        """Generate a cost summary report.

        Returns:
            Dict with total_cost, total_savings, savings_pct, per-model breakdown.
        """
        by_model: Dict[str, Dict[str, Any]] = {}
        for r in self._records:
            model = r["model"]
            if model not in by_model:
                by_model[model] = {"requests": 0, "cost": 0.0, "tokens": 0}
            by_model[model]["requests"] += 1
            by_model[model]["cost"] += r["cost"]
            by_model[model]["tokens"] += r["tokens_in"] + r["tokens_out"]

        return {
            "total_cost": self.total_cost(),
            "total_savings": self.total_savings(),
            "savings_percentage": self.savings_percentage(),
            "total_requests": len(self._records),
            "over_budget": self.is_over_budget(),
            "by_model": by_model,
        }
