"""LiteLLM proxy integration layer.

Bridges the SemanticRouter with LiteLLM's proxy server by providing
custom callbacks that intercept requests, classify tasks, and record
costs — all without modifying LiteLLM's core behaviour.
"""

from typing import Any, Dict, List, Optional
import logging
import yaml
from pathlib import Path

from modules.base import EventBus
from modules.routing.router import SemanticRouter
from modules.routing.fallback import FallbackExecutor, RequestResult
from modules.costs.tracker import CostTracker
from modules.health.checker import HealthChecker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model parameter registry
# Maps each model alias to the litellm kwargs needed to call it.
# Loaded at import time from config/litellm_config.yaml where available;
# otherwise uses these built-in defaults so the module works standalone.
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_PARAMS: Dict[str, Dict[str, Any]] = {
    "gpt-3.5-turbo": {
        "model": "openai/deepseek-coder:6.7b",
        "api_base": "http://192.168.120.211:11434/v1",
        "api_key": "ollama",
    },
    "local-llama": {
        "model": "openai/llama3.2:1b",
        "api_base": "http://192.168.120.211:11434/v1",
        "api_key": "ollama",
    },
    "local-reasoning": {
        "model": "openai/deepseek-r1:7b",
        "api_base": "http://192.168.120.211:11434/v1",
        "api_key": "ollama",
    },
    "claude-sonnet": {
        "model": "anthropic/claude-sonnet-4-5-20250929",
        # api_key resolved at call time from os.environ["ANTHROPIC_API_KEY"]
    },
    "claude-opus": {
        "model": "anthropic/claude-opus-4-5-20251101",
    },
    "gpt-4o": {
        "model": "openai/gpt-4o",
    },
    "gpt-4o-mini": {
        "model": "openai/gpt-4o-mini",
    },
    "grok-beta": {
        "model": "xai/grok-beta",
    },
    "gemini-pro": {
        "model": "gemini-1.5-flash",
    },
}


class LodestarProxy:
    """Orchestrates routing, fallback, and cost tracking for LLM requests.

    This is the main integration point between Lodestar's modules and
    the LiteLLM router. It reads module configs, initialises the router
    and cost tracker, and provides a single `handle_request` method that
    does classification -> routing -> fallback -> cost recording.

    Args:
        config_dir: Path to the config/ directory containing modules.yaml
                    and per-module configs.
        event_bus: Optional shared EventBus instance.
    """

    def __init__(
        self,
        config_dir: str = "config",
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self.config_dir = Path(config_dir)
        self.event_bus = event_bus or EventBus()
        self._load_configs()

        self.router = SemanticRouter(self._routing_config)
        self.cost_tracker = CostTracker(self._costs_config)
        self.health_checker = HealthChecker(self._health_config, self.event_bus)
        self.fallback_executor = FallbackExecutor()
        
        # Initialize Cache
        from modules.routing.cache import CacheManager
        self.cache = CacheManager()

    def _load_configs(self) -> None:
        """Load module configurations from YAML files."""
        modules_yaml = self.config_dir / "modules.yaml"
        if modules_yaml.exists():
            with open(modules_yaml) as f:
                self._modules_config = yaml.safe_load(f) or {}
        else:
            self._modules_config = {}

        routing_yaml = self.config_dir.parent / "modules" / "routing" / "config.yaml"
        if routing_yaml.exists():
            with open(routing_yaml) as f:
                raw = yaml.safe_load(f) or {}
                self._routing_config = raw.get("routing", {"enabled": True})
        else:
            self._routing_config = {"enabled": True}

        costs_yaml = self.config_dir.parent / "modules" / "costs" / "config.yaml"
        if costs_yaml.exists():
            with open(costs_yaml) as f:
                raw = yaml.safe_load(f) or {}
                self._costs_config = raw.get("costs", {"enabled": True})
        else:
            self._costs_config = {"enabled": True}

        health_yaml = self.config_dir.parent / "modules" / "health" / "config.yaml"
        if health_yaml.exists():
            with open(health_yaml) as f:
                raw = yaml.safe_load(f) or {}
                self._health_config = raw.get("health", {"enabled": True})
        else:
            self._health_config = {"enabled": True}

    def start(self) -> None:
        """Start all modules."""
        self.router.start()
        self.cost_tracker.start()
        self.health_checker.start()
        logger.info("LodestarProxy started")

    def stop(self) -> None:
        """Stop all modules gracefully."""
        self.router.stop()
        self.cost_tracker.stop()
        self.health_checker.stop()
        logger.info("LodestarProxy stopped")

    def handle_request(
        self,
        prompt: str,
        request_fn: Any = None,
        task_override: Optional[str] = None,
        model_override: Optional[str] = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> Dict[str, Any]:
        """Process an LLM request through the full pipeline.

        1. Classify the task (or use override)
        2. Route to best model (or use override)
        3. Execute with fallback chain
        4. Record cost
        5. Publish event

        Args:
            prompt: The user's input prompt.
            request_fn: Callable(model) -> response. If None, returns
                        routing decision only (dry-run mode).
            task_override: Force a specific task classification.
            model_override: Force a specific model (bypasses routing).
            tokens_in: Input token count (from LiteLLM callback).
            tokens_out: Output token count (from LiteLLM callback).

        Returns:
            Dict with task, model, result, cost_entry keys.
        """
        # Step 1: Classify
        task = task_override or self.router.classify_task(prompt)

        # Step 2: Route
        model = model_override or self.router.route(prompt, task_override=task)

        # Step 2.5: Check Cache (after routing so we have the model for the key)
        if not task_override and not model_override:
            cached_response = self.cache.get(model=model, messages=[{"role": "user", "content": prompt}])
            if cached_response:
                logger.info("Serving from cache")
                return cached_response

        # Step 3: Execute (or dry-run)
        if request_fn is not None:
            fallback_chain = self.router.get_fallback_chain(model)
            result = self.fallback_executor.execute(
                model, fallback_chain, request_fn
            )
            actual_model = result.model if result.success else model
        else:
            result = RequestResult(success=True, model=model, response=None)
            actual_model = model

        # Step 4: Record cost
        cost_entry = self.cost_tracker.record(
            model=actual_model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            task=task,
        )

        # Step 5: Publish event
        event_data = {
            "task": task,
            "model": actual_model,
            "success": result.success,
            "cost": cost_entry["cost"],
            "savings": cost_entry["savings"],
        }
        self.event_bus.publish("request_completed", event_data)

        result_dict = {
            "task": task,
            "model": actual_model,
            "result": result,
            "cost_entry": cost_entry,
        }
        
        # Step 6: Cache success (store serializable data only)
        if result.success and not task_override and not model_override:
            cacheable_data = {
                "task": task,
                "model": actual_model,
                "cost_entry": cost_entry,
                # Don't cache the full RequestResult, just the essential info
                "success": True,
            }
            self.cache.set(
                model=actual_model,
                messages=[{"role": "user", "content": prompt}],
                response=cacheable_data
            )
            
        return result_dict

    def _build_model_params(self) -> Dict[str, Dict[str, Any]]:
        """Load model parameters from litellm_config.yaml, falling back to defaults.

        Returns:
            Dict mapping alias → litellm kwargs (model, api_base, api_key, …).
        """
        litellm_cfg = self.config_dir / "litellm_config.yaml"
        if not litellm_cfg.exists():
            return dict(_DEFAULT_MODEL_PARAMS)

        try:
            with open(litellm_cfg) as f:
                raw = yaml.safe_load(f) or {}
            params: Dict[str, Dict[str, Any]] = {}
            for entry in raw.get("model_list", []):
                alias = entry.get("model_name")
                lp = entry.get("litellm_params", {})
                if alias and lp.get("model"):
                    params[alias] = {
                        k: v for k, v in lp.items() if k != "model_name"
                    }
            return params or dict(_DEFAULT_MODEL_PARAMS)
        except Exception:
            logger.warning("Could not parse litellm_config.yaml; using defaults")
            return dict(_DEFAULT_MODEL_PARAMS)

    def complete(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        task_override: Optional[str] = None,
        model_override: Optional[str] = None,
        max_tokens: int = 1024,
        **litellm_kwargs: Any,
    ) -> Dict[str, Any]:
        """Route a prompt and execute it via litellm, recording cost.

        This is the primary entry point for all LLM calls from Lodestar.
        It combines routing, execution with fallback, and cost tracking
        into a single call.

        Args:
            prompt: The user's prompt (used for task classification).
            messages: Full messages list. Defaults to [{"role":"user","content":prompt}].
            task_override: Force a specific task classification.
            model_override: Force a specific model alias.
            max_tokens: Maximum tokens in the response.
            **litellm_kwargs: Additional kwargs forwarded to litellm.completion().

        Returns:
            Dict with keys: task, model, response, cost_entry, savings.
        """
        import litellm
        import os

        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        model_params = self._build_model_params()

        def _request_fn(alias: str) -> Any:
            """Call litellm with the params for the given model alias."""
            params = model_params.get(alias, {})
            if not params:
                raise ValueError(f"Unknown model alias: {alias!r}")

            call_kwargs: Dict[str, Any] = {
                "model":      params["model"],
                "messages":   messages,
                "max_tokens": max_tokens,
            }
            if "api_base" in params:
                call_kwargs["api_base"] = params["api_base"]
            if "api_key" in params:
                call_kwargs["api_key"] = params["api_key"]
            call_kwargs.update(litellm_kwargs)

            resp = litellm.completion(**call_kwargs)
            return resp

        # Run through handle_request (routing → fallback → cost recording)
        pipeline = self.handle_request(
            prompt=prompt,
            request_fn=_request_fn,
            task_override=task_override,
            model_override=model_override,
        )

        result = pipeline["result"]
        if result.success and result.response is not None:
            resp = result.response
            usage = resp.usage if hasattr(resp, "usage") else None
            if usage:
                # Re-record with actual token counts (handle_request used 0)
                pipeline["cost_entry"] = self.cost_tracker.record(
                    model=pipeline["model"],
                    tokens_in=usage.prompt_tokens,
                    tokens_out=usage.completion_tokens,
                    task=pipeline["task"],
                )
            pipeline["response"] = resp.choices[0].message.content
        else:
            pipeline["response"] = None

        return pipeline

    def cost_summary(self) -> Dict[str, Any]:
        """Return the current cost summary from the tracker."""
        return self.cost_tracker.summary()

    def health_check(self) -> Dict[str, Any]:
        """Return health status of all modules."""
        return {
            "proxy": "healthy",
            "router": self.router.health_check(),
            "cost_tracker": self.cost_tracker.health_check(),
            "health_checker": self.health_checker.health_check(),
        }
