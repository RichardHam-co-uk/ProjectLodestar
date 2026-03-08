"""Semantic router for task-based model selection.

Classifies incoming requests by task type and routes them to the
most appropriate model based on configurable rules and fallback chains.
"""

from typing import Any, Dict, List, Optional
import logging

from modules.base import LodestarPlugin

logger = logging.getLogger(__name__)

# Default task-to-model mapping (three-tier: local fast → local reasoning → premium)
#
#  Tier 1 (free, fast)  — gpt-3.5-turbo alias → deepseek-coder:6.7b on T600 Ollama
#  Tier 2 (free, smart) — local-reasoning alias → deepseek-r1:7b on T600 Ollama
#  Tier 3 (paid)        — claude-sonnet alias → Anthropic API
#
DEFAULT_ROUTING_RULES: Dict[str, str] = {
    # Tier 1: fast local code model
    "code_generation": "gpt-3.5-turbo",
    "bug_fix":         "gpt-3.5-turbo",
    "documentation":   "gpt-3.5-turbo",
    "general":         "gpt-3.5-turbo",
    # Tier 2: local reasoning model (more capable, still free)
    "refactor":        "local-reasoning",
    "debug_analysis":  "local-reasoning",
    # Tier 3: premium (complex multi-step reasoning, security, architecture)
    "code_review":     "claude-sonnet",
    "architecture":    "claude-sonnet",
    "security_audit":  "claude-sonnet",
}

# Fallback chains: if primary model fails, try these in order
DEFAULT_FALLBACK_CHAINS: Dict[str, List[str]] = {
    "gpt-3.5-turbo":   ["local-reasoning", "claude-sonnet"],
    "local-reasoning":  ["claude-sonnet"],
    "local-llama":      ["local-reasoning", "claude-sonnet"],
}


class SemanticRouter(LodestarPlugin):
    """Routes requests to models based on task classification.

    Uses keyword-based task classification to select the best model
    for each request, with configurable rules and fallback chains.

    Args:
        config: Router configuration with routing_rules and fallback_chains.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.routing_rules: Dict[str, str] = config.get(
            "routing_rules", DEFAULT_ROUTING_RULES
        )
        self.fallback_chains: Dict[str, List[str]] = config.get(
            "fallback_chains", DEFAULT_FALLBACK_CHAINS
        )
        self._started = False

    def start(self) -> None:
        """Start the routing module."""
        if not self.enabled:
            logger.info("Routing module disabled, skipping start")
            return
        self._started = True
        logger.info(
            "Semantic router started with %d rules", len(self.routing_rules)
        )

    def stop(self) -> None:
        """Stop the routing module."""
        self._started = False
        logger.info("Semantic router stopped")

    def health_check(self) -> Dict[str, Any]:
        """Return router health status."""
        return {
            "status": "healthy" if self._started else "down",
            "enabled": self.enabled,
            "rules_count": len(self.routing_rules),
            "fallback_chains_count": len(self.fallback_chains),
        }

    def classify_task(self, prompt: str) -> str:
        """Classify a prompt into a task type using keyword matching.

        Args:
            prompt: The user's input prompt.

        Returns:
            Task type string (e.g. 'code_generation', 'bug_fix').
        """
        prompt_lower = prompt.lower()

        keywords: Dict[str, List[str]] = {
            # Tier 3 — premium (checked first; these override cheaper tiers)
            "security_audit": ["security", "vulnerabilit", "cve", "exploit", "pentest", "threat"],
            "architecture": [
                "architect", "design", "structure", "pattern", "system",
                "scalab", "diagram", "microservice",
            ],
            "code_review": ["review", "audit", "inspect", "quality"],
            # Tier 2 — local reasoning
            "debug_analysis": ["debug", "trace", "diagnose", "root cause", "analyse", "analyze"],
            "refactor": ["refactor", "clean", "simplify", "reorganize", "improve", "optimise", "optimize"],
            # Tier 1 — fast local
            "bug_fix": ["fix", "bug", "error", "broken", "crash", "issue"],
            "documentation": ["document", "readme", "comment", "docstring", "explain"],
            "code_generation": [
                "create", "build", "implement", "add", "write", "generate", "make",
            ],
        }

        best_task = "general"
        best_score = 0

        for task, task_keywords in keywords.items():
            score = sum(1 for kw in task_keywords if kw in prompt_lower)
            if score > best_score:
                best_score = score
                best_task = task

        return best_task

    def route(self, prompt: str, task_override: Optional[str] = None) -> str:
        """Select the best model for a given prompt.

        Args:
            prompt: The user's input prompt.
            task_override: Optional explicit task type, bypassing classification.

        Returns:
            Model alias string (e.g. 'gpt-3.5-turbo', 'claude-sonnet').
        """
        task = task_override if task_override else self.classify_task(prompt)
        model = self.routing_rules.get(task, self.routing_rules.get("general", "gpt-3.5-turbo"))

        logger.debug("Routed task '%s' to model '%s'", task, model)
        return model

    def get_fallback_chain(self, model: str) -> List[str]:
        """Get the fallback chain for a model.

        Args:
            model: Primary model alias.

        Returns:
            List of fallback model aliases, or empty list if none configured.
        """
        return self.fallback_chains.get(model, [])
