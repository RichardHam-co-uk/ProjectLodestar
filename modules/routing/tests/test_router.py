"""Tests for the SemanticRouter."""

import pytest
from modules.routing.router import SemanticRouter, DEFAULT_ROUTING_RULES, DEFAULT_FALLBACK_CHAINS


@pytest.fixture
def router_config():
    """Standard enabled router config."""
    return {
        "enabled": True,
        "routing_rules": DEFAULT_ROUTING_RULES,
        "fallback_chains": {
            "claude-sonnet": ["gpt-4o-mini", "gpt-3.5-turbo"],
            "gpt-3.5-turbo": ["local-llama"],
        },
    }


@pytest.fixture
def router(router_config):
    """A started SemanticRouter instance."""
    r = SemanticRouter(router_config)
    r.start()
    return r


class TestSemanticRouterLifecycle:
    """Tests for start/stop/health_check."""

    def test_start_enabled(self, router_config):
        r = SemanticRouter(router_config)
        r.start()
        assert r._started is True

    def test_start_disabled(self):
        r = SemanticRouter({"enabled": False})
        r.start()
        assert r._started is False

    def test_stop(self, router):
        router.stop()
        assert router._started is False

    def test_health_check_started(self, router):
        health = router.health_check()
        assert health["status"] == "healthy"
        assert health["rules_count"] == len(DEFAULT_ROUTING_RULES)

    def test_health_check_stopped(self, router_config):
        r = SemanticRouter(router_config)
        health = r.health_check()
        assert health["status"] == "down"


class TestTaskClassification:
    """Tests for classify_task keyword matching."""

    def test_classify_bug_fix(self, router):
        assert router.classify_task("fix the login bug") == "bug_fix"

    def test_classify_code_review(self, router):
        assert router.classify_task("review this pull request") == "code_review"

    def test_classify_architecture(self, router):
        assert router.classify_task("design the system architecture") == "architecture"

    def test_classify_documentation(self, router):
        assert router.classify_task("write a readme document") == "documentation"

    def test_classify_refactor(self, router):
        assert router.classify_task("refactor the database layer") == "refactor"

    def test_classify_code_generation(self, router):
        assert router.classify_task("create a new user model") == "code_generation"

    def test_classify_general_fallback(self, router):
        assert router.classify_task("hello world") == "general"

    def test_classify_case_insensitive(self, router):
        assert router.classify_task("FIX THE BUG") == "bug_fix"

    def test_classify_empty_prompt(self, router):
        assert router.classify_task("") == "general"

    def test_classify_whitespace_only(self, router):
        assert router.classify_task("   \n\t  ") == "general"

    def test_classify_multiple_keywords_highest_score_wins(self, router):
        # "fix bug crash debug error" has 5 bug_fix keywords
        # vs "create" has 1 code_generation keyword
        result = router.classify_task("fix the bug crash debug error and create something")
        assert result == "bug_fix"

    def test_classify_keyword_in_longer_word(self, router):
        # "fix" appears in "fixing", "scalab" appears in "scalability"
        assert router.classify_task("I'm fixing the broken code") == "bug_fix"

    def test_classify_mixed_case_keywords(self, router):
        assert router.classify_task("REVIEW the CODE quality") == "code_review"

    def test_classify_all_task_types_reachable(self, router):
        """Every configured task type should be classifiable."""
        prompts = {
            "bug_fix": "fix the crash bug",
            "code_review": "review this pull request",
            "architecture": "design system architecture diagram",
            "documentation": "write the readme document",
            "refactor": "refactor this function",
            "code_generation": "create a new module",
        }
        for expected_task, prompt in prompts.items():
            assert router.classify_task(prompt) == expected_task


class TestRouting:
    """Tests for the route() method."""

    def test_route_uses_classification(self, router):
        model = router.route("fix the crash bug")
        assert model == "gpt-3.5-turbo"

    def test_route_architecture_to_claude(self, router):
        model = router.route("design the system architecture")
        assert model == "claude-sonnet"

    def test_route_with_task_override(self, router):
        model = router.route("anything here", task_override="code_review")
        assert model == "claude-sonnet"

    def test_route_unknown_task_uses_general(self, router):
        model = router.route("something random", task_override="unknown_task")
        assert model == "gpt-3.5-turbo"

    def test_route_defaults_to_free_model(self, router):
        model = router.route("do something")
        assert model == "gpt-3.5-turbo"

    def test_route_unknown_override_falls_to_general(self, router):
        """Task override with unknown task should fall back to general model."""
        model = router.route("anything", task_override="imaginary_task")
        assert model == "gpt-3.5-turbo"

    def test_route_empty_prompt(self, router):
        model = router.route("")
        assert model == "gpt-3.5-turbo"  # general

    def test_route_code_review_to_claude(self, router):
        model = router.route("review the PR")
        assert model == "claude-sonnet"


class TestFallbackChains:
    """Tests for fallback chain retrieval."""

    def test_get_fallback_chain(self, router):
        chain = router.get_fallback_chain("claude-sonnet")
        assert chain == ["gpt-4o-mini", "gpt-3.5-turbo"]

    def test_get_fallback_chain_missing(self, router):
        chain = router.get_fallback_chain("nonexistent-model")
        assert chain == []

    def test_fallback_chain_preserves_order(self, router):
        chain = router.get_fallback_chain("claude-sonnet")
        assert chain[0] == "gpt-4o-mini"
        assert chain[1] == "gpt-3.5-turbo"

    def test_fallback_chain_free_model(self, router):
        chain = router.get_fallback_chain("gpt-3.5-turbo")
        assert chain == ["local-llama"]


class TestRouterConfig:
    """Tests for router configuration edge cases."""

    def test_custom_routing_rules(self):
        r = SemanticRouter({
            "enabled": True,
            "routing_rules": {"custom_task": "custom-model", "general": "fallback-model"},
        })
        assert r.route("anything", task_override="custom_task") == "custom-model"

    def test_empty_fallback_chains(self):
        r = SemanticRouter({"enabled": True, "fallback_chains": {}})
        assert r.get_fallback_chain("any-model") == []

    def test_default_config_uses_defaults(self):
        r = SemanticRouter({"enabled": True})
        assert r.routing_rules == DEFAULT_ROUTING_RULES
        assert r.fallback_chains == DEFAULT_FALLBACK_CHAINS

    def test_health_check_counts(self):
        r = SemanticRouter({
            "enabled": True,
            "routing_rules": {"a": "m1", "b": "m2"},
            "fallback_chains": {"m1": ["m2"]},
        })
        health = r.health_check()
        assert health["rules_count"] == 2
        assert health["fallback_chains_count"] == 1
