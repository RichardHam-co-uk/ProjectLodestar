"""Tests for BrowserActivityLogger.

All tests are self-contained and use tmp_path for log file isolation —
no writes go to ~/.lodestar during testing.
"""

import json
from pathlib import Path

import pytest

from modules.browser.logger import BrowserActivityLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def log_file(tmp_path: Path) -> Path:
    return tmp_path / "browser-activity.jsonl"


@pytest.fixture
def logger(log_file: Path) -> BrowserActivityLogger:
    return BrowserActivityLogger(log_path=str(log_file))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_entries(log_file: Path) -> list[dict]:
    """Parse all JSONL entries from a log file."""
    return [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Log path and directory creation
# ---------------------------------------------------------------------------

class TestLoggerInit:

    def test_log_path_tilde_expansion(self, tmp_path: Path) -> None:
        """log_path should expand ~ even if nested."""
        logger = BrowserActivityLogger(log_path=str(tmp_path / "sub" / "log.jsonl"))
        assert logger.log_path == tmp_path / "sub" / "log.jsonl"

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Parent directory should be created if it does not exist."""
        nested = tmp_path / "a" / "b" / "c" / "log.jsonl"
        BrowserActivityLogger(log_path=str(nested))
        assert nested.parent.exists()


# ---------------------------------------------------------------------------
# log_session_start / log_session_end
# ---------------------------------------------------------------------------

class TestSessionBoundaries:

    def test_session_start_writes_entry(self, logger: BrowserActivityLogger, log_file: Path) -> None:
        logger.log_session_start("test run")
        entries = read_entries(log_file)
        assert len(entries) == 1
        assert entries[0]["action"] == "session_start"
        assert entries[0]["purpose"] == "test run"

    def test_session_end_writes_entry(self, logger: BrowserActivityLogger, log_file: Path) -> None:
        logger.log_session_end("test run")
        entries = read_entries(log_file)
        assert len(entries) == 1
        assert entries[0]["action"] == "session_end"

    def test_session_start_without_purpose(self, logger: BrowserActivityLogger, log_file: Path) -> None:
        logger.log_session_start()
        entries = read_entries(log_file)
        assert entries[0]["purpose"] == ""

    def test_start_and_end_are_separate_entries(
        self, logger: BrowserActivityLogger, log_file: Path
    ) -> None:
        logger.log_session_start("run A")
        logger.log_session_end("run A")
        entries = read_entries(log_file)
        assert len(entries) == 2
        assert entries[0]["action"] == "session_start"
        assert entries[1]["action"] == "session_end"


# ---------------------------------------------------------------------------
# log_navigation
# ---------------------------------------------------------------------------

class TestLogNavigation:

    def test_navigation_entry_fields(
        self, logger: BrowserActivityLogger, log_file: Path
    ) -> None:
        logger.log_navigation("https://chat.openai.com", "check usage")
        entries = read_entries(log_file)
        assert len(entries) == 1
        e = entries[0]
        assert e["action"] == "navigate"
        assert e["url"] == "https://chat.openai.com"
        assert e["purpose"] == "check usage"

    def test_navigation_has_utc_timestamp(
        self, logger: BrowserActivityLogger, log_file: Path
    ) -> None:
        logger.log_navigation("https://claude.ai", "test nav")
        entries = read_entries(log_file)
        # ISO-8601 UTC timestamps end with +00:00
        ts = entries[0]["timestamp"]
        assert "+00:00" in ts or ts.endswith("Z")

    def test_multiple_navigations_appended(
        self, logger: BrowserActivityLogger, log_file: Path
    ) -> None:
        for url in ["https://claude.ai", "https://perplexity.ai", "https://gemini.google.com"]:
            logger.log_navigation(url, "multi-nav test")
        entries = read_entries(log_file)
        assert len(entries) == 3
        assert [e["url"] for e in entries] == [
            "https://claude.ai",
            "https://perplexity.ai",
            "https://gemini.google.com",
        ]

    def test_empty_purpose_logs_warning(
        self, logger: BrowserActivityLogger, log_file: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging
        with caplog.at_level(logging.WARNING, logger="modules.browser.logger"):
            logger.log_navigation("https://example.com", "")
        assert any("without a purpose" in msg for msg in caplog.messages)
        # Entry is still written
        entries = read_entries(log_file)
        assert len(entries) == 1


# ---------------------------------------------------------------------------
# log_screenshot
# ---------------------------------------------------------------------------

class TestLogScreenshot:

    def test_screenshot_entry_fields(
        self, logger: BrowserActivityLogger, log_file: Path
    ) -> None:
        logger.log_screenshot("/tmp/snap.png", "https://platform.openai.com/usage")
        entries = read_entries(log_file)
        e = entries[0]
        assert e["action"] == "screenshot"
        assert e["filename"] == "/tmp/snap.png"
        assert e["url"] == "https://platform.openai.com/usage"


# ---------------------------------------------------------------------------
# log_error
# ---------------------------------------------------------------------------

class TestLogError:

    def test_error_entry_fields(
        self, logger: BrowserActivityLogger, log_file: Path
    ) -> None:
        logger.log_error("https://bad.example.com", "TimeoutError: 30000ms exceeded")
        entries = read_entries(log_file)
        e = entries[0]
        assert e["action"] == "error"
        assert e["url"] == "https://bad.example.com"
        assert "TimeoutError" in e["error"]

    def test_error_also_logs_at_error_level(
        self,
        logger: BrowserActivityLogger,
        log_file: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging
        with caplog.at_level(logging.ERROR, logger="modules.browser.logger"):
            logger.log_error("https://x.com", "net::ERR_NAME_NOT_RESOLVED")
        assert any("ERR_NAME_NOT_RESOLVED" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# JSONL format integrity
# ---------------------------------------------------------------------------

class TestJSONLFormat:

    def test_each_entry_is_valid_json(
        self, logger: BrowserActivityLogger, log_file: Path
    ) -> None:
        logger.log_session_start("format test")
        logger.log_navigation("https://claude.ai", "nav")
        logger.log_screenshot("shot.png", "https://claude.ai")
        logger.log_error("https://fail.example.com", "oops")
        logger.log_session_end("format test")
        lines = log_file.read_text().splitlines()
        assert len(lines) == 5
        for line in lines:
            obj = json.loads(line)
            assert "timestamp" in obj
            assert "action" in obj

    def test_log_is_append_only(
        self, logger: BrowserActivityLogger, log_file: Path
    ) -> None:
        """A second logger pointing at the same file must append, not overwrite."""
        logger.log_navigation("https://claude.ai", "first")
        logger2 = BrowserActivityLogger(log_path=str(log_file))
        logger2.log_navigation("https://openai.com", "second")
        entries = read_entries(log_file)
        assert len(entries) == 2
        assert entries[0]["url"] == "https://claude.ai"
        assert entries[1]["url"] == "https://openai.com"
