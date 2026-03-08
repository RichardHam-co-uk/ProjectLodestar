"""Tests for DerekBrowserClient.

All tests mock Playwright so they run without a display server or
network access — no real browser is launched. Tests verify that the
client correctly calls start/stop, logs audit events, enforces the
security contract, and raises cleanly on errors.
"""

import signal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from modules.browser import client as browser_client_module
from modules.browser.client import (
    DerekBrowserClient,
    SESSION_TIMEOUT_SECONDS,
    OPERATION_TIMEOUT_MS,
    _SECURE_CHROMIUM_ARGS,
    _session_timeout_handler,
)
from modules.browser.logger import BrowserActivityLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def log_file(tmp_path: Path) -> Path:
    return tmp_path / "test-activity.jsonl"


@pytest.fixture
def mock_pw():
    """Return a fully-mocked sync_playwright context manager."""
    page = MagicMock()
    page.url = "https://example.com"
    page.title.return_value = "Example Domain"

    context = MagicMock()
    context.new_page.return_value = page

    browser = MagicMock()
    browser.new_context.return_value = context

    chromium = MagicMock()
    chromium.launch.return_value = browser

    playwright_instance = MagicMock()
    playwright_instance.chromium = chromium

    return playwright_instance, browser, context, page


@pytest.fixture
def client(log_file: Path) -> DerekBrowserClient:
    return DerekBrowserClient(purpose="unit test", log_path=str(log_file))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_client_with_mock(log_file: Path, mock_pw_tuple: tuple) -> tuple:
    """Patch sync_playwright and return (client, pw_mocks)."""
    playwright_instance, browser, context, page = mock_pw_tuple
    patcher = patch(
        "modules.browser.client.sync_playwright",
        return_value=MagicMock(
            __enter__=MagicMock(return_value=playwright_instance),
            __exit__=MagicMock(return_value=False),
            start=MagicMock(return_value=playwright_instance),
        ),
    )
    return patcher, playwright_instance, browser, context, page


# ---------------------------------------------------------------------------
# Security contract: mandatory launch args
# ---------------------------------------------------------------------------

class TestSecurityContract:

    def test_secure_args_include_no_sandbox(self) -> None:
        assert "--no-sandbox" in _SECURE_CHROMIUM_ARGS

    def test_secure_args_disable_sync(self) -> None:
        assert "--disable-sync" in _SECURE_CHROMIUM_ARGS

    def test_secure_args_incognito(self) -> None:
        assert "--incognito" in _SECURE_CHROMIUM_ARGS

    def test_secure_args_disable_extensions(self) -> None:
        assert "--disable-extensions" in _SECURE_CHROMIUM_ARGS

    def test_secure_args_disable_breakpad(self) -> None:
        assert "--disable-breakpad" in _SECURE_CHROMIUM_ARGS

    def test_secure_args_disable_background_networking(self) -> None:
        assert "--disable-background-networking" in _SECURE_CHROMIUM_ARGS

    def test_secure_args_disable_gpu(self) -> None:
        assert "--disable-gpu" in _SECURE_CHROMIUM_ARGS

    def test_session_timeout_is_300_seconds(self) -> None:
        assert SESSION_TIMEOUT_SECONDS == 300

    def test_operation_timeout_is_30_seconds(self) -> None:
        assert OPERATION_TIMEOUT_MS == 30_000


# ---------------------------------------------------------------------------
# _session_timeout_handler
# ---------------------------------------------------------------------------

class TestSessionTimeoutHandler:

    def test_raises_timeout_error(self) -> None:
        with pytest.raises(TimeoutError, match="hard limit"):
            _session_timeout_handler(signal.SIGALRM, None)


# ---------------------------------------------------------------------------
# start()
# ---------------------------------------------------------------------------

class TestClientStart:

    def test_start_launches_chromium_with_secure_args(
        self, client: DerekBrowserClient, mock_pw: tuple
    ) -> None:
        playwright_instance, browser, context, page = mock_pw
        with patch("modules.browser.client.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = playwright_instance
            with patch("signal.alarm"):
                client.start()

        playwright_instance.chromium.launch.assert_called_once_with(
            headless=True,
            args=_SECURE_CHROMIUM_ARGS,
            slow_mo=0,
        )

    def test_start_sets_context_timeouts(
        self, client: DerekBrowserClient, mock_pw: tuple
    ) -> None:
        playwright_instance, browser, context, page = mock_pw
        with patch("modules.browser.client.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = playwright_instance
            with patch("signal.alarm"):
                client.start()

        context.set_default_timeout.assert_called_once_with(OPERATION_TIMEOUT_MS)
        context.set_default_navigation_timeout.assert_called_once_with(OPERATION_TIMEOUT_MS)

    def test_start_arms_signal_alarm(
        self, client: DerekBrowserClient, mock_pw: tuple
    ) -> None:
        playwright_instance, browser, context, page = mock_pw
        with patch("modules.browser.client.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = playwright_instance
            with patch("signal.alarm") as mock_alarm:
                client.start()

        mock_alarm.assert_called_with(SESSION_TIMEOUT_SECONDS)

    def test_start_idempotent(
        self, client: DerekBrowserClient, mock_pw: tuple
    ) -> None:
        """Calling start() twice must not launch a second browser."""
        playwright_instance, browser, context, page = mock_pw
        with patch("modules.browser.client.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = playwright_instance
            with patch("signal.alarm"):
                client.start()
                client.start()

        assert playwright_instance.chromium.launch.call_count == 1

    def test_start_writes_session_start_log(
        self, client: DerekBrowserClient, mock_pw: tuple, log_file: Path
    ) -> None:
        import json
        playwright_instance, browser, context, page = mock_pw
        with patch("modules.browser.client.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = playwright_instance
            with patch("signal.alarm"):
                client.start()
                client.stop()

        entries = [json.loads(l) for l in log_file.read_text().splitlines() if l.strip()]
        actions = [e["action"] for e in entries]
        assert "session_start" in actions

    def test_start_headed_mode_passes_headless_false(
        self, mock_pw: tuple, tmp_path: Path
    ) -> None:
        """headless=False and slow_mo should be forwarded to chromium.launch."""
        playwright_instance, browser, context, page = mock_pw
        headed_client = DerekBrowserClient(
            purpose="headed test",
            headless=False,
            slow_mo=250,
            log_path=str(tmp_path / "log.jsonl"),
        )
        with patch("modules.browser.client.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = playwright_instance
            with patch("signal.alarm"):
                headed_client.start()
                headed_client.stop()

        playwright_instance.chromium.launch.assert_called_once_with(
            headless=False,
            args=_SECURE_CHROMIUM_ARGS,
            slow_mo=250,
        )

    def test_start_raises_import_error_if_playwright_missing(
        self, client: DerekBrowserClient
    ) -> None:
        with patch("modules.browser.client.sync_playwright", None):
            with pytest.raises(ImportError, match="playwright is not installed"):
                client.start()


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------

class TestClientStop:

    def test_stop_closes_browser(
        self, client: DerekBrowserClient, mock_pw: tuple
    ) -> None:
        playwright_instance, browser, context, page = mock_pw
        with patch("modules.browser.client.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = playwright_instance
            with patch("signal.alarm"):
                client.start()
                client.stop()

        browser.close.assert_called()

    def test_stop_disarms_alarm(
        self, client: DerekBrowserClient, mock_pw: tuple
    ) -> None:
        playwright_instance, browser, context, page = mock_pw
        with patch("modules.browser.client.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = playwright_instance
            with patch("signal.alarm") as mock_alarm:
                client.start()
                client.stop()

        # signal.alarm(0) must be called in stop()
        calls = mock_alarm.call_args_list
        assert call(0) in calls

    def test_stop_writes_session_end_log(
        self, client: DerekBrowserClient, mock_pw: tuple, log_file: Path
    ) -> None:
        import json
        playwright_instance, browser, context, page = mock_pw
        with patch("modules.browser.client.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = playwright_instance
            with patch("signal.alarm"):
                client.start()
                client.stop()

        entries = [json.loads(l) for l in log_file.read_text().splitlines() if l.strip()]
        actions = [e["action"] for e in entries]
        assert "session_end" in actions

    def test_stop_idempotent(
        self, client: DerekBrowserClient, mock_pw: tuple
    ) -> None:
        """stop() must not raise even if called before start() or twice."""
        with patch("signal.alarm"):
            client.stop()  # not started — should not raise
            client.stop()


# ---------------------------------------------------------------------------
# navigate()
# ---------------------------------------------------------------------------

class TestClientNavigate:

    def test_navigate_calls_page_goto(
        self, client: DerekBrowserClient, mock_pw: tuple
    ) -> None:
        playwright_instance, browser, context, page = mock_pw
        with patch("modules.browser.client.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = playwright_instance
            with patch("signal.alarm"):
                client.start()
                client.navigate("https://claude.ai", "test nav")

        page.goto.assert_called_once_with(
            "https://claude.ai", wait_until="domcontentloaded"
        )

    def test_navigate_logs_navigation(
        self, client: DerekBrowserClient, mock_pw: tuple, log_file: Path
    ) -> None:
        import json
        playwright_instance, browser, context, page = mock_pw
        with patch("modules.browser.client.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = playwright_instance
            with patch("signal.alarm"):
                client.start()
                client.navigate("https://perplexity.ai", "perplexity nav")
                client.stop()

        entries = [json.loads(l) for l in log_file.read_text().splitlines() if l.strip()]
        nav_entries = [e for e in entries if e["action"] == "navigate"]
        assert len(nav_entries) == 1
        assert nav_entries[0]["url"] == "https://perplexity.ai"
        assert nav_entries[0]["purpose"] == "perplexity nav"

    def test_navigate_raises_without_start(self, client: DerekBrowserClient) -> None:
        with pytest.raises(RuntimeError, match="not started"):
            client.navigate("https://example.com")

    def test_navigate_logs_error_on_goto_failure(
        self, client: DerekBrowserClient, mock_pw: tuple, log_file: Path
    ) -> None:
        import json
        playwright_instance, browser, context, page = mock_pw
        page.goto.side_effect = Exception("net::ERR_NAME_NOT_RESOLVED")
        with patch("modules.browser.client.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = playwright_instance
            with patch("signal.alarm"):
                client.start()
                with pytest.raises(Exception, match="ERR_NAME_NOT_RESOLVED"):
                    client.navigate("https://blocked.example.com", "blocked nav")

        entries = [json.loads(l) for l in log_file.read_text().splitlines() if l.strip()]
        error_entries = [e for e in entries if e["action"] == "error"]
        assert len(error_entries) == 1
        assert "ERR_NAME_NOT_RESOLVED" in error_entries[0]["error"]


# ---------------------------------------------------------------------------
# screenshot()
# ---------------------------------------------------------------------------

class TestClientScreenshot:

    def test_screenshot_calls_page_screenshot(
        self, client: DerekBrowserClient, mock_pw: tuple, tmp_path: Path
    ) -> None:
        playwright_instance, browser, context, page = mock_pw
        dest = tmp_path / "snap.png"
        client2 = DerekBrowserClient(
            purpose="shot test",
            log_path=str(tmp_path / "log.jsonl"),
            screenshot_dir=str(tmp_path),
        )
        with patch("modules.browser.client.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = playwright_instance
            with patch("signal.alarm"):
                client2.start()
                result = client2.screenshot("snap.png")

        page.screenshot.assert_called_once_with(path=str(dest), full_page=True)
        assert result == dest

    def test_screenshot_logs_screenshot_entry(
        self, mock_pw: tuple, tmp_path: Path
    ) -> None:
        import json
        playwright_instance, browser, context, page = mock_pw
        page.url = "https://platform.openai.com/usage"
        log_path = tmp_path / "log.jsonl"
        client2 = DerekBrowserClient(
            purpose="screenshot log test",
            log_path=str(log_path),
            screenshot_dir=str(tmp_path),
        )
        with patch("modules.browser.client.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = playwright_instance
            with patch("signal.alarm"):
                client2.start()
                client2.screenshot("usage.png")
                client2.stop()

        entries = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
        shot_entries = [e for e in entries if e["action"] == "screenshot"]
        assert len(shot_entries) == 1
        assert "usage.png" in shot_entries[0]["filename"]

    def test_screenshot_raises_without_start(self, client: DerekBrowserClient) -> None:
        with pytest.raises(RuntimeError, match="not started"):
            client.screenshot("snap.png")


# ---------------------------------------------------------------------------
# page_title()
# ---------------------------------------------------------------------------

class TestClientPageTitle:

    def test_page_title_returns_title(
        self, client: DerekBrowserClient, mock_pw: tuple
    ) -> None:
        playwright_instance, browser, context, page = mock_pw
        page.title.return_value = "OpenAI Usage"
        with patch("modules.browser.client.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = playwright_instance
            with patch("signal.alarm"):
                client.start()
                title = client.page_title()

        assert title == "OpenAI Usage"

    def test_page_title_raises_without_start(self, client: DerekBrowserClient) -> None:
        with pytest.raises(RuntimeError, match="not started"):
            client.page_title()


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:

    def test_context_manager_calls_start_and_stop(
        self, mock_pw: tuple, tmp_path: Path
    ) -> None:
        playwright_instance, browser, context, page = mock_pw
        with patch("modules.browser.client.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = playwright_instance
            with patch("signal.alarm"):
                with DerekBrowserClient(
                    purpose="ctx test",
                    log_path=str(tmp_path / "log.jsonl"),
                ) as c:
                    assert c._browser is not None

        # After exiting context, browser should be closed
        browser.close.assert_called()

    def test_context_manager_closes_on_exception(
        self, mock_pw: tuple, tmp_path: Path
    ) -> None:
        playwright_instance, browser, context, page = mock_pw
        with patch("modules.browser.client.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = playwright_instance
            with patch("signal.alarm"):
                with pytest.raises(ValueError):
                    with DerekBrowserClient(
                        purpose="error test",
                        log_path=str(tmp_path / "log.jsonl"),
                    ):
                        raise ValueError("simulated error")

        # Browser must still be closed even when an exception propagates
        browser.close.assert_called()
