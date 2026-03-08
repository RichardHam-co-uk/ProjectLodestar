"""Tests for modules/costs/dashboard_scraper.py.

All Playwright calls are mocked — no browser or network access required.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_playwright(monkeypatch):
    """Stub out sync_playwright so no browser is launched."""
    mock_page = MagicMock()
    mock_page.goto.return_value = None
    mock_page.wait_for_timeout.return_value = None
    mock_page.screenshot.return_value = None

    mock_context = MagicMock()
    mock_context.new_page.return_value = mock_page
    mock_context.storage_state.return_value = {
        "cookies": [{"name": "session", "value": "abc", "domain": ".anthropic.com"}],
        "origins": [],
    }

    mock_browser = MagicMock()
    mock_browser.new_context.return_value = mock_context

    mock_pw_instance = MagicMock()
    mock_pw_instance.__enter__ = lambda s: s
    mock_pw_instance.__exit__ = MagicMock(return_value=False)
    mock_pw_instance.chromium.launch.return_value = mock_browser

    mock_sync_playwright = MagicMock(return_value=mock_pw_instance)

    monkeypatch.setattr(
        "modules.costs.dashboard_scraper.sync_playwright",
        mock_sync_playwright,
        raising=False,
    )

    # Patch signal.alarm so tests don't arm real SIGALRM
    monkeypatch.setattr("modules.costs.dashboard_scraper.signal.alarm", MagicMock())
    monkeypatch.setattr("modules.costs.dashboard_scraper.signal.signal", MagicMock())

    return mock_page, mock_context, mock_browser


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

def test_module_imports():
    """dashboard_scraper imports without error."""
    import modules.costs.dashboard_scraper as m
    assert hasattr(m, "capture_dashboards")
    assert hasattr(m, "DASHBOARDS")
    assert hasattr(m, "COOKIE_FILE")


# ---------------------------------------------------------------------------
# DASHBOARDS constant
# ---------------------------------------------------------------------------

def test_dashboards_list_has_three_providers():
    from modules.costs.dashboard_scraper import DASHBOARDS
    assert len(DASHBOARDS) == 3


def test_dashboards_providers():
    from modules.costs.dashboard_scraper import DASHBOARDS
    providers = [d[0] for d in DASHBOARDS]
    assert "anthropic" in providers
    assert "openai" in providers
    assert "xai" in providers


def test_dashboards_urls_are_https():
    from modules.costs.dashboard_scraper import DASHBOARDS
    for _, url, _ in DASHBOARDS:
        assert url.startswith("https://"), f"URL not HTTPS: {url}"


def test_dashboards_filenames_are_png():
    from modules.costs.dashboard_scraper import DASHBOARDS
    for _, _, fname in DASHBOARDS:
        assert fname.endswith(".png"), f"Filename not PNG: {fname}"


# ---------------------------------------------------------------------------
# headless=True without cookie file raises FileNotFoundError
# ---------------------------------------------------------------------------

def test_headless_without_cookie_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "modules.costs.dashboard_scraper.COOKIE_FILE",
        tmp_path / "nonexistent-auth.json",
    )
    from modules.costs.dashboard_scraper import capture_dashboards

    with pytest.raises(FileNotFoundError, match="No saved session"):
        capture_dashboards(headless=True)


# ---------------------------------------------------------------------------
# First run (no cookie file) — headed, saves cookies after
# ---------------------------------------------------------------------------

def test_first_run_saves_cookies(tmp_path, monkeypatch, _patch_playwright):
    cookie_file = tmp_path / "auth.json"
    screenshot_base = tmp_path / "screenshots"

    monkeypatch.setattr("modules.costs.dashboard_scraper.COOKIE_FILE", cookie_file)
    monkeypatch.setattr(
        "modules.costs.dashboard_scraper.SCREENSHOT_BASE", screenshot_base
    )
    # Bypass the input() call
    monkeypatch.setattr("builtins.input", lambda _: "")

    from modules.costs.dashboard_scraper import capture_dashboards

    capture_dashboards(headless=False)

    assert cookie_file.exists(), "Cookie file should be created on first run"
    data = json.loads(cookie_file.read_text())
    assert "cookies" in data


def test_first_run_creates_screenshot_dir(tmp_path, monkeypatch, _patch_playwright):
    cookie_file = tmp_path / "auth.json"
    screenshot_base = tmp_path / "screenshots"

    monkeypatch.setattr("modules.costs.dashboard_scraper.COOKIE_FILE", cookie_file)
    monkeypatch.setattr(
        "modules.costs.dashboard_scraper.SCREENSHOT_BASE", screenshot_base
    )
    monkeypatch.setattr("builtins.input", lambda _: "")

    from modules.costs.dashboard_scraper import capture_dashboards

    capture_dashboards(headless=False)

    # A timestamped sub-directory should exist under screenshot_base
    subdirs = list(screenshot_base.iterdir())
    assert len(subdirs) == 1
    assert subdirs[0].is_dir()


def test_first_run_returns_path_list(tmp_path, monkeypatch, _patch_playwright):
    cookie_file = tmp_path / "auth.json"
    screenshot_base = tmp_path / "screenshots"

    monkeypatch.setattr("modules.costs.dashboard_scraper.COOKIE_FILE", cookie_file)
    monkeypatch.setattr(
        "modules.costs.dashboard_scraper.SCREENSHOT_BASE", screenshot_base
    )
    monkeypatch.setattr("builtins.input", lambda _: "")

    from modules.costs.dashboard_scraper import DASHBOARDS, capture_dashboards

    paths = capture_dashboards(headless=False)

    assert isinstance(paths, list)
    assert len(paths) == len(DASHBOARDS)
    for p in paths:
        assert isinstance(p, Path)


# ---------------------------------------------------------------------------
# Subsequent run (cookie file exists) — headless, loads cookies
# ---------------------------------------------------------------------------

def test_headless_run_loads_cookies(tmp_path, monkeypatch, _patch_playwright):
    cookie_file = tmp_path / "auth.json"
    screenshot_base = tmp_path / "screenshots"
    fake_state = {"cookies": [{"name": "tok", "value": "x"}], "origins": []}
    cookie_file.write_text(json.dumps(fake_state))

    monkeypatch.setattr("modules.costs.dashboard_scraper.COOKIE_FILE", cookie_file)
    monkeypatch.setattr(
        "modules.costs.dashboard_scraper.SCREENSHOT_BASE", screenshot_base
    )

    _, mock_context, mock_browser = _patch_playwright

    from modules.costs.dashboard_scraper import capture_dashboards

    capture_dashboards(headless=True)

    # new_context should have been called with storage_state
    call_kwargs = mock_browser.new_context.call_args[1]
    assert "storage_state" in call_kwargs
    assert call_kwargs["storage_state"] == fake_state


def test_headless_run_does_not_overwrite_cookies(tmp_path, monkeypatch, _patch_playwright):
    cookie_file = tmp_path / "auth.json"
    screenshot_base = tmp_path / "screenshots"
    fake_state = {"cookies": [{"name": "tok", "value": "original"}], "origins": []}
    cookie_file.write_text(json.dumps(fake_state))
    original_mtime = cookie_file.stat().st_mtime

    monkeypatch.setattr("modules.costs.dashboard_scraper.COOKIE_FILE", cookie_file)
    monkeypatch.setattr(
        "modules.costs.dashboard_scraper.SCREENSHOT_BASE", screenshot_base
    )

    from modules.costs.dashboard_scraper import capture_dashboards

    capture_dashboards(headless=True)

    # Cookie file should not have been rewritten
    assert cookie_file.stat().st_mtime == original_mtime


# ---------------------------------------------------------------------------
# --reauth flag
# ---------------------------------------------------------------------------

def test_reauth_deletes_existing_cookie_file(tmp_path, monkeypatch, _patch_playwright):
    cookie_file = tmp_path / "auth.json"
    screenshot_base = tmp_path / "screenshots"
    cookie_file.write_text('{"cookies": [], "origins": []}')

    monkeypatch.setattr("modules.costs.dashboard_scraper.COOKIE_FILE", cookie_file)
    monkeypatch.setattr(
        "modules.costs.dashboard_scraper.SCREENSHOT_BASE", screenshot_base
    )
    monkeypatch.setattr("builtins.input", lambda _: "")

    from modules.costs.dashboard_scraper import capture_dashboards

    capture_dashboards(headless=False, reauth=True)

    # After reauth, new cookie file should be written (fresh state)
    assert cookie_file.exists()


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def test_cli_parser_defaults():
    from modules.costs.dashboard_scraper import _build_parser

    args = _build_parser().parse_args([])
    assert args.headless is False
    assert args.reauth is False
    assert args.slow_mo == 500


def test_cli_parser_headless_flag():
    from modules.costs.dashboard_scraper import _build_parser

    args = _build_parser().parse_args(["--headless"])
    assert args.headless is True


def test_cli_parser_reauth_flag():
    from modules.costs.dashboard_scraper import _build_parser

    args = _build_parser().parse_args(["--reauth"])
    assert args.reauth is True


def test_cli_parser_slow_mo():
    from modules.costs.dashboard_scraper import _build_parser

    args = _build_parser().parse_args(["--slow-mo", "1000"])
    assert args.slow_mo == 1000


# ---------------------------------------------------------------------------
# main() return codes
# ---------------------------------------------------------------------------

def test_main_headless_no_cookie_returns_1(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "modules.costs.dashboard_scraper.COOKIE_FILE",
        tmp_path / "nonexistent.json",
    )
    from modules.costs.dashboard_scraper import main

    rc = main(["--headless"])
    assert rc == 1


def test_main_success_returns_0(tmp_path, monkeypatch, _patch_playwright):
    cookie_file = tmp_path / "auth.json"
    screenshot_base = tmp_path / "screenshots"
    monkeypatch.setattr("modules.costs.dashboard_scraper.COOKIE_FILE", cookie_file)
    monkeypatch.setattr(
        "modules.costs.dashboard_scraper.SCREENSHOT_BASE", screenshot_base
    )
    monkeypatch.setattr("builtins.input", lambda _: "")

    from modules.costs.dashboard_scraper import main

    rc = main([])
    assert rc == 0


# ---------------------------------------------------------------------------
# Bot-detection hardening
# ---------------------------------------------------------------------------

def test_chromium_args_has_automation_controlled():
    """--disable-blink-features=AutomationControlled must be in launch args."""
    from modules.costs.dashboard_scraper import _CHROMIUM_ARGS
    assert "--disable-blink-features=AutomationControlled" in _CHROMIUM_ARGS


def test_chromium_args_no_obvious_automation_fingerprints():
    """Flags that fingerprint an automated Chromium must not appear."""
    from modules.costs.dashboard_scraper import _CHROMIUM_ARGS
    forbidden = [
        "--disable-extensions",
        "--disable-background-networking",
        "--disable-breakpad",
        "--disable-component-update",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    for flag in forbidden:
        assert flag not in _CHROMIUM_ARGS, f"Automation fingerprint flag present: {flag}"


def test_user_agent_is_set_in_context(tmp_path, monkeypatch, _patch_playwright):
    """A realistic user-agent must be passed to new_context()."""
    cookie_file = tmp_path / "auth.json"
    screenshot_base = tmp_path / "screenshots"
    monkeypatch.setattr("modules.costs.dashboard_scraper.COOKIE_FILE", cookie_file)
    monkeypatch.setattr("modules.costs.dashboard_scraper.SCREENSHOT_BASE", screenshot_base)
    monkeypatch.setattr("builtins.input", lambda _: "")

    _, _, mock_browser = _patch_playwright

    from modules.costs.dashboard_scraper import capture_dashboards, _USER_AGENT
    capture_dashboards(headless=False)

    call_kwargs = mock_browser.new_context.call_args[1]
    assert "user_agent" in call_kwargs
    assert call_kwargs["user_agent"] == _USER_AGENT


def test_user_agent_is_not_playwright_default():
    """User-agent must not contain 'Playwright' or 'HeadlessChrome'."""
    from modules.costs.dashboard_scraper import _USER_AGENT
    assert "Playwright" not in _USER_AGENT
    assert "HeadlessChrome" not in _USER_AGENT


def test_stealth_init_script_applied(tmp_path, monkeypatch, _patch_playwright):
    """add_init_script() must be called with the stealth patch on every run."""
    cookie_file = tmp_path / "auth.json"
    screenshot_base = tmp_path / "screenshots"
    monkeypatch.setattr("modules.costs.dashboard_scraper.COOKIE_FILE", cookie_file)
    monkeypatch.setattr("modules.costs.dashboard_scraper.SCREENSHOT_BASE", screenshot_base)
    monkeypatch.setattr("builtins.input", lambda _: "")

    _, mock_context, _ = _patch_playwright

    from modules.costs.dashboard_scraper import capture_dashboards, _STEALTH_INIT_SCRIPT
    capture_dashboards(headless=False)

    mock_context.add_init_script.assert_called_once_with(_STEALTH_INIT_SCRIPT)


def test_stealth_script_patches_webdriver():
    """Stealth init script must patch navigator.webdriver."""
    from modules.costs.dashboard_scraper import _STEALTH_INIT_SCRIPT
    assert "navigator" in _STEALTH_INIT_SCRIPT
    assert "webdriver" in _STEALTH_INIT_SCRIPT


def test_stealth_script_patches_chrome_runtime():
    """Stealth init script must restore window.chrome."""
    from modules.costs.dashboard_scraper import _STEALTH_INIT_SCRIPT
    assert "window.chrome" in _STEALTH_INIT_SCRIPT
