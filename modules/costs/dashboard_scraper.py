"""AI provider usage dashboard scraper.

Captures screenshots of the Anthropic, OpenAI, and xAI usage dashboards
so spend can be cross-referenced against the local SQLite cost records.

Authentication model
--------------------
Dashboards require a logged-in session. This module uses Playwright's
storage_state mechanism to persist cookies between runs:

  First run  — launches a HEADED browser (visible window on $DISPLAY).
               Rich logs in manually to each provider. When done, the
               session cookies are saved to COOKIE_FILE.

  Subsequent — loads cookies from COOKIE_FILE and runs HEADLESS.
               No manual interaction needed.

Usage
-----
From the repo root with the venv active::

    # First run — headed, manual login
    cd ~/Documents/github-repos/ProjectLodestar
    source .venv/bin/activate
    python -m modules.costs.dashboard_scraper

    # Force a re-login (rotate cookies)
    python -m modules.costs.dashboard_scraper --reauth

    # Headless / cron — assumes cookies already saved
    python -m modules.costs.dashboard_scraper --headless

Screenshots are saved to ~/.lodestar/screenshots/<timestamp>/.
"""

import argparse
import json
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Module-level import so tests can patch modules.costs.dashboard_scraper.sync_playwright.
# Graceful degradation: attribute exists (as None) even if playwright is absent.
try:
    from playwright.sync_api import sync_playwright  # type: ignore[import]
except ImportError:  # pragma: no cover
    sync_playwright = None  # type: ignore[assignment]

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LODESTAR_DIR = Path.home() / ".lodestar"
COOKIE_FILE = LODESTAR_DIR / "dashboard-auth.json"
SCREENSHOT_BASE = LODESTAR_DIR / "screenshots"

# ---------------------------------------------------------------------------
# Dashboard targets
# ---------------------------------------------------------------------------
DASHBOARDS: list[tuple[str, str, str]] = [
    (
        "anthropic",
        "https://console.anthropic.com/settings/usage",
        "anthropic-usage.png",
    ),
    (
        "openai",
        "https://platform.openai.com/usage",
        "openai-usage.png",
    ),
    (
        "xai",
        "https://console.x.ai/",
        "xai-usage.png",
    ),
]

# ---------------------------------------------------------------------------
# Chromium args — same hardened set as DerekBrowserClient but WITHOUT
# --incognito so that storage_state (cookie persistence) works correctly.
# ---------------------------------------------------------------------------
_SECURE_ARGS: list[str] = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
    "--disable-accelerated-2d-canvas",
    "--disable-gpu",
    "--window-size=1440,900",
    "--disable-features=TranslateUI",
    "--disable-extensions",
    "--disable-plugins",
    "--disable-sync",
    "--disable-notifications",
    "--disable-background-networking",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-breakpad",
    "--disable-component-update",
    "--disable-domain-reliability",
    "--no-first-run",
    "--no-default-browser-check",
]

SESSION_TIMEOUT_SECONDS = 600  # 10 min — longer for manual login
OPERATION_TIMEOUT_MS = 30_000


def _timeout_handler(signum: int, frame: Any) -> None:
    raise TimeoutError("Dashboard scraper session exceeded timeout")


# ---------------------------------------------------------------------------
# Core scraper
# ---------------------------------------------------------------------------

def capture_dashboards(
    headless: bool = True,
    reauth: bool = False,
    slow_mo: int = 0,
) -> list[Path]:
    """Capture screenshots of all configured dashboards.

    Args:
        headless: Run without visible browser window. Requires COOKIE_FILE
                  to exist (i.e. a prior headed login session).
        reauth:   Delete existing COOKIE_FILE and force a new login.
        slow_mo:  Milliseconds between Playwright actions (headed mode).

    Returns:
        List of Path objects for each screenshot written.

    Raises:
        ImportError: If playwright is not installed.
        FileNotFoundError: If headless=True but COOKIE_FILE does not exist.
        TimeoutError: If the session exceeds SESSION_TIMEOUT_SECONDS.
    """
    if sync_playwright is None:
        raise ImportError(
            "playwright not installed. Run:\n"
            "  pip install playwright --break-system-packages\n"
            "  python -m playwright install chromium"
        )

    if reauth and COOKIE_FILE.exists():
        COOKIE_FILE.unlink()
        _log.info("Removed existing cookie file for re-authentication")

    first_run = not COOKIE_FILE.exists()

    if headless and first_run:
        raise FileNotFoundError(
            f"No saved session found at {COOKIE_FILE}.\n"
            "Run without --headless first to log in and save cookies."
        )

    # Prepare screenshot directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    screenshot_dir = SCREENSHOT_BASE / timestamp
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    _log.info("Screenshots will be saved to %s", screenshot_dir)

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(SESSION_TIMEOUT_SECONDS)

    saved_paths: list[Path] = []

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=headless,
                args=_SECURE_ARGS,
                slow_mo=slow_mo,
            )

            # Load existing cookies or start fresh
            storage: Optional[dict] = None
            if COOKIE_FILE.exists():
                with open(COOKIE_FILE) as f:
                    storage = json.load(f)
                _log.info("Loaded session cookies from %s", COOKIE_FILE)

            ctx_kwargs: dict[str, Any] = {
                "viewport": {"width": 1440, "height": 900},
            }
            if storage:
                ctx_kwargs["storage_state"] = storage

            context = browser.new_context(**ctx_kwargs)
            context.set_default_timeout(OPERATION_TIMEOUT_MS)
            context.set_default_navigation_timeout(OPERATION_TIMEOUT_MS)
            page = context.new_page()

            if first_run:
                _log.info("FIRST RUN — headed browser opened for manual login")
                print("\n" + "=" * 60)
                print("FIRST RUN — Manual login required")
                print("=" * 60)
                print(
                    "A browser window has opened. Please log in to each provider:\n"
                )
                for provider, url, _ in DASHBOARDS:
                    print(f"  {provider:10s}  {url}")

                print(
                    "\nAfter logging in to ALL providers, come back here and"
                    " press ENTER to capture screenshots and save your session."
                )
                input("\nPress ENTER when logged in to all providers... ")

            # Capture each dashboard
            for provider, url, filename in DASHBOARDS:
                _log.info("Navigating to %s (%s)", provider, url)
                print(f"  Capturing {provider}...", end=" ", flush=True)
                try:
                    page.goto(url, wait_until="networkidle", timeout=45_000)
                    # Brief wait for any JS-rendered charts to settle
                    page.wait_for_timeout(2000)
                    out_path = screenshot_dir / filename
                    page.screenshot(path=str(out_path), full_page=False)
                    saved_paths.append(out_path)
                    print(f"saved → {out_path.name}")
                except Exception as exc:
                    _log.warning("Failed to capture %s: %s", provider, exc)
                    print(f"FAILED — {exc}")

            # Persist session cookies for future headless runs
            if first_run or reauth:
                COOKIE_FILE.parent.mkdir(parents=True, exist_ok=True)
                state = context.storage_state()
                with open(COOKIE_FILE, "w") as f:
                    json.dump(state, f, indent=2)
                COOKIE_FILE.chmod(0o600)
                _log.info("Session cookies saved to %s (chmod 600)", COOKIE_FILE)
                print(f"\nSession saved to {COOKIE_FILE}")
                print("Future runs can use --headless (no login needed).")

            context.close()
            browser.close()

    finally:
        signal.alarm(0)

    return saved_paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Capture AI provider usage dashboard screenshots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # First run (headed, manual login):\n"
            "  python -m modules.costs.dashboard_scraper\n\n"
            "  # Headless (cookies already saved):\n"
            "  python -m modules.costs.dashboard_scraper --headless\n\n"
            "  # Force re-login:\n"
            "  python -m modules.costs.dashboard_scraper --reauth\n"
        ),
    )
    p.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without a visible browser window (requires prior login).",
    )
    p.add_argument(
        "--reauth",
        action="store_true",
        default=False,
        help="Delete saved cookies and re-authenticate.",
    )
    p.add_argument(
        "--slow-mo",
        type=int,
        default=500,
        metavar="MS",
        help="Milliseconds between actions in headed mode (default: 500).",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _build_parser().parse_args(argv)

    print("\nLodestar Dashboard Scraper")
    print("-" * 40)

    if args.headless:
        print("Mode: headless (using saved session)")
    elif args.reauth:
        print("Mode: headed + re-authentication")
    else:
        first = not COOKIE_FILE.exists()
        print(f"Mode: {'headed — first run, login required' if first else 'headed (cookies exist)'}")

    print(f"Cookie file: {COOKIE_FILE}")
    print(f"Screenshots: {SCREENSHOT_BASE}/<timestamp>/\n")

    try:
        paths = capture_dashboards(
            headless=args.headless,
            reauth=args.reauth,
            slow_mo=args.slow_mo,
        )
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    except TimeoutError as exc:
        print(f"\nTIMEOUT: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        _log.exception("Unexpected error")
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\n{'=' * 40}")
    print(f"Done — {len(paths)} screenshot(s) captured:")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
