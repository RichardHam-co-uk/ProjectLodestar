"""Hardened Playwright/Chromium browser client for Derek.

Implements the security requirements from docs/BROWSER_SECURITY.md:

- Mandatory hardened Chromium launch arguments
- Headless or headed mode (headed shows browser on screen — use DISPLAY=:0)
- 5-minute hard session timeout via signal.alarm
- 30-second per-operation timeout on all page interactions
- Full audit logging of every navigation and screenshot via BrowserActivityLogger
- Optional slow_mo delay for visibility when running headed
- Explicit browser.close() on exit (even under exceptions)

Usage (context manager — preferred)::

    # Visible on screen — shows browser window, 500ms between actions
    with DerekBrowserClient(purpose="Check OpenAI usage", headless=False, slow_mo=500) as c:
        c.navigate("https://platform.openai.com/usage")
        c.screenshot("openai-usage.png")

    # Silent / CI — headless, no display needed
    with DerekBrowserClient(purpose="Automated check") as c:
        c.navigate("https://platform.openai.com/usage")
        c.screenshot("openai-usage.png")

Usage (manual)::

    client = DerekBrowserClient(purpose="Spot check", headless=False)
    client.start()
    try:
        client.navigate("https://perplexity.ai")
    finally:
        client.stop()
"""

import logging
import signal
from pathlib import Path
from typing import Any, Optional

from modules.browser.logger import BrowserActivityLogger

# Module-level import so tests can patch modules.browser.client.sync_playwright.
# Graceful degradation: the attribute exists (as None) even if playwright is absent.
try:
    from playwright.sync_api import sync_playwright  # type: ignore[import]
except ImportError:  # pragma: no cover
    sync_playwright = None  # type: ignore[assignment]

_log = logging.getLogger(__name__)

# Maximum wall-clock duration for an entire browser session.
SESSION_TIMEOUT_SECONDS = 300  # 5 minutes

# Per-operation timeout forwarded to Playwright context.
OPERATION_TIMEOUT_MS = 30_000  # 30 seconds

# Hardened Chromium launch args — see docs/BROWSER_SECURITY.md
_SECURE_CHROMIUM_ARGS: list[str] = [
    "--no-sandbox",                            # Required in VM/container environments
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",                 # Use /tmp instead of /dev/shm
    "--disable-accelerated-2d-canvas",
    "--disable-gpu",
    "--window-size=1280,720",
    "--disable-features=TranslateUI",
    "--disable-extensions",                    # No browser extensions
    "--disable-plugins",                       # No Flash/Java
    "--disable-sync",                          # Never sync with Google account
    "--incognito",                             # Private browsing; no persistent data
    "--disable-notifications",
    "--disable-background-networking",         # No background requests
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-breakpad",                      # No crash reports to Google
    "--disable-component-update",              # No auto-updates during session
    "--disable-domain-reliability",            # No usage stats to Google
    "--no-first-run",
    "--no-default-browser-check",
]


def _session_timeout_handler(signum: int, frame: Any) -> None:
    """SIGALRM handler — raised when a session exceeds SESSION_TIMEOUT_SECONDS."""
    raise TimeoutError(
        f"Browser session exceeded {SESSION_TIMEOUT_SECONDS}s hard limit"
    )


class DerekBrowserClient:
    """Hardened, auditable Playwright Chromium client.

    Each instance manages a single browser session. Sessions are bounded
    by a 5-minute hard timeout (signal.alarm). All navigations and
    screenshots are logged to ~/.lodestar/browser-activity.jsonl.

    Args:
        purpose: Human-readable description of what this session will do.
            Stored in the session_start log entry and included in INFO logs.
        headless: If False, opens a visible browser window on $DISPLAY.
            Defaults to True (no display needed, safe for unattended runs).
            Set to False when you want to watch the browser live.
        slow_mo: Milliseconds to wait between Playwright actions. Only
            meaningful when headless=False — gives you time to follow
            what the browser is doing. Defaults to 0 (no delay).
        log_path: Override the default activity log path. Primarily used
            in tests to redirect logs to a temp directory.
        screenshot_dir: Directory for saving screenshots. Defaults to the
            current working directory.
    """

    def __init__(
        self,
        purpose: str = "",
        headless: bool = True,
        slow_mo: int = 0,
        log_path: Optional[str] = None,
        screenshot_dir: Optional[str] = None,
    ) -> None:
        self._purpose = purpose
        self._headless = headless
        self._slow_mo = slow_mo
        self._screenshot_dir = Path(screenshot_dir) if screenshot_dir else Path(".")
        self._activity_log = (
            BrowserActivityLogger(log_path) if log_path else BrowserActivityLogger()
        )

        # Playwright objects — populated in start()
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the browser session.

        Launches Chromium with hardened args, sets up audit logging, and
        arms the 5-minute session timeout. Idempotent — calling start()
        on an already-started client is a no-op.

        Raises:
            ImportError: If playwright is not installed in the venv.
            RuntimeError: If the browser fails to launch.
        """
        if self._browser is not None:
            _log.debug("DerekBrowserClient.start() called on already-started client")
            return

        if sync_playwright is None:
            raise ImportError(
                "playwright is not installed. Run: "
                "uv pip install playwright && python -m playwright install chromium"
            )

        mode = "headed" if not self._headless else "headless"
        _log.info("Starting browser session [%s]: %s", mode, self._purpose)
        if not self._headless:
            _log.info(
                "Browser window will open on $DISPLAY. "
                "slow_mo=%dms between actions.", self._slow_mo
            )
        self._activity_log.log_session_start(self._purpose)

        # Arm session hard-kill timeout
        signal.signal(signal.SIGALRM, _session_timeout_handler)
        signal.alarm(SESSION_TIMEOUT_SECONDS)

        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(
                headless=self._headless,
                args=_SECURE_CHROMIUM_ARGS,
                slow_mo=self._slow_mo,
            )
            self._context = self._browser.new_context()
            self._context.set_default_timeout(OPERATION_TIMEOUT_MS)
            self._context.set_default_navigation_timeout(OPERATION_TIMEOUT_MS)
            self._page = self._context.new_page()
            _log.info("Browser session started successfully [%s]", mode)
        except Exception as exc:
            # Disarm alarm and clean up partial state before re-raising
            signal.alarm(0)
            self._activity_log.log_error("", f"Failed to start browser: {exc}")
            self._cleanup()
            raise RuntimeError(f"Browser failed to start: {exc}") from exc

    def stop(self) -> None:
        """Stop the browser session and release all resources.

        Disarms the session timeout, closes the browser, and writes the
        session_end log entry. Idempotent — safe to call multiple times.
        """
        signal.alarm(0)  # Cancel hard-kill timer
        self._activity_log.log_session_end(self._purpose)
        self._cleanup()
        _log.info("Browser session stopped: %s", self._purpose)

    # ------------------------------------------------------------------
    # Browser actions
    # ------------------------------------------------------------------

    def navigate(self, url: str, purpose: str = "") -> None:
        """Navigate the browser to a URL.

        Args:
            url: Full URL to navigate to.
            purpose: Reason for this navigation, written to the audit log.
                Defaults to the session purpose if not supplied.

        Raises:
            RuntimeError: If the client has not been started.
            Exception: Re-raises Playwright navigation errors after logging.
        """
        self._assert_started()
        nav_purpose = purpose or self._purpose
        _log.info("  → navigate  %s  (%s)", url, nav_purpose)
        self._activity_log.log_navigation(url, nav_purpose)

        try:
            self._page.goto(url, wait_until="domcontentloaded")
            _log.info("  ✓ loaded    %s  title=%r", url, self._page.title())
        except Exception as exc:
            _log.error("  ✗ failed    %s  error=%s", url, exc)
            self._activity_log.log_error(url, str(exc))
            raise

    def screenshot(self, filename: str, url: str = "") -> Path:
        """Capture a screenshot of the current page.

        Args:
            filename: Filename (or full path) to save the screenshot.
                If relative, it is resolved against screenshot_dir.
            url: URL being captured, for the audit log. Falls back to the
                current page URL if not supplied.

        Returns:
            Resolved Path of the saved screenshot.

        Raises:
            RuntimeError: If the client has not been started.
            Exception: Re-raises Playwright screenshot errors after logging.
        """
        self._assert_started()
        dest = (
            Path(filename)
            if Path(filename).is_absolute()
            else self._screenshot_dir / filename
        )
        dest.parent.mkdir(parents=True, exist_ok=True)

        current_url = url or self._page.url
        _log.info("  → screenshot %s  (%s)", dest.name, current_url)

        try:
            self._page.screenshot(path=str(dest), full_page=True)
        except Exception as exc:
            _log.error("  ✗ screenshot failed: %s", exc)
            self._activity_log.log_error(current_url, f"Screenshot failed: {exc}")
            raise

        self._activity_log.log_screenshot(str(dest), current_url)
        _log.info("  ✓ saved      %s", dest)
        return dest

    def page_title(self) -> str:
        """Return the title of the current page.

        Returns:
            Page title string, or empty string if not available.

        Raises:
            RuntimeError: If the client has not been started.
        """
        self._assert_started()
        return self._page.title()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "DerekBrowserClient":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_started(self) -> None:
        """Raise RuntimeError if the browser has not been started."""
        if self._browser is None:
            raise RuntimeError(
                "Browser session not started. Call start() or use as a context manager."
            )

    def _cleanup(self) -> None:
        """Close Playwright objects in reverse-creation order."""
        for attr, method in [
            ("_page", "close"),
            ("_context", "close"),
            ("_browser", "close"),
        ]:
            obj = getattr(self, attr)
            if obj is not None:
                try:
                    getattr(obj, method)()
                except Exception as exc:
                    _log.warning("Error closing %s: %s", attr, exc)
                finally:
                    setattr(self, attr, None)

        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception as exc:
                _log.warning("Error stopping playwright: %s", exc)
            finally:
                self._playwright = None
