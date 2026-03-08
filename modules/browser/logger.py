"""Audit logger for Derek's browser automation activity.

Every navigation, screenshot, error, and session boundary is appended as
a JSON line to ~/.lodestar/browser-activity.jsonl. This log is the audit
trail for all browser sessions and is reviewed per the weekly checklist
in docs/BROWSER_SECURITY.md.

Usage::

    logger = BrowserActivityLogger()
    logger.log_session_start()
    logger.log_navigation("https://chat.openai.com", "Check usage dashboard")
    logger.log_screenshot("openai-usage.png", "https://chat.openai.com")
    logger.log_session_end()
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_module_logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = "~/.lodestar/browser-activity.jsonl"


class BrowserActivityLogger:
    """Append-only JSONL audit log for browser automation sessions.

    Each call appends a single JSON object followed by a newline to the
    configured log file. The log directory is created on first use if it
    does not exist.

    Args:
        log_path: Path to the JSONL log file. Tilde expansion is applied.
            Defaults to ~/.lodestar/browser-activity.jsonl.
    """

    def __init__(self, log_path: str = DEFAULT_LOG_PATH) -> None:
        self._log_path = Path(log_path).expanduser()
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def log_path(self) -> Path:
        """Resolved path to the active log file."""
        return self._log_path

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def log_session_start(self, purpose: str = "") -> None:
        """Record the start of a browser session.

        Args:
            purpose: Human-readable description of why the session was started.
        """
        self._append({
            "action": "session_start",
            "purpose": purpose,
        })

    def log_session_end(self, purpose: str = "") -> None:
        """Record the end of a browser session.

        Args:
            purpose: Matches the purpose given to log_session_start, for
                easy correlation in the log.
        """
        self._append({
            "action": "session_end",
            "purpose": purpose,
        })

    def log_navigation(self, url: str, purpose: str) -> None:
        """Record a page navigation.

        Args:
            url: The full URL navigated to.
            purpose: Mandatory human-readable reason for this navigation.
                Must not be empty — callers who cannot supply a purpose
                should use 'unspecified' rather than leaving it blank.
        """
        if not purpose:
            _module_logger.warning(
                "log_navigation called without a purpose for URL %s", url
            )
        self._append({
            "action": "navigate",
            "url": url,
            "purpose": purpose,
        })

    def log_screenshot(self, filename: str, url: str) -> None:
        """Record a screenshot capture.

        Args:
            filename: Local path or filename of the saved screenshot.
            url: The URL of the page that was captured.
        """
        self._append({
            "action": "screenshot",
            "filename": filename,
            "url": url,
        })

    def log_error(self, url: str, error: str) -> None:
        """Record a browser error or unexpected exception.

        Args:
            url: The URL being accessed when the error occurred.
            error: String representation of the error.
        """
        _module_logger.error("Browser error at %s: %s", url, error)
        self._append({
            "action": "error",
            "url": url,
            "error": error,
        })

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append(self, fields: dict) -> None:
        """Write a timestamped JSON line to the log file.

        Args:
            fields: Dict of event-specific fields. A 'timestamp' key is
                automatically injected in UTC ISO-8601 format.
        """
        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **fields,
        }
        try:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError as exc:
            _module_logger.error(
                "Failed to write browser activity log entry: %s", exc
            )
