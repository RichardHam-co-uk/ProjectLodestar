"""Browser automation module for ProjectLodestar.

Provides a hardened, auditable Playwright/Chromium wrapper for Derek's
browser automation tasks (usage dashboard scraping, AI UI interaction).

Security model:
- All navigations logged to ~/.lodestar/browser-activity.jsonl
- Mandatory hardened Chromium launch args (per docs/BROWSER_SECURITY.md)
- 5-minute hard session kill via signal.alarm
- Headless-only via Xvfb virtual display
"""

from modules.browser.logger import BrowserActivityLogger
from modules.browser.client import DerekBrowserClient

__all__ = ["BrowserActivityLogger", "DerekBrowserClient"]
