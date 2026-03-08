#!/usr/bin/env python3
"""Headed-mode browser demo — opens a visible browser window on screen.

Run from the ProjectLodestar root with the venv active:

    cd ~/Documents/github-repos/ProjectLodestar
    source .venv/bin/activate
    python scripts/demo_browser.py

You should see a Chromium window open, navigate to Perplexity, then close.
Screenshots are saved to ~/Desktop/ for easy viewing.
"""

import logging
import sys
from pathlib import Path

# Make sure the repo root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.browser import DerekBrowserClient

# Show INFO-level logs in the terminal so you can follow what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

SCREENSHOT_DIR = Path.home() / "Desktop"

def main() -> None:
    print("\n" + "=" * 60)
    print("Derek Browser Demo — headed mode")
    print("A Chromium window will open. Watch it navigate.")
    print("=" * 60 + "\n")

    with DerekBrowserClient(
        purpose="headed demo",
        headless=False,          # show the browser window
        slow_mo=800,             # 800ms pause between actions — easy to follow
        screenshot_dir=str(SCREENSHOT_DIR),
    ) as c:
        # Perplexity
        c.navigate("https://perplexity.ai", "demo nav 1")
        c.screenshot("demo-perplexity.png")

        # Claude
        c.navigate("https://claude.ai", "demo nav 2")
        c.screenshot("demo-claude.png")

    print("\n✅ Done. Screenshots saved to ~/Desktop/")
    print("   demo-perplexity.png")
    print("   demo-claude.png\n")


if __name__ == "__main__":
    main()
