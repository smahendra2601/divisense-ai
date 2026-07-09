"""Tier 4 — Presentation: CLI entry point.

Usage:
    python forecast.py "Will Infosys increase its dividend next quarter?"
    python forecast.py ITC

Runs the same LangGraph pipeline as the Streamlit app and prints the
final markdown report (with timestamp and disclaimer) to stdout. Errors
at any node are handled inside the graph and surface as a friendly
message rather than a stack trace.
"""

from __future__ import annotations

import logging
import sys


def main(argv: list[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    # Quiet the noisy third-party INFO logs; keep warnings.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if not argv:
        print('Usage: python forecast.py "<ticker or question>"')
        print('Examples:')
        print('  python forecast.py ITC')
        print('  python forecast.py "Will Infosys increase its dividend next quarter?"')
        return 1

    user_query = " ".join(argv).strip()

    # Import here so `--help`-style misuse above doesn't pay the heavy import cost.
    from src.graph import run_pipeline

    try:
        final_state = run_pipeline(user_query)
    except Exception as exc:  # noqa: BLE001 - last-resort guard; graph handles node errors itself
        print(f"⚠️  Sorry, something went wrong: {exc}")
        return 1

    print("\n" + (final_state.get("final_report") or "(no report produced)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
