"""CLI entry point for computing SafeAlert metrics."""

from __future__ import annotations

try:
    from metrics import main
except ModuleNotFoundError:
    from scripts.metrics import main


if __name__ == "__main__":
    main()
