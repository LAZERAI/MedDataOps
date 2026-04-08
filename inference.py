from __future__ import annotations

import sys
from typing import Any


def _safe_log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _emit_fallback_end_line() -> None:
    print(
        "[END] run_id=fallback mean_score=0.000000 total_elapsed_s=0.000 statuses=error",
        flush=True,
    )


def main() -> None:
    try:
        from inference_impl import main as impl_main  # Local import to guard import-time errors.

        impl_main()
    except BaseException as exc:  # pragma: no cover - validator stability fallback
        _safe_log(f"[fatal] Unhandled inference exception: {exc}")
        _emit_fallback_end_line()


if __name__ == "__main__":
    # Always exit 0; success/failure is encoded via structured logs.
    try:
        main()
    except BaseException as exc:  # pragma: no cover - hard stop guard
        _safe_log(f"[fatal] Unhandled inference exception: {exc}")
        _emit_fallback_end_line()
    sys.exit(0)
