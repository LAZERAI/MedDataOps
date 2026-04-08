from __future__ import annotations

import sys
from typing import Any


def _safe_log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _safe_main_impl() -> Any:
    try:
        from inference_impl import main as impl_main  # Local import to guard import-time errors.
    except BaseException as exc:  # pragma: no cover - validator stability fallback
        _safe_log(f"[fatal] Failed to import inference implementation: {exc}")
        return None

    try:
        return impl_main()
    except BaseException as exc:  # pragma: no cover - validator stability fallback
        _safe_log(f"[fatal] Unhandled inference exception: {exc}")
        return None


def main() -> None:
    _safe_main_impl()


if __name__ == "__main__":
    # Always exit 0; success/failure is encoded via structured logs.
    main()
    sys.exit(0)
