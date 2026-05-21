import traceback
from pathlib import Path
from typing import Any

from .io_utils import JsonObject, write_json


def checkpoint_status(exc: BaseException) -> str:
    return "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"


def error_to_json(exc: BaseException) -> JsonObject:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "is_keyboard_interrupt": isinstance(exc, KeyboardInterrupt),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=12)),
    }


def save_checkpoint(path: Path, payload: Any) -> None:
    try:
        write_json(path, payload)
    except Exception as save_exc:  # pragma: no cover - best effort during failure handling.
        print(
            f"[checkpoint-save-failed] path={path} "
            f"error={save_exc.__class__.__name__}: {save_exc}",
            flush=True,
        )
        return
    print(f"[checkpoint-saved] {path}", flush=True)
