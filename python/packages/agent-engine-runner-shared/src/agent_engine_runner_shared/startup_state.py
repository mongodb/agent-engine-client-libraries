"""Process startup state, marked only after the listener binds."""

__all__ = ["has_startup_completed", "mark_startup_complete", "reset_startup_state"]

_startup_complete = False


def has_startup_completed() -> bool:
    return _startup_complete


def mark_startup_complete() -> None:
    global _startup_complete
    _startup_complete = True


def reset_startup_state() -> None:
    global _startup_complete
    _startup_complete = False
