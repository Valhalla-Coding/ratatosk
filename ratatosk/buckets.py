import math

BUCKETS = (8192, 16384, 32768, 65536, 131072)
SHRINK_WINDOW_SECONDS = 600


def needed_ctx(prompt_tokens: int, max_tokens: int) -> int:
    return math.ceil(1.25 * (prompt_tokens + max_tokens))


def pick_bucket(needed: int, current: int | None, recent: list[tuple[float, int]], now: float,
                limit: int = 131072) -> int | None:
    """Context size to load, or None if `needed` is bigger than the largest allowed bucket.

    `recent` holds (timestamp, needed) of earlier requests. A reload costs a minute or more, so the
    size only grows when it must, and shrinks (two buckets at a time) only when everything in the last
    ten minutes would have fitted there."""
    allowed = [b for b in BUCKETS if b <= limit]
    if not allowed or needed > allowed[-1]:
        return None
    fitting = next(b for b in allowed if b >= needed)
    if current is None or current not in allowed or needed > current:
        return fitting
    index = allowed.index(current)
    if index >= 2:
        lower = allowed[index - 2]
        window = [n for t, n in recent if now - t <= SHRINK_WINDOW_SECONDS]
        if needed <= lower and all(n <= lower for n in window):
            return lower
    return current
