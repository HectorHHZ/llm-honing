def clamp_index(i: int, n: int) -> int | None:
    """Move i into the indices of a sequence of length n."""
    return max(0, min(i, n - 1))
