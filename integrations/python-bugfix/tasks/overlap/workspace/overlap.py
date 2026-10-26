def overlap(a_lo: int, a_hi: int, b_lo: int, b_hi: int) -> int | None:
    """Return how much the two ranges overlap."""
    return max(0, min(a_hi, b_hi) - max(a_lo, b_lo) + 1)
