def clamp(value: float, low: float, high: float) -> float:
    """Return ``value`` limited to the range ``[low, high]``."""
    return max(low, min(high, value))
