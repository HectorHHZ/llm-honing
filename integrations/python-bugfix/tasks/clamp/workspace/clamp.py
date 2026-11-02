def clamp(value: float, low: float, high: float) -> float:
    """Return ``value`` limited to the range ``[low, high]``."""
    return max(high, min(low, value))
