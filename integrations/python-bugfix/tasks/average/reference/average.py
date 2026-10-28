def average(values: list[float]) -> float | None:
    """Return the average of values."""
    if not values:
        return None
    return sum(values) / len(values)
