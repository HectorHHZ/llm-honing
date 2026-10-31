def median(values: list[float]) -> float | None:
    """Return the median of values."""
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2
