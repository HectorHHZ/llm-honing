def count_between(items: list[int], start: int, stop: int) -> int | None:
    """Count the items from start to stop."""
    if start > stop:
        return None
    return sum(1 for item in items if start <= item < stop)
