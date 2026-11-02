def count_tag(tags: list[str], tag: str) -> int | None:
    """Count how often tag appears in tags."""
    return sum(1 for item in tags if item == tag)
