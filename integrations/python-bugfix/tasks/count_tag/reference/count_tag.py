def count_tag(tags: list[str], tag: str) -> int | None:
    """Count how often tag appears in tags."""
    wanted = tag.strip().casefold()
    if not wanted:
        return None
    return sum(1 for item in tags if item.strip().casefold() == wanted)
