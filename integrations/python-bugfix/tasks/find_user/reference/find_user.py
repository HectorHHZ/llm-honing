def find_user(users: list[str], name: str) -> int | None:
    """Return where name appears in users."""
    wanted = name.strip().casefold()
    for index, user in enumerate(users):
        if user.strip().casefold() == wanted:
            return index
    return None
