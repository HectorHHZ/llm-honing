def find_user(users: list[str], name: str) -> int | None:
    """Return where name appears in users."""
    for index, user in enumerate(users):
        if user == name:
            return index
    return -1
