def reverse_words(text: str) -> str:
    """Return the words of ``text`` in reverse order, separated by single spaces."""
    return " ".join(reversed(text.split()))
