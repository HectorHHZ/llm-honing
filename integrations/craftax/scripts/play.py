"""Start the upstream full Craftax human interface in a graphical session."""

import os
import platform


def main() -> None:
    if platform.system() == "Linux" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise SystemExit(
            "Human play needs a graphical Linux session. "
            "On a headless VM, use scripts/match.py to record the official pixel video."
        )
    from craftax.craftax.play_craftax import entry_point

    entry_point()


if __name__ == "__main__":
    main()
