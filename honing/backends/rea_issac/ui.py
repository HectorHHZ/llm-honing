"""Compatibility imports for the standalone REA-Issac game."""

import sys

from integrations.rea_issac import ui as _ui
from integrations.rea_issac.ui import ROOT as ROOT
from integrations.rea_issac.ui import Session as Session
from integrations.rea_issac.ui import handler as handler
from integrations.rea_issac.ui import main as main

if __name__ == "__main__":
    main()
else:
    # Share module state so assignments through the old import affect the runtime.
    sys.modules[__name__] = _ui
