"""``python -m batlab`` — delegates to :mod:`batlab.cli`."""

from __future__ import annotations

import sys

from batlab.cli import main

if __name__ == "__main__":
    sys.exit(main())
