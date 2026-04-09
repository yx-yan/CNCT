"""Thin wrapper around ``cnct_dataprep.cli.projection:main``.

Equivalent to the ``cnct-projection`` console script installed by
``pip install -e data_prep/``. Prefer the console script when possible.
"""
from __future__ import annotations

import sys

from cnct_dataprep.cli.projection import main

if __name__ == "__main__":
    sys.exit(main())
