"""Thin wrapper around ``cnct_dataprep.cli.evaluation:main``.

Equivalent to the ``cnct-evaluation`` console script installed by
``pip install -e data_prep/``. Prefer the console script when possible.
"""
from __future__ import annotations

import sys

from cnct_dataprep.cli.evaluation import main

if __name__ == "__main__":
    sys.exit(main())
