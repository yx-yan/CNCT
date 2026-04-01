"""Training entry-point with clean log output.

Configures logging before pytorch3dunet imports to:
  - Eliminate duplicate log lines (pytorch3dunet adds its own handlers)
  - Suppress per-iteration spam (only log every log_after_iters)
  - Hide DEBUG noise (h5py converters, etc.)
  - Suppress the giant config-dict dump
"""
import logging
import re
import sys

# ── Logging setup (must precede any pytorch3dunet import) ────────────────────

_fmt = logging.Formatter(
    fmt="%(asctime)s  %(levelname)-5s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_fmt)

logging.root.setLevel(logging.INFO)
logging.root.addHandler(_handler)


class _IterationFilter(logging.Filter):
    """Only pass 'Training iteration [N/M]' lines every 500 iterations."""

    _pat = re.compile(r"Training iteration \[(\d+)/")

    def filter(self, record):
        m = self._pat.search(record.getMessage())
        if m:
            return int(m.group(1)) % 500 == 0
        # Suppress the huge config dict dump
        if record.getMessage().startswith("{") and "model" in record.getMessage():
            return False
        return True


_handler.addFilter(_IterationFilter())


# ── Import and patch pytorch3dunet logging ───────────────────────────────────

from pytorch3dunet.train import main  # noqa: E402
from pytorch3dunet.unet3d import utils as _u3d_utils  # noqa: E402

# Remove per-logger handlers added by get_logger() — root handler is enough.
# This prevents every message from appearing twice.
for _lg in _u3d_utils.loggers.values():
    _lg.handlers.clear()

# ── Run ──────────────────────────────────────────────────────────────────────
main()
