"""Make the reference experiment scripts (pipeline, run, report, tune) importable."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "reference"))
