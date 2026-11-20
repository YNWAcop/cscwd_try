"""Paths are relative to this checkout, regardless of the working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
AUDIT = DATA / "audit"
DESIGN = DATA / "design"
RESULTS = ROOT / "results"
OUTPUT = ROOT / "runs"
OUTPUT.mkdir(exist_ok=True)
