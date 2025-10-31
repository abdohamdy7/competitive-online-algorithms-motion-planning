# src/motion_planning/utils/paths.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Commonly used directories
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"
GRAPH_DIR = RESULTS_DIR / "graphs"
DATA_DIR = PROJECT_ROOT / RESULTS_DIR / "data"
FIGURES_DIR = PROJECT_ROOT / RESULTS_DIR / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"
