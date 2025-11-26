# src/motion_planning/utils/paths.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Commonly used directories
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = RESULTS_DIR / "data"
FIGURES_DIR = RESULTS_DIR / "figures"
GRAPH_DIR = RESULTS_DIR / "graphs"
ROUTES_DIR = RESULTS_DIR / "routes"
LOGS_DIR = PROJECT_ROOT / "logs"

# Offline / online artifact roots
OFFLINE_RESULTS_DIR = DATA_DIR / "offline problems"
ONLINE_RESULTS_DIR = DATA_DIR / "online solutions"
ONLINE_RESULTS_CANDIDATES_DIR = ONLINE_RESULTS_DIR / "candidates"
ONLINE_RESULTS_GRAPH_DIR = ONLINE_RESULTS_DIR / "graph-based"
