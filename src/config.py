from pathlib import Path

RAIZ = Path(__file__).resolve().parent[1]

DATA = RAIZ / "data"
DATA_PROCESSED = DATA / "processed"
DATA_RAW = DATA / "raw"

REPORTS = RAIZ / "reports"
FIGURES = REPORTS / "figures"
METRICS = REPORTS / "metrics"

MODELS = RAIZ / "models"

SEMILLA = 42