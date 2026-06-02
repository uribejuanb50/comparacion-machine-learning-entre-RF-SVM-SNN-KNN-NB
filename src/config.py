from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

DATA = RAIZ / "data"
DATA_PROCESSED = DATA / "processed"
DATA_RAW = DATA / "raw"
DATASET1 = DATA_RAW / "primary_data.csv"
DATASET2 = DATA_RAW / "secondary_data.csv"

REPORTS = RAIZ / "reports"
FIGURES = REPORTS / "figures"
METRICS = REPORTS / "metrics"

MODELS = RAIZ / "models"

SEMILLA = 42

ESPERADOS = [
    "class", "cap-diameter", "cap-shape", "cap-surface", "cap-color", 
    "does-bruise-or-bleed", "gill-attachment", "gill-spacing", "gill-color", 
    "stem-height", "stem-width", "stem-root", "stem-surface", "stem-color", 
    "veil-type", "veil-color", "has-ring", "ring-type", "spore-print-color", 
    "habitat", "season"
]
CARACTERISTICAS = ["cap-diameter", "cap-shape", "cap-surface", "cap-color", 
    "does-bruise-or-bleed", "gill-attachment", "gill-spacing", "gill-color", 
    "stem-height", "stem-width", "stem-root", "stem-surface", "stem-color", 
    "veil-type", "veil-color", "has-ring", "ring-type", "spore-print-color", 
    "habitat", "season"]
OBJETIVO = ["class"] 

# Lista de variables categóricas (nominales - 'n')
CATEGORICAS = [
    "cap-shape",
    "cap-surface",
    "cap-color",
    "does-bruise-or-bleed",
    "gill-attachment",
    "gill-spacing",
    "gill-color",
    "stem-root",
    "stem-surface",
    "stem-color",
    "veil-type",
    "veil-color",
    "has-ring",
    "ring-type",
    "spore-print-color",
    "habitat",
    "season"
]

# Lista de variables numéricas (métricas - 'm')
NUMERICAS = [
    "cap-diameter",
    "stem-height",
    "stem-width"
]

TRAD_NAN = { "NaN" : "none"}
COMESTIBLES = { "e" : 1, "p" : 0}
