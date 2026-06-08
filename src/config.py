from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

DATA = RAIZ / "data"
DATA_PROCESSED = DATA / "processed"
DATA_RAW = DATA / "raw"
DATASET = DATA_RAW / "adult.data"


REPORTS = RAIZ / "reports"
FIGURES = REPORTS / "figures"
METRICS = REPORTS / "metrics"

MODELS = RAIZ / "models"

SEMILLA = 42

#Manejo de datos:
#Manejo de datos:
#Manejo de datos:
ESPERADOS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income"
]

COLS_A_DROPEAR = [
    "fnlwgt",       # peso muestral del censo, no es predictivo
    "education",    # redundante con education-num (info ordinal mejor preservada)
]

CARACTERISTICAS = [
    "age",
    "workclass",
    #"fnlwgt",          # dropped
    #"education",       # dropped, usamos education-num
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]

OBJETIVO = ["income"]

# Lista de variables categóricas (nominales)
CATEGORICAS = [
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Lista de variables numéricas
NUMERICAS = [
    "age",
    "education-num",
    "hours-per-week",
]

LOG_TRANSFORM = [
    "capital-gain",
    "capital-loss",
]

TRAD_NAN = {"?": "none", " ?": "none"}   # Adult usa "?" como missing

INGRESO = {" >50K": 1, " <=50K": 0}        # antes COMESTIBLES; renombra en main.py


#MODELS

#RF--------------------------------------
HIPERPARAMETROS_RF = {
        'n_estimators': 300,
        'max_depth': 5,
        'learning_rate': 0.05,
        'random_state': SEMILLA,
        'subsample': 0.8
    }


#ANN-------------------------
CAPAS_OCULTAS = [128, 64, 32]

HIPERPARAMETROS_ANN = {
    "capas_ocultas": CAPAS_OCULTAS,
    "dropout": 0.3,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 50,
    "patience": 10,
    # semilla y device los puedes omitir (usan defaults) o sobreescribir:
    "semilla": SEMILLA,
    # "device": "cuda",
}


#KNN-----------------------------------

K_GRIDSEARCH = [5, 10, 25, 50, 100]

KNN_GRID = {
    "n_neighbors": K_GRIDSEARCH,       # Cambia esto por tu variable K_GRIDSEARCH
    "weights": ["uniform", "distance"], 
    "metric": ["minkowski"],            
    "p": [1, 2],                 
}

HIPERPARAMETROS_KNN = {
    "algorithm" : "brute",
    "n_jobs" : 1,
    "semilla" : SEMILLA,
    "n_muestras" : 8000,
    "dict" : KNN_GRID
}