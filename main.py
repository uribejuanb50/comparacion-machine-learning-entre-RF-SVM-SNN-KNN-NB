import src.data.loader as loader
import src.data.preprocessor as prepocessor

from src.config import (DATASET1,
                        DATASET2,
                        ESPERADOS,
                        TRAD_NAN,
                        COMESTIBLES)

def main():

    #dataframe = loader.leer_archivo(DATASET1, ESPERADOS)
    #print(dataframe)
    dataframe = loader.leer_archivo(DATASET2, ESPERADOS)
    
    dataframe = dataframe.fillna("none")
    dataframe = prepocessor.codificar_columna(dataframe, "class", COMESTIBLES, int)


    print(dataframe)
    pass

if __name__ == "__main__" :
    main()