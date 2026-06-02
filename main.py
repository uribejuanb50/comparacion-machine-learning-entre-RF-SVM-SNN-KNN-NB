import src.data.loader as loader

from src.config import (DATASET1,
                        DATASET2,
                        ESPERADOS)

def main():

    #dataframe = loader.leer_archivo(DATASET1, ESPERADOS)
    #print(dataframe)
    dataframe = loader.leer_archivo(DATASET2, ESPERADOS)
    print(dataframe)
    pass

if __name__ == "__main__" :
    main()