from abc import ABC, abstractmethod

class model(ABC):
   @abstractmethod
   def __init__(self):
      self.entrenado = False
      self.tiempo_prediccion = 0

   def fit(self, x_train, y_train, x_val = None, y_val = None) :
      pass
   @abstractmethod  
   def predict(self, x_test) :
      pass

   @abstractmethod
   def predict_proba(self, x_test) :
      pass

   @abstractmethod
   def save(self, path) :
      pass

   @abstractmethod
   def load(self, path) :
      pass 
    
   def devolver_tiempo_prediccion(self) :
      tiempo_pred = self.tiempo_prediccion
      return tiempo_pred
   
   def verificar_entrenado(self) :
      if(not self.entrenado) :
         raise RuntimeError(
            "[ERROR] No hubo fit, ¿Cómo piensas predecir?"
         )
        
   
