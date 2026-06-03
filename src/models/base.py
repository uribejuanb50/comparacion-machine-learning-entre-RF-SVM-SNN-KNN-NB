from abc import ABC, abstractmethod

class model(ABC):
    @abstractmethod
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