from src.models.base import model

import torch
import torch.nn as nn
import torch.nn.functional as F


class ANNmodel(model) :
    def __init__(self, capas_ocultas, dropout, learning_rate, batch_size, epochs, patience)

class NnModule(nn.Module) :
    def __init__(self, n_features, n_classes, capas_ocultas, dropout) :

        self.lineales = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        entrada_actual = n_features

        for neuronas_ocultas in capas_ocultas :

            self.lineales.append(nn.Linear(entrada_actual, neuronas_ocultas))
            self.dropouts.append(nn.Dropout(dropout))

            entrada_actual = neuronas_ocultas

        self.salida = nn.Linear(entrada_actual, n_classes)
        
        self.relu = nn.ReLU()

    def forward(self, X) :

        for capa_lineal, capa_dropouts in zip(self.lineales, self.dropouts) :

            x = capa_lineal(x)
            x = self.relu(x)
            x = capa_dropouts(x)

        logits = self.salida(x)

        return logits



