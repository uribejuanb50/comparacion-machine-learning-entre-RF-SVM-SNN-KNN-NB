from src.models.base import model

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy as cpy

import numpy as np

from torch.utils.data import TensorDataset, DataLoader


class ANNmodel(model) :
    def __init__(self, capas_ocultas, dropout, learning_rate, batch_size, epochs, patience, semilla = 42,
                 device = "cuda" if torch.cuda.is_available() else "cpu") :

        torch.manual_seed(semilla)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.capas_ocultas = capas_ocultas
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = device

    def fit(self, x_train, y_train, x_validar = None, y_validar = None):

        x_train_input = torch.FloatTensor(x_train.toarray())
        y_train_targets = torch.FloatTensor(y_train)

        dataset = TensorDataset(x_train_input, y_train_targets)

        data_loader = DataLoader(dataset, self.batch_size, shuffle = True)

        n_classes = 1

        nnModule = NnModule(x_train.shape[1],
                            n_classes,
                            self.capas_ocultas,
                            self.dropout).to(self.device)
        
        criterio = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(params = nnModule.parameters(),
                                     lr = self.learning_rate
                                     )

        tiempos_en_0 = 0
        mejor_loss_validar = float('inf')

        for epoca in self.epochs :

            nnModule.train()

            acum_perdida = 0
            loss_validar = 0

            for batch_x, batch_y in data_loader :

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                outputs = nnModule(batch_x)
                perdida = criterio(outputs, batch_y)
                perdida.backward()

                optimizer.step()

                acum_perdida += perdida.item()

            nnModule.eval()

            with torch.no_grad() :

                x_validar_input = torch.FloatTensor(x_validar.toarray()).to(self.device)
                y_validar_target = torch.FloatTensor(y_validar).to(self.device)

                outputs_validar = nnModule(x_validar_input)
                loss_validar = criterio(outputs_validar, y_validar_target)

                #Vamos aquí  

            if loss_validar < mejor_loss_validar :

                mejor_loss_validar = loss_validar
                tiempos_en_0 = 0

                mejores_pesos = cpy.deepcopy(nnModule.state_dict())
            
            else :
                tiempos_en_0 += 1

            if tiempos_en_0 >= self.patience :
                break

            self.modelo = nnModule
            nnModule.load_state_dict(mejores_pesos)
                
        return
    
    def predict(self, x_test) :
        self.modelo.eval()

        with torch.no_grad() :

            x_test_input = torch.FloatTensor(x_test.toarray()).to(self.device)

            outputs_predict = self.modelo(x_test_input)
            outputs_predict = torch.sigmoid(outputs_predict)

            prediccion = (outputs_predict >= 0.5).int()

            return prediccion.cpu().numpy()



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



