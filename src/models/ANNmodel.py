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

        print("[ANNmodel] Comenzando el entrenamiento...")
        print(f"[ANNmodel] Device: {self.device} | Épocas: {self.epochs} | Batch: {self.batch_size}")
        print(f"[ANNmodel] Arquitectura: {self.capas_ocultas} | Dropout: {self.dropout} | LR: {self.learning_rate}")
        print(f"[ANNmodel] size train: x - {x_train.shape[0]}, y - {y_train.shape[0]} | size val: x - {x_validar.shape[0]}, y - {y_validar.shape[0]}")

        x_train_input = torch.FloatTensor(x_train.toarray())
        y_train_targets = torch.FloatTensor(y_train.values).unsqueeze(1)

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
        mejor_epoca = 0
        status = ""

        for epoca in range(self.epochs) :

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
                y_validar_target = torch.FloatTensor(y_validar.values).unsqueeze(1).to(self.device)

                outputs_validar = nnModule(x_validar_input)
                loss_validar = criterio(outputs_validar, y_validar_target)

                #Vamos aquí  

            train_loss_promedio = acum_perdida / len(data_loader)
            val_loss_actual = loss_validar.item()

            if val_loss_actual < mejor_loss_validar :

                mejor_loss_validar = val_loss_actual
                tiempos_en_0 = 0
                mejor_epoca = epoca + 1
                status = "!!Nuevo mejor"

                mejores_pesos = cpy.deepcopy(nnModule.state_dict())
            
            else :
                tiempos_en_0 += 1
                status = f"Sin mejora ({tiempos_en_0} / {self.patience})"

            print(f"Epoca {epoca+1:3d}/{self.epochs} | train_loss: {train_loss_promedio:.4f} | val_loss: {val_loss_actual:.4f} | {status}")

            if tiempos_en_0 >= self.patience :
                print(f"[ANNmodel] Early stopping, {self.patience} sin mejora")
                break

        self.modelo = nnModule
        nnModule.load_state_dict(mejores_pesos)

        print("[ANNmodel] Entrenamiento finalizado")      
        return
    
    def predict(self, x_test) :
        self.modelo.eval()

        with torch.no_grad() :

            x_test_input = torch.FloatTensor(x_test.toarray()).to(self.device)

            outputs_predict = self.modelo(x_test_input)
            outputs_predict = torch.sigmoid(outputs_predict)
            outputs_predict = outputs_predict.squeeze()

            prediccion = (outputs_predict >= 0.5).int()

            return prediccion.cpu().numpy()

    def predict_proba(self, x_test):
        self.modelo.eval()

        with torch.no_grad() :

            x_test_input = torch.FloatTensor(x_test.toarray()).to(self.device)

            outputs_predict = self.modelo(x_test_input)
            outputs_predict = torch.sigmoid(outputs_predict)

            prob_clase_0 = 1 - outputs_predict

            prediccion = torch.cat((prob_clase_0, outputs_predict), dim = 1)

            return prediccion.cpu().numpy()

    def save(self, path) :

        torch.save(self.modelo, path)
        print(f"[ANNmodel] Modelo guardado en {path}")

    def load(self, path) :

        self.modelo = torch.load(path)
        print(f"[ANNmodel] Modelo cargado de {path}")


class NnModule(nn.Module) :
    def __init__(self, n_features, n_classes, capas_ocultas, dropout) :
        super().__init__()

        self.lineales = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        entrada_actual = n_features

        for neuronas_ocultas in capas_ocultas :

            self.lineales.append(nn.Linear(entrada_actual, neuronas_ocultas))
            self.dropouts.append(nn.Dropout(dropout))

            entrada_actual = neuronas_ocultas

        self.salida = nn.Linear(entrada_actual, n_classes)
        
        self.relu = nn.ReLU()

    def forward(self, x) :

        for capa_lineal, capa_dropouts in zip(self.lineales, self.dropouts) :

            x = capa_lineal(x)
            x = self.relu(x)
            x = capa_dropouts(x)

        logits = self.salida(x)

        return logits



