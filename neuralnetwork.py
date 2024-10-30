import json
import random
import time
from operator import itemgetter

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from model.mlp.classes.KinematicsModel import KinematicsModel
from utils.simulator import plotSimulation
from utils.utils import read_data, plot_metric


def random_search(X, Y, input_size, output_size, num_epochs, num_trials=20, kinematics='inverse'):
    all_results = []

    start_time = time.perf_counter()
    for trial in range(num_trials):
        print('Numero trial:', trial)
        # Genera iperparametri casuali
        hidden_sizes = [random.choice([64, 128, 256, 512]) for _ in range(random.randint(2, 5))]
        learning_rate = random.choice([0.1, 1e-2, 1e-3, 1e-4])
        batch_size = random.choice([32, 64, 128, 256])
        dropout_rate = random.uniform(0.1, 0.5)
        weight_decay = random.choice([0, 1e-4, 1e-5, 1e-6])
        optimization = random.choice(['Adam'])
        print('Generati parametri: hidden size: ', hidden_sizes, '- lr: ', learning_rate, '- batch_size:', batch_size,
              '- dropout: ', dropout_rate, '- decay: ', weight_decay, '- optimizer: ', optimization)

        # Crea e addestra il modello
        model = KinematicsModel(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size,
                                dropout_rate=dropout_rate)
        if optimization == 'Adam':
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train)),
                                  batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = nn.MSELoss()(outputs, targets)
                loss.backward()
                optimizer.step()

        # Valuta il modello
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(X_val))
            val_loss = nn.MSELoss()(val_outputs, torch.FloatTensor(Y_val))

        print(f"Trial {trial + 1}/{num_trials}: Val Loss: {val_loss:.4f}")

        best_params = {
            'hidden_sizes': hidden_sizes,
            'optimization': optimization,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'dropout_rate': dropout_rate,
            'val_loss': val_loss.item()
        }
        all_results.append(best_params)

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Il modello è stato addestrato in {execution_time:.4f} secondi.")

    # Ordina i risultati per validation loss
    sorted_results = sorted(all_results, key=itemgetter('val_loss'))

    # Salva i risultati ordinati in un file JSON
    with open('search/random_search_results_' + kinematics + '.json', 'w') as f:
        json.dump(sorted_results, f, indent=2)


def train_final_model(X_train, X_val, Y_train, Y_val, best_params, input_size, output_size, num_epochs, kinematics,
                      model=None, normalize=True):
    if model is None:
        model = KinematicsModel(input_size=input_size, hidden_sizes=best_params['hidden_sizes'],
                                output_size=output_size,
                                dropout_rate=best_params['dropout_rate'], kinematics=kinematics)

    if best_params['optimization'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'],
                               weight_decay=best_params['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=best_params['learning_rate'],
                              weight_decay=best_params['weight_decay'])
    criterion = nn.MSELoss()

    if normalize:
        input_scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_normalized = input_scaler.fit_transform(X_train)
        X_val_normalized = input_scaler.transform(X_val)

        # Normalizzazione dell'output (se necessario)
        output_scaler = MinMaxScaler(feature_range=(0, 1))
        Y_train_normalized = output_scaler.fit_transform(Y_train)
        Y_val_normalized = output_scaler.fit_transform(Y_val)

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train_normalized), torch.FloatTensor(Y_train_normalized)),
            batch_size=best_params['batch_size'], shuffle=True, drop_last=True)

        # salvo gli scaler per l'inferenza
        joblib.dump(input_scaler, 'config/input_scaler.pkl')
        joblib.dump(output_scaler, 'config/output_scaler.pkl')

    else:
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train)),
                                  batch_size=best_params['batch_size'], shuffle=True, drop_last=True)

    train_losses = []
    val_losses = []
    epochs_time = []
    train_start_time = time.perf_counter()
    model.train()
    for epoch in range(num_epochs):
        start_time = time.perf_counter()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to('cpu'), batch_y.to('cpu')

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(X_val))
            val_loss = criterion(val_outputs, torch.FloatTensor(Y_val)).item()

        val_losses.append(val_loss)
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss}')
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        epochs_time.append(execution_time)
    train_end_time = time.perf_counter()
    train_execution_time = train_end_time-train_start_time

    return model, train_losses, val_losses, epochs_time, train_execution_time


def load_best_param(filename):
    with open(filename, 'r') as f:
        results = json.load(f)

    if results:
        best_trial = results[0]  # Il primo elemento è il migliore (loss più bassa)
        best_params = {
            'hidden_sizes': best_trial['hidden_sizes'],
            'learning_rate': best_trial['learning_rate'],
            'batch_size': best_trial['batch_size'],
            'dropout_rate': best_trial['dropout_rate'],
            'optimization': best_trial['optimization'],
            'weight_decay': best_trial['weight_decay']
        }

        return best_params
    else:
        print(f"Il file {filename} è vuoto.")
        return None


def final_training(X, Y, kinematics='inverse', normalize=True):
    # Addestra il modello finale con i migliori iperparametri
    X_train, Y_train, X_val, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    best_params = load_best_param('search/random_search_results_' + kinematics + '.json')
    final_model, train_losses, val_losses, execution_time, train_execution_time = train_final_model(X_train, Y_train, X_val, Y_val,
                                                                              best_params, 3, 3,
                                                                              200, normalize=normalize,
                                                                              kinematics=kinematics)

    # Salva il modello e i risultati
    torch.save(final_model.state_dict(), 'model/mlp/final_model_' + kinematics + '.pth')
    np.savez('model/metrics/final_mlp-' + kinematics + '-_training_metrics.npz', train_losses=train_losses,
             val_losses=val_losses, epoch_execution_time=execution_time, train_execution_time = train_execution_time)

    plot_metric(train_losses, val_losses, 'Loss',
                'model/mlp/plot/final_training_' + kinematics + '_loss.png')


def inference(X, kinematics, normalized=False):
    # Carica il modello salvato
    best_params = load_best_param('search/random_search_results_' + kinematics + '.json')
    model = KinematicsModel(input_size=3, hidden_sizes=best_params['hidden_sizes'], output_size=3,
                            dropout_rate=best_params['dropout_rate'])
    if normalized:
        model.load_state_dict(torch.load('model/mlp/final_model_normalized' + kinematics + '.pth'))
    else:
        model.load_state_dict(torch.load('model/mlp/final_model_' + kinematics + '.pth'))
    model.eval()  # Imposta il modello in modalità di valutazione

    # carico gli scaler di normalizzazione
    if normalized:
        input_scaler = joblib.load('config/input_scaler.pkl')
        output_scaler = joblib.load('config/output_scaler.pkl')

        X_normalized = input_scaler.fit_transform(X)
        X_test_tensor = torch.FloatTensor(X_normalized)
        with torch.no_grad():
            y_pred_normalized = model(X_test_tensor).numpy()
            y_pred = output_scaler.inverse_transform(y_pred_normalized)
    else:
        X_test_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            y_pred = model(X_test_tensor).numpy()

    return y_pred


def main():
    kinematics = 'inverse'

    ###### TRAINING MODEL ####################
    # # Usa la ricerca casuale per trovare i migliori iperparametri
    X, Y = read_data('datasets/dataset.txt', kinematics)
    # # # random_search(X, Y, 3, 3, 200, 20, kinematics)
    final_training(X, Y, kinematics, normalize=False)

    #### INFERENCE ########################
    # expected = [0., 0., 0.1]
    # prediction = inference(np.array([[.1, .1, .1]]), kinematics, normalized=False)
    # print("Test Results:")
    # if kinematics == 'inverse':
    #     print(f"True actuation: {expected}")
    #     print(f"Predicted actuation: {prediction}")
    #     plotSimulation(prediction)
    # else:
    #     # print(f"Input (x, y, z):, {poses} ")
    #     print(f"Ture poses: {expected}")
    #     print(f"Predicted poses: {prediction}")


# Test del modello
if __name__ == "__main__":
    main()
