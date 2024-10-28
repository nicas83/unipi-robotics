import json
import random
import time
from operator import itemgetter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score

from simulator import plotSimulation
from utils import read_data, prepare_data, plot_metric, generate_sample_data


# Definizione del modello
class KinematicsModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(KinematicsModel, self).__init__()

        input_size = kwargs.get('input_size', 10)
        hidden_sizes = kwargs.get('hidden_sizes', [64, 64])
        output_size = kwargs.get('output_size', 1)
        dropout_rate = kwargs.get('dropout_rate', 0.2)

        layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.Tanh())
            layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


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


def train_final_model(X, Y, best_params, input_size, output_size, num_epochs, model=None):
    start_time = time.perf_counter()
    if model is None:
        model = KinematicsModel(input_size=input_size, hidden_sizes=best_params['hidden_sizes'],
                                output_size=output_size,
                                dropout_rate=best_params['dropout_rate'])

    if best_params['optimization'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'],
                               weight_decay=best_params['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=best_params['learning_rate'],
                              weight_decay=best_params['weight_decay'])
    criterion = nn.MSELoss()

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train)),
                              batch_size=best_params['batch_size'], shuffle=True, drop_last=True)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_losses = []
        epoch_train_loss = 0.0
        epoch_train_outputs = []
        epoch_train_targets = []

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())
            epoch_train_loss += loss.item() * inputs.size(0)
            epoch_train_outputs.append(outputs.detach())
            epoch_train_targets.append(targets)

        epoch_train_loss /= len(X_train)
        train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(X_val))
            val_loss = criterion(val_outputs, torch.FloatTensor(Y_val)).item()

        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        # train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Il modello è stato addestrato in {execution_time:.4f} secondi.")

    return model, train_losses, val_losses, execution_time


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


def final_training(X, Y, kinematics='inverse'):
    # Addestra il modello finale con i migliori iperparametri
    best_params = load_best_param('search/random_search_results_' + kinematics + '.json')
    final_model, train_losses, val_losses, execution_time = train_final_model(X, Y, best_params, 3, 3,
                                                                              200)

    # Salva il modello e i risultati
    torch.save(final_model.state_dict(), 'model/mlp/final_model_' + kinematics + '.pth')
    np.savez('model/metrics/final_mlp-' + kinematics + '_training_metrics.npz', train_losses=train_losses,
             val_losses=val_losses, execution_time=execution_time)

    plot_metric(train_losses, val_losses, 'Loss', 'model/mlp/plot/final_training_' + kinematics + '_loss.png')


def inference(X, kinematics):
    # Carica il modello salvato
    best_params = load_best_param('search/random_search_results_' + kinematics + '.json')
    model = KinematicsModel(input_size=3, hidden_sizes=best_params['hidden_sizes'], output_size=3,
                            dropout_rate=best_params['dropout_rate'])

    model.load_state_dict(torch.load('model/mlp/final_model_' + kinematics + '.pth'))
    model.eval()  # Imposta il modello in modalità di valutazione
    #
    # test del modello con esempi generati randomicamente
    num_test_samples = 1
    X_test, y_true = generate_sample_data(num_test_samples)
    # X_test_tensor = torch.FloatTensor(X_test)
    #
    actuator = np.array([
        [.237, .312, .13],
        #     # [.2, .18, .2],
        #     # [.2, .2, .18]
    ])
    poses = np.array([[-0.006, 0.006, 0.017]])
    # -0.002 -0.02 -0.036
    X_test_tensor = torch.FloatTensor(X)
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy()

    return y_pred


def continual_learning(X, Y, kinematics, save_new_model=True, model=None):
    # Carica il modello salvato
    best_params = load_best_param('search/random_search_results_' + kinematics + '.json')
    if model is None:
        model = KinematicsModel(input_size=3, hidden_sizes=best_params['hidden_sizes'], output_size=3,
                                dropout_rate=best_params['dropout_rate'])
        model.load_state_dict(torch.load('model/mlp/final_model_' + kinematics + '.pth'))

    final_model, train_losses, val_losses, execution_time = train_final_model(X, Y, best_params, 3, 3,
                                                                              200, model)
    if save_new_model:
        torch.save(final_model.state_dict(), 'model/mlp/final_model_' + kinematics + '.pth')

    return train_losses, val_losses, final_model


def main():
    kinematics = 'inverse'

    ###### TRAINING MODEL ####################
    # # Usa la ricerca casuale per trovare i migliori iperparametri
    # X, Y = read_data('datasets/dataset.txt', kinematics)
    # # random_search(X, Y, 3, 3, 200, 20, kinematics)
    # final_training(X, Y, kinematics)

    ####### CONTINUAL LEARNING ################
    X, Y = read_data("datasets/workspaces/quadrante_I.csv", kinematics)
    train_losses, val_losses, model = continual_learning(X, Y, kinematics, save_new_model=True)
    np.savez('continual_learning/metrics/final_mlp-' + kinematics + '_continual_learning_quadI_metrics.npz',
             train_losses=train_losses, val_losses=val_losses)
    plot_metric(train_losses, val_losses, 'Continual Learning Loss',
                'continual_learning/mlp_' + kinematics + '_quadrante_I.png')

    X, Y = read_data("datasets/workspaces/quadrante_II.csv", kinematics)
    train_losses, val_losses, model = continual_learning(X, Y, kinematics, save_new_model=True, model=model)
    np.savez('continual_learning/metrics/final_mlp-' + kinematics + '_continual_learning_quadII_metrics.npz',
             train_losses=train_losses, val_losses=val_losses)
    plot_metric(train_losses, val_losses, 'Continual Learning Loss',
                'continual_learning/mlp_' + kinematics + '_quadrante_II.png')

    #### INFERENCE ########################
    # expected = [0.1, 0.1, 0.1]
    # prediction = inference(np.array([[0.0, 0.0, 0.1]]), kinematics)
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
