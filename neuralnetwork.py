import json
import random
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


#### A PARTIRE DA UNA CONFIGURAZIONE, PREDIRE L'ATTUAZIONE
# Definizione del modello
class KinematicsModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super(KinematicsModel, self).__init__()
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

    for trial in range(num_trials):
        print('Numero trial:', trial)
        # Genera iperparametri casuali
        hidden_sizes = [random.choice([64, 128, 256, 512]) for _ in range(random.randint(2, 5))]
        learning_rate = random.choice([0.1, 0.01, 0.001, 0.0001])
        batch_size = random.choice([32, 64, 128, 256])
        dropout_rate = random.uniform(0.1, 0.5)
        weight_decay = random.choice([0, 0.0001, 0.00001, 0.000001])
        optimization = random.choice(['Adam', 'SGD'])
        print('Generati parametri: hidden size: ', hidden_sizes, '- lr: ', learning_rate, '- batch_size:', batch_size,
              '- dropout: ', dropout_rate, '- decay: ', weight_decay, '- optimizer: ', optimization)

        # Crea e addestra il modello
        model = KinematicsModel(input_size, hidden_sizes, output_size, dropout_rate)
        if optimization == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
            val_r2 = r2_score(Y_val, val_outputs.numpy())

        print(f"Trial {trial + 1}/{num_trials}: Val Loss: {val_loss:.4f}, R² Score: {val_r2:.4f}")

        best_params = {
            'hidden_sizes': hidden_sizes,
            'optimization': optimization,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'dropout_rate': dropout_rate,
            'val_loss': val_loss.item(),
            'r2_score': val_r2
        }
        all_results.append(best_params)

    # Ordina i risultati per validation loss
    sorted_results = sorted(all_results, key=itemgetter('val_loss'))

    # Salva i risultati ordinati in un file JSON
    with open('search/random_search_results_' + kinematics + '.json', 'w') as f:
        json.dump(sorted_results, f, indent=2)


def train_final_model(X, Y, best_params, input_size, output_size, num_epochs):
    model = KinematicsModel(input_size, best_params['hidden_sizes'], output_size, best_params['dropout_rate'])
    if best_params['optimization'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'],
                               weight_decay=best_params['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=best_params['learning_rate'],
                              weight_decay=best_params['weight_decay'])
    criterion = nn.MSELoss()

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train)),
                              batch_size=best_params['batch_size'], shuffle=True)

    train_losses = []
    val_losses = []
    train_r2_scores = []
    val_r2_scores = []

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

        epoch_train_outputs = torch.cat(epoch_train_outputs)
        epoch_train_targets = torch.cat(epoch_train_targets)
        train_r2 = r2_score(epoch_train_targets.numpy(), epoch_train_outputs.numpy())
        train_r2_scores.append(train_r2)

        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(X_val))
            val_loss = criterion(val_outputs, torch.FloatTensor(Y_val)).item()
            val_r2 = r2_score(Y_val, val_outputs.numpy())

        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        # train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_r2_scores.append(val_r2)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Train R² Score: {train_r2:.4f}, Val R² Score: {val_r2:.4f}')

    return model, train_losses, val_losses, train_r2_scores, val_r2_scores


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
    final_model, train_losses, val_losses, train_r2_scores, val_r2_scores = train_final_model(X, Y, best_params, 3, 3,
                                                                                              200)

    # Salva il modello e i risultati
    torch.save(final_model.state_dict(), 'model/mlp/final_model_' + kinematics + '.pth')
    np.savez('model/mlp/final_' + kinematics + '_training_metrics.npz', train_losses=train_losses,
             val_losses=val_losses,
             val_r2_scores=val_r2_scores)

    plot_metric(train_losses, val_losses, 'Loss', 'model/mlp/final_training_' + kinematics + '_loss.png')
    plot_metric(train_r2_scores, val_r2_scores, 'R² Score', 'model/mlp/final_training_' + kinematics + '_r2.png')


def main():
    # inserire un modo per chiede se addestrare un modello per cinematica diretta o indiretta per cambiare gli input

    # Usa la ricerca casuale per trovare i migliori iperparametri
    # X, Y = read_data('dataset.txt', 'inverse')
    # random_search(X, Y, 3, 3, 200, 20, 'inverse')

    # final_training(X, Y, 'inverse')

    # Carica il modello salvato
    model = KinematicsModel()
    model.load_state_dict(torch.load('model/mlp/nn_model.pth'))
    model.eval()  # Imposta il modello in modalità di valutazione
    #
    # test del modello con esempi generati randomicamente
    num_test_samples = 1
    X_test, y_true = generate_sample_data(num_test_samples)
    X_test_tensor = torch.FloatTensor(X_test)
    #
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy()

    print("Test Results:")
    for i in range(num_test_samples):
        print(f"Input (x, y, z): {X_test[i]}")
        print(f"Predicted actuation: {y_pred[i]}")
        print(f"True actuation: {y_true[i]}")
        print()

    plotSimulation(y_pred)

    # Aggancio con il simulatore


# Test del modello
if __name__ == "__main__":
    main()
