import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


def read_data(filename, kinematics='inverse'):
    X = []  # input
    Y = []  # output/label

    with open(filename, 'r') as f:
        # Salta la riga di intestazione
        next(f)
        for line in f:
            parts = line.strip().split()

            # poses () è composto dai primi 3 valori per la base e i successivi per la punta
            base_poses = list(map(float, parts[3:6]))
            peak_poses = list(map(float, parts[9:12]))
            act = list(map(float, parts[:3]))

            if kinematics == 'direct':
                X.append(act)
                # Y.append(base_poses + peak_poses)
                Y.append(peak_poses)
            else:
                # inverse kinematics
                # X.append(base_poses + peak_poses)
                X.append(peak_poses)
                Y.append(act)
    return np.array(X), np.array(Y)


def prepare_data(X, Y, batch_size=256, shuffle=False):
    # Converti in tensori PyTorch
    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(Y)

    # Crea dataset e dataloader
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def generate_sample_data(num_samples):
    # poses
    x = np.random.uniform(0, 1, num_samples)
    y = np.random.uniform(0, 1, num_samples)
    z = np.random.uniform(0, 1, num_samples)

    # actuator
    xa = np.random.uniform(0, 1, num_samples)
    ya = np.random.uniform(0, 1, num_samples)
    za = np.random.uniform(0, 1, num_samples)

    return np.column_stack((x, y, z)), np.column_stack((xa, ya, za))


def plot_metric(train_metric, val_metric, metric_name, save_path=None, yscale='linear'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_metric, label=f'Training {metric_name}')
    plt.plot(val_metric, label=f'Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.yscale(yscale)
    plt.legend()
    plt.title(f'Training and Validation {metric_name}')
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
        print(f"Grafico {metric_name} salvato in {save_path}")
    else:
        plt.show()


def split_cartesian_workspace(dataset, num_quadranti=2):
    # Carica il file txt nel DataFrame
    # Utilizza '\s+' come separatore per gestire uno o più spazi consecutivi
    df = pd.read_csv(dataset, sep='\s+')

    # Verifica il numero di colonne nel DataFrame
    numero_colonne = df.shape[1]
    print(f'Il file ha {numero_colonne} colonne.')

    nomi_colonne = [
        'actuation1', 'actuation2', 'actuation3',
        'pose1', 'pose2', 'pose3', 'pose4', 'pose5', 'pose6',
        'x', 'y', 'z',  # Le colonne 10, 11, 12
        'config1', 'config2', 'config3',
        'additional1', 'additional2', 'additional3'
    ]

    # Assicurati che il numero di nomi corrisponda al numero di colonne
    if len(nomi_colonne) != numero_colonne:
        print("Attenzione: Il numero di nomi di colonne non corrisponde al numero di colonne nel file.")
        # Se necessario, aggiusta la lista nomi_colonne qui

    df.columns = nomi_colonne

    if num_quadranti == 2:
        condizioni = [
            (df['x'] > 0),  # Quadrante I-IV
            (df['x'] < 0)  # Quadrante II-III
        ]
        valori_quadranti = [1, 2]
    else:
        condizioni = [
            (df['x'] > 0) & (df['y'] > 0),  # Quadrante I
            (df['x'] < 0) & (df['y'] > 0),  # Quadrante II
            (df['x'] < 0) & (df['y'] < 0),  # Quadrante III
            (df['x'] > 0) & (df['y'] < 0)  # Quadrante IV
        ]
        valori_quadranti = [1, 2, 3, 4]

    # Valori da assegnare per ciascun quadrante

    # Creazione di una nuova colonna 'quadrante' nel DataFrame
    df['quadrante'] = np.select(condizioni, valori_quadranti, default=0)  # default=0 per i punti sugli assi

    # Ora puoi filtrare il DataFrame per ciascun quadrante
    quadrante_I = df[df['quadrante'] == 1]
    quadrante_II = df[df['quadrante'] == 2]
    if num_quadranti == 4:
        quadrante_III = df[df['quadrante'] == 3]
        quadrante_IV = df[df['quadrante'] == 4]

    # Se desideri, puoi esportare i dati di ciascun quadrante
    quadrante_I.to_csv('datasets/workspaces/quadrante_I.csv', sep=' ', index=False)
    quadrante_II.to_csv('datasets/workspaces/quadrante_II.csv', sep=' ', index=False)
    if num_quadranti == 4:
        quadrante_III.to_csv('datasets/workspaces/quadrante_III.csv', sep=' ', index=False)
        quadrante_IV.to_csv('datasets/workspaces/quadrante_IV.csv', sep=' ', index=False)


def compare_model_metrics(file_pattern):
    npz_files = glob.glob(file_pattern)

    metrics = {}
    for file in npz_files:
        # Carica i dati dal file .npz
        data = np.load(file)

        # Estrai il nome del modello dal nome del file
        model_name = os.path.basename(file).split('_')[
            1]  # Questo prende la parte dopo 'final_' e prima del prossimo '_'

        # Memorizza le metriche nel dizionario
        metrics[model_name] = {
            'train_losses': data['train_losses'],
            'val_losses': data['val_losses'],
            'execution_time': data['execution_time']
        }

    plt.figure(figsize=(12, 8))

    for model_name, metric in metrics.items():
        # train_losses = metric['train_losses']
        val_losses = metric['val_losses']

        # Crea un array di epoche in base alla lunghezza delle perdite
        epochs = range(1, len(val_losses) + 1)

        # Plotta le perdite di training e validazione
        # plt.plot(epochs, train_losses, label=f'Train Loss - {model_name}', linestyle='--')
        plt.plot(epochs, val_losses, label=f'{model_name}')

    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.title('Validation Loss per Diversi Modelli')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('model/metrics/plot/model_loss_comparison.png')

    # execution time
    model_names = []
    execution_times = []

    for model_name, metric in metrics.items():
        model_names.append(model_name)
        execution_times.append(metric['execution_time'])

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, execution_times, color='skyblue')
    plt.xlabel('Modelli')
    plt.ylabel('Tempo di Esecuzione (secondi)')
    plt.title('Confronto del Tempo di Esecuzione tra Modelli')
    plt.grid(axis='y')
    plt.show()
    plt.savefig('model/metrics/plot/model_time_comparison.png')


def compare_continual_learning_metrics(file_pattern):
    npz_files = glob.glob(file_pattern)
    other_file = glob.glob('continual_learning/metrics/final_*_training_metrics.npz')
    full_list = npz_files + other_file
    metrics = {}
    for file in full_list:
        # Carica i dati dal file .npz
        data = np.load(file)

        # Estrai il nome del modello dal nome del file
        model_name = os.path.basename(file).split('_')[
            1]  # Questo prende la parte dopo 'final_' e prima del prossimo '_'

        # Memorizza le metriche nel dizionario
        metrics[model_name] = {
            'train_losses': data['train_losses'],
            'val_losses': data['val_losses']
        }

    plt.figure(figsize=(12, 8))

    for model_name, metric in metrics.items():
        # train_losses = metric['train_losses']
        val_losses = metric['val_losses']

        # Crea un array di epoche in base alla lunghezza delle perdite
        epochs = range(1, len(val_losses) + 1)

        # Plotta le perdite di training e validazione
        # plt.plot(epochs, train_losses, label=f'Train Loss - {model_name}', linestyle='--')
        plt.plot(epochs, val_losses, label=f'{model_name}')

    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.title('Validation Loss Continual Learning per Diversi Modelli')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('continual_learning/plot/model_continual_learning_loss_comparison.png')
