import numpy as np
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
                #Y.append(base_poses + peak_poses)
                Y.append(peak_poses)
            else:
                # inverse kinematics
                # X.append(base_poses + peak_poses)
                X.append(peak_poses)
                Y.append(act)
    print(X[0], "\n")
    print(Y[0], "\n")
    return np.array(X), np.array(Y)


def prepare_data(X, Y, batch_size=256):
    # Converti in tensori PyTorch
    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(Y)

    # Crea dataset e dataloader
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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


def plot_metric(train_metric, val_metric, metric_name, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_metric, label=f'Training {metric_name}')
    plt.plot(val_metric, label=f'Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f'Training and Validation {metric_name}')
    plt.savefig(save_path)
    plt.close()
    print(f"Grafico {metric_name} salvato in {save_path}")
