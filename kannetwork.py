import json

import numpy as np
import torch
import time

from kan import KAN, create_dataset_from_data
from matplotlib import pyplot as plt

from simulator import plotSimulation
from utils import read_data, plot_metric, split_cartesian_workspace
from itertools import product


def grid_search_kan(X, Y, test_size=0.2):
    # Definizione della griglia di parametri estesa
    param_grid = {
        'hidden_layers': [1, 2],
        'neurons': [7, 13, 19],
        'grid': [5, 7, 10, 15],
        'k': [2, 3, 4],
        'optimizer': ['Adam', 'LBFGS'],
        'learning_rate': [0.001, 0.01, 0.1],
        'lambda': [0.001, 0.01, 0.0],
        'steps': [200]
    }

    # Dividi i dati in training e test set
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    X_train, X_test = X[indices[:-n_test]], X[indices[-n_test:]]
    Y_train, Y_test = Y[indices[:-n_test]], Y[indices[-n_test:]]

    # Converti in tensori PyTorch
    X_train, Y_train = torch.FloatTensor(X_train), torch.FloatTensor(Y_train)
    X_test, Y_test = torch.FloatTensor(X_test), torch.FloatTensor(Y_test)

    all_configs = []

    # Genera tutte le possibili combinazioni di parametri
    param_combinations = list(product(*param_grid.values()))
    start_time = time.perf_counter()
    for i, params in enumerate(param_combinations):
        # Estrai i parametri
        hidden_layers, neurons, grid, k, optimizer, learning_rate, lambda_reg, steps = params
        # Costruisci la configurazione width
        width = [X.shape[1]] + [neurons] * hidden_layers + [Y.shape[1]]

        # Crea il modello
        model = KAN(width=width, grid=grid, k=k)

        # Prepara il dataset
        dataset = create_dataset_from_data(X_train, Y_train, train_ratio=0.8, device='cpu')

        # Addestra il modello
        results = model.fit(dataset, opt=optimizer, lr=learning_rate, steps=steps, lamb=lambda_reg)

        mse_kan = np.mean(results['test_loss'])
        print(f"Combination {i + 1}/{len(param_combinations)}: MSE = {mse_kan}")
        config = {
            'hidden_layers': hidden_layers,
            'neurons': neurons,
            'grid': grid,
            'k': k,
            'optimizer': optimizer,
            'learning_rate': learning_rate,
            'lambda': lambda_reg,
            'steps': steps,
            'mse': float(np.mean(results['test_loss']))
        }
        all_configs.append(config)

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Grid Search eseguita in {execution_time:.4f} secondi.")
    # Ordina le configurazioni per MSE crescente
    all_configs.sort(key=lambda x: x['mse'])

    # Salva tutte le configurazioni in un file JSON
    with open('grid_search_kan_direct.json', 'w') as f:
        json.dump(all_configs, f, indent=2)


def train_final_model(X, Y, config_file, kinematics='inverse'):
    with open(config_file, 'r') as f:
        all_configs = json.load(f)

    config = all_configs[0]  # La prima configurazione è la migliore

    print("Training final model with best configuration:")
    print(config)

    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(Y)

    dataset = create_dataset_from_data(X_tensor, Y_tensor, train_ratio=0.8, device='cpu')
    #  n,2n+1
    width = [X.shape[1]] + [config['neurons']] + [Y.shape[1]]
    # grid = [7, 10, 15]

    start_time = time.perf_counter()

    model = KAN(width=width, grid=config['grid'], k=config['k'], seed=config['seed'], save_act=True)
    history = model.fit(dataset, opt=config['optimizer'], lr=config['learning_rate'], steps=config['steps'],
                        lamb=config['lambda'])

    plot_metric(history['train_loss'], history['test_loss'], 'Loss',
                'model/kan/plot/1training_kan_' + kinematics + '_loss.png')

    # for value in grid:
    #     model = model.refine(value)
    #     history = model.fit(dataset, opt='Adam', lr=0.001, steps=200, lamb=0.0)
    #     # Plotta e salva i risultati
    #     plot_metric(history['train_loss'], history['test_loss'], 'Loss',
    #                 'model/kan/plot/training_kan_' + kinematics + '_grid_' + str(value) + '_loss.png')

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Il modello è stato addestrato in {execution_time:.4f} secondi.")
    np.savez('model/metrics/1final_kan-' + kinematics + '_training_metrics.npz', train_losses=history['train_loss'],
             val_losses=history['test_loss'], execution_time=execution_time)

    # plot prepruning
    plt.figure(figsize=(15, 20))
    model.plot(beta=10)  # beta controlla la risoluzione del grafo
    plt.savefig('model/kan/plot/1network_architecture_prepruning_' + kinematics + '.png')
    plt.close()

    model = model.prune()
    # plot post pruning
    plt.figure(figsize=(15, 20))
    model.plot(beta=10)  # beta controlla la risoluzione del grafo
    plt.savefig('model/kan/plot/1network_architecture_postpruning_' + kinematics + '.png')
    plt.close()

    # Salva il modello finale
    model.saveckpt(path='model/kan/1final_model_' + kinematics)
    print("Final model saved")
    return history


def load_best_param(filename):
    with open(filename, 'r') as f:
        results = json.load(f)

    if results:
        best_trial = results[0]  # Il primo elemento è il migliore (loss più bassa)
        best_params = {
            "hidden_layers": best_trial['hidden_layers'],
            "neurons": best_trial['neurons'],
            "grid": best_trial['grid'],
            "k": best_trial['k'],
            "optimizer": best_trial['optimizer'],
            "learning_rate": best_trial['learning_rate'],
            "lambda": best_trial['lambda'],
            "seed": best_trial['seed'],
            "steps": best_trial['steps']
        }

        return best_params
    else:
        print(f"Il file {filename} è vuoto.")
        return None


def inference(X, output_size, kinematics):
    # Carica il modello salvato
    best_params = load_best_param('search/grid_search_kan_' + kinematics + '.json')
    model = KAN(width=[X.shape[1], best_params['neurons'], output_size], grid=best_params['grid'], k=best_params['k'],
                seed=best_params['seed'])
    # carico l'ultimo modello addestrato
    model = model.loadckpt('model/kan/final_model_' + kinematics)
    #
    # Imposta il modello in modalità valutazione
    model.eval()
    actuator = np.array([
        [.237, .312, .13],
        #     # [.2, .18, .2],
        #     # [.2, .2, .18]
    ])
    poses = np.array([[0.0, 0.0, 0.1]])
    X_test_tensor = torch.FloatTensor(X)
    # Esegui la predizione
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy()
    return y_pred


def continual_learning(model, config, X, Y, kinematics, save_new_model=False):
    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(Y)

    dataset = create_dataset_from_data(X_tensor, Y_tensor, train_ratio=0.8, device='cpu')

    history = model.fit(dataset, opt=config['optimizer'], lr=config['learning_rate'], steps=config['steps'],
                        lamb=config['lambda'])

    if save_new_model:
        model.saveckpt(path='model/kan/final_model_' + kinematics)

    return history, model


def main():
    kinematics = 'inverse'

    ####### MODEL TRAINING ########################
    X, Y = read_data("datasets/new_dataset.txt", kinematics)
    # # grid_search_kan(X, Y)
    train_final_model(X, Y, 'search/grid_search_kan_' + kinematics + '.json', kinematics)

    ##### CONTINUAL LEARNING ############
    # X, Y = read_data("datasets/workspaces/quadrante_I.csv", kinematics)
    # # Carica il modello salvato
    # config = load_best_param('search/grid_search_kan_' + kinematics + '.json')
    # model = KAN(width=[3, config['neurons'], 3], grid=config['grid'], k=config['k'],
    #             seed=config['seed'])
    # # carico l'ultimo modello addestrato
    # model = model.loadckpt('model/kan/final_model_' + kinematics)
    #
    # results, model = continual_learning(model, config, X, Y, kinematics, save_new_model=True)
    # plot_metric(results['train_loss'], results['test_loss'], 'Continual Learning Loss',
    #             'continual_learning/kan_' + kinematics + '_quadrante_I.png')
    # np.savez('model/metrics/final_kan-' + kinematics + '_continual_learning_quadI_metrics.npz',
    #          train_losses=results['train_loss'],
    #          val_losses=results['test_loss'])
    #
    # X, Y = read_data("datasets/workspaces/quadrante_II.csv", kinematics)
    # results,_ = continual_learning(model, config, X, Y, kinematics, save_new_model=True)
    # plot_metric(results['train_loss'], results['test_loss'], 'Continual Learning Loss',
    #             'continual_learning/kan_' + kinematics + '_quadrante_II.png')
    # np.savez('continual_learning/metrics/final_kan-' + kinematics + '_continual_learning_quadII_metrics.npz',
    #          train_losses=results['train_loss'],
    #          val_losses=results['test_loss'])

    ###### INFERENCE ##########################
    # expected = [.1, .1, .1]
    # prediction = inference(np.array([[0, 0, 0.1]]), 3, kinematics)
    # print("Test Results:")
    # if kinematics == 'inverse':
    #     print(f"True actuation: {expected}")
    #     print(f"Predicted actuation: {prediction}")
    #     plotSimulation(prediction)
    # else:
    #     # print(f"Input (x, y, z):, {poses} ")
    #     print(f"Ture poses: {expected}")
    #     print(f"Predicted poses: {prediction}")


if __name__ == "__main__":
    main()
