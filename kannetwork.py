import json

from pykan.kan import *
from utils import read_data, plot_metric
from itertools import product


def grid_search_kan(X, Y, param_grid, test_size=0.2):
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

    for i, params in enumerate(param_combinations):
        # Estrai i parametri
        hidden_layers, neurons, grid, k, optimizer, learning_rate, lambda_reg, steps, batch = params
        # Costruisci la configurazione width
        width = [X.shape[1]] + [neurons] * hidden_layers + [Y.shape[1]]

        # Crea il modello
        model = KAN(width=width, grid=grid, k=k)

        # Prepara il dataset
        dataset = create_dataset_from_data(X_train, Y_train, train_ratio=0.8, device='cpu')

        # Addestra il modello
        results = model.fit(dataset, opt=optimizer, lr=learning_rate, steps=steps, lamb=lambda_reg, batch=batch)

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
            'batch': batch,
            'mse': float(np.mean(results['test_loss']))
        }
        all_configs.append(config)
    # Ordina le configurazioni per MSE crescente
    all_configs.sort(key=lambda x: x['mse'])

    # Salva tutte le configurazioni in un file JSON
    with open('grid_search_kan_direct.json', 'w') as f:
        json.dump(all_configs, f, indent=2)


def train_final_model(X, Y, config):
    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(Y)

    dataset = create_dataset_from_data(X_tensor, Y_tensor, train_ratio=0.8, device='cpu')
    #  n,2n+1
    width = [X.shape[1]] + [X.shape[1] * 2 + 1] * config['hidden_layers'] + [Y.shape[1]]
    width = [3, 1, 1, 3]
    grid = [7, 10]
    # model = KAN(width=width, grid=config['grid'], k=config['k'], seed=1)
    # history = model.fit(dataset, opt=config['optimizer'], lr=config['learning_rate'],
    #                     steps=config['steps'], lamb=config['lambda'], lamb_entropy=5, batch=config['batch'])

    model = KAN(width=[3, 7, 3], grid=5, k=3, seed=42)
    history = model.fit(dataset, opt='LBFGS', lr=0.001, steps=200, lamb=0.01, lamb_entropy=10)

    # for value in grid:
    #     model = model.refine(value)
    #     history = model.fit(dataset, opt=config['optimizer'], lr=config['learning_rate'],
    #                         steps=config['steps'], lamb=config['lambda'], lamb_entropy=5, batch=config['batch'])
    #
    # model = model.prune()

    # Salva il modello finale
    model.saveckpt(path='model/kan/final_model_kan_direct.pth')
    # model.plot(folder='model/kan', beta=100)
    print("Final model saved to 'final_model.pth'")
    return history


def execute_final_training(X, Y, config_file='grid_search_kan_direct.json'):
    with open(config_file, 'r') as f:
        all_configs = json.load(f)

    best_config = all_configs[0]  # La prima configurazione è la migliore

    print("Training final model with best configuration:")
    print(best_config)

    # Addestra il modello finale
    history = train_final_model(X, Y, best_config)

    # Plotta e salva i risultati
    plot_metric(history['train_loss'], history['test_loss'], 'Loss', 'model/kan/training_kan_direct_results.png')

    print("Final model trained. Results plotted and saved to 'training_results.png'")


def main():
    # Definizione della griglia di parametri estesa
    param_grid = {
        'hidden_layers': [1, 2, 3],
        'neurons': [5, 10, 15],
        'grid': [3, 5, 7],
        'k': [2, 3, 4],
        'optimizer': ['Adam', 'LBFGS'],
        'learning_rate': [0.001, 0.01, 0.1],
        'lambda': [0.001, 0.01, 0.1],
        'steps': [200],
        'batch': [32, 64, 128, 256]
    }

    X, Y = read_data("dataset.txt", 'direct')
    # X, Y = read_data("dataset.txt", 'inverse')
    # grid_search_kan(X, Y, param_grid)
    execute_final_training(X, Y, 'grid_search_kan_direct.json')

    # train_model()
    # valutare la parametrizzazione dell'inizializzazione del modello
    # model = KAN(width=[3, 5, 3], grid=5, k=3)
    # # carico l'ultimo modello addestrato
    # model.loadckpt('model/0.0')
    #
    # # Imposta il modello in modalità valutazione
    # model.eval()
    # # test del modello con esempi generati randomicamente
    # num_test_samples = 1
    # # X_test, y_true = generate_sample_data(num_test_samples)
    # act = np.array([
    #     [-0.002, -0.02, -0.036],
    #     #     # [0.237, 0.312, 0.13],
    #     #     # [.2, .2, .18]
    # ])
    # X_test_tensor = torch.FloatTensor(act)
    # # Esegui la predizione
    # with torch.no_grad():
    #     y_pred = model(X_test_tensor).numpy()
    #
    # print("Test Results:")
    # for i in range(num_test_samples):
    #     print(f"Input (x, y, z): {act[i]}")
    #     print(f"Predicted actuation: {y_pred[i]}")
    #     # print(f"True actuation: {y_true[i]}")
    # #     print()


if __name__ == "__main__":
    main()

# Best configuration found:
# {'hidden_layers': 1, 'neurons': 10, 'grid': 3, 'k': 4, 'optimizer': 'Adam', 'learning_rate': 0.01, 'lambda': 0.001,
# 'steps': 200, 'batch': 64, 'mse': 0.003728143870830536}
