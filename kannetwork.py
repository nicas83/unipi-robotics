from torch.utils.data import TensorDataset, DataLoader

from pykan.kan import *


def generate_sample_data(num_samples):
    # configuration
    x = np.random.uniform(0, 1, num_samples)
    y = np.random.uniform(0, 1, num_samples)
    z = np.random.uniform(0, 1, num_samples)

    # actuator
    xa = np.random.uniform(0, 1, num_samples)
    ya = np.random.uniform(0, 1, num_samples)
    za = np.random.uniform(0, 1, num_samples)

    return np.column_stack((x, y, z)), np.column_stack((xa, ya, za))


def read_data(filename):
    X = []  # configuration
    Y = []  # act

    with open(filename, 'r') as f:
        # Salta la riga di intestazione
        next(f)
        for line in f:
            parts = line.strip().split()

            # configuration (input) è composto dai primi 3 valori
            configuration = list(map(float, parts[:3]))
            X.append(configuration)
            # act (output) è composto dagli ultimi valori
            act = list(map(float, parts[3:6]))
            Y.append(act)

    return torch.FloatTensor(X), torch.FloatTensor(Y)


def prepare_data_for_pytorch(X, Y, batch_size=32):
    # Converti in tensori PyTorch
    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(Y)

    # Crea dataset e dataloader
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def train_model():
    # width rappresenta la struttura del modello: primo parametri dimension input (x,y in questo caso), secondo il
    # numero di neuroni, il terzo dimension dell'output, k è il grado della funzione spline (3=cubica), capire cosa
    # è il grid
    X, Y = read_data("dataset.txt")

    # capire come fare lo shuffle dei dati
    dataset = create_dataset_from_data(X, Y, train_ratio=0.8, device='cpu')

    model = KAN(width=[3, 5, 3], grid=5, k=3, seed=42)

    # model(dataset['train_input'])
    # model.plot(beta=100)

    # Training loop: steps corrisponde al numero di epoche del fit
    model.fit(dataset, opt="Adam", steps=100, lamb=0.01, batch=5)


def main():
    #train_model()

    # valutare la parametrizzazione dell'inizializzazione del modello
    model = KAN(width=[3, 5, 3], grid=5, k=3)
    # carico l'ultimo modello addestrato
    model.loadckpt('model/0.1')

    # Imposta il modello in modalità valutazione
    model.eval()
    # test del modello con esempi generati randomicamente
    num_test_samples = 1
    X_test, y_true = generate_sample_data(num_test_samples)
    X_test_tensor = torch.FloatTensor(X_test)
    # Esegui la predizione
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy()

    print("Test Results:")
    for i in range(num_test_samples):
        print(f"Input (x, y, z): {X_test[i]}")
        print(f"Predicted actuation: {y_pred[i]}")
        print(f"True actuation: {y_true[i]}")
    #     print()


if __name__ == "__main__":
    main()
