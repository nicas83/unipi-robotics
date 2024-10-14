import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


#### A PARTIRE DA UNA CONFIGURAZIONE, PREDIRE L'ATTUAZIONE
# Definizione del modello
class InverseKinematicsModel(nn.Module):
    def __init__(self):
        super(InverseKinematicsModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.network(x)


# Generazione di dati di esempio
def generate_sample_data(num_samples):

    # configuration
    x = np.random.uniform(0, 1, num_samples)
    y = np.random.uniform(0, 1, num_samples)
    z = np.random.uniform(0, 1, num_samples)

    #actuator
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

    return np.array(X), np.array(Y)


def prepare_data(X, Y, batch_size=32):
    # Converti in tensori PyTorch
    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(Y)

    # Crea dataset e dataloader
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def train():
    X, Y = read_data('dataset.txt')

    print(f"Dimensioni di X (configuration): {X.shape}")
    print(f"Dimensioni di Y (act): {Y.shape}")

    # Prepara i dati per PyTorch
    dataloader = prepare_data(X, Y)

    # Inizializzazione del modello, della funzione di loss e dell'ottimizzatore
    model = InverseKinematicsModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Addestramento del modello
    num_epochs = 1000
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Salva il modello
    torch.save(model.state_dict(), 'model/mlp/nn_model.pth')


# Funzione per testare il modello
def main():
    train()

    # Carica il modello salvato
    model = InverseKinematicsModel()
    model.load_state_dict(torch.load('model/mlp/nn_model.pth'))
    model.eval()  # Imposta il modello in modalità di valutazione

    # test del modello con esempi generati randomicamente
    num_test_samples = 1
    X_test, y_true = generate_sample_data(num_test_samples)
    X_test_tensor = torch.FloatTensor(X_test)

    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy()

    print("Test Results:")
    for i in range(num_test_samples):
        print(f"Input (x, y, z): {X_test[i]}")
        print(f"Predicted actuation: {y_pred[i]}")
        print(f"True actuation: {y_true[i]}")
        print()


# Test del modello
if __name__ == "__main__":
    main()
