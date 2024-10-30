import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from model.mlp.classes.KinematicsModel import KinematicsModel
from neuralnetwork import load_best_param
from utils.utils import read_data, plot_training_loss, plot_validation_loss


def train_model(model, train_loader, criterion, optimizer, epochs, device):
    model.train()
    losses = []
    val_losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        model.eval()
        with torch.no_grad():
            val_outputs = model(batch_X)
            val_loss = criterion(val_outputs, batch_y).item()

        val_losses.append(val_loss)
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:}')

    return losses, val_losses


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    val_losses = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            val_losses.append(loss.item())

    avg_loss = total_loss / len(test_loader)

    return avg_loss, val_losses, predictions, actuals


def main():
    ####### CONTINUAL LEARNING ################
    kinematics = 'inverse'
    device = 'cpu'
    # Carica il modello salvato
    best_params = load_best_param('search/random_search_results_' + kinematics + '.json')
    epochs = 200

    model = KinematicsModel(input_size=3, hidden_sizes=best_params['hidden_sizes'], output_size=3,
                            dropout_rate=best_params['dropout_rate'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    X, Y = read_data("datasets/workspaces/quadrante_I.csv", kinematics)
    X_train_I, X_val_I, Y_train_I, Y_val_I = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_loaderI = DataLoader(TensorDataset(torch.FloatTensor(X_train_I), torch.FloatTensor(Y_train_I)),
                               batch_size=best_params['batch_size'], shuffle=True)
    test_loaderI = DataLoader(TensorDataset(torch.FloatTensor(X_val_I), torch.FloatTensor(Y_val_I)),
                              batch_size=best_params['batch_size'], shuffle=True)

    X, Y = read_data("datasets/workspaces/quadrante_II.csv", kinematics)
    X_train_II, X_val_II, Y_train_II, Y_val_II = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_loaderII = DataLoader(TensorDataset(torch.FloatTensor(X_train_II), torch.FloatTensor(Y_train_II)),
                                batch_size=best_params['batch_size'],
                                shuffle=True)
    test_loaderII = DataLoader(TensorDataset(torch.FloatTensor(X_val_II), torch.FloatTensor(Y_val_II)),
                               batch_size=best_params['batch_size'], shuffle=True)

    print("Training sul primo dataset...")
    losses1, val_losses1 = train_model(model, train_loaderI, criterion, optimizer, epochs, device)
    avg_loss1, evaluate_losses1, pred1, act1 = evaluate_model(model, test_loaderI, criterion, device)

    # Salva il modello dopo il primo training
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    # }, 'model_phase1.pth')

    # Fase 2: Continual learning sul secondo dataset
    print("\nContinual learning sul secondo dataset...")
    losses2, val_losses2 = train_model(model, train_loaderII, criterion, optimizer, epochs, device)

    # Valutazione finale sul secondo dataset
    print("\nValutazione finale sul secondo dataset...")
    avg_loss2, evaluate_losses2, pred2, act2 = evaluate_model(model, test_loaderII, criterion, device)

    # Valutazione finale sul primo dataset
    print("\nValutazione finale sul primo dataset...")
    avg_loss3, evaluate_losses3, pred3, act3 = evaluate_model(model, test_loaderI, criterion, device)

    # Visualizza l'andamento del training
    plot_training_loss(losses1, val_losses1, losses2, val_losses2,'mlp')
    plot_validation_loss(evaluate_losses1, evaluate_losses2, evaluate_losses3,'mlp')


if __name__ == "__main__":
    main()

    # np.savez('continual_learning/metrics/final_mlp-' + kinematics + '_continual_learning_quadI_metrics.npz',
    #          train_losses=train_losses_I, val_losses=val_losses_I)
