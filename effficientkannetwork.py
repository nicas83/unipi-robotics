import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from efficient_kan import KAN
from utils import read_data, prepare_data, plot_metric


def train_model(X, Y, num_epochs, kinematics):
    start_time = time.perf_counter()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

    train_dataset = prepare_data(X_train, Y_train, batch_size=64, shuffle=True)
    test_dataset = prepare_data(X_test, Y_test, batch_size=64)

    if kinematics == 'direct':
        model = KAN([3, 22, 3], grid_size=15)  # 15
    else:
        model = KAN([3, 12, 3], grid_size=15)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # Define ReduceLROnPlateau scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3, verbose=True)
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        # Train
        model.train()
        total_loss = 0
        # total_accuracy = 0
        with tqdm(train_dataset) as pbar:
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        total_loss /= len(train_dataset)
        train_losses.append(total_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in test_dataset:
                inputs = inputs.to(device)
                labels = labels.to(device)
                output = model(inputs)
                val_loss += criterion(output, labels).item()
                # val_accuracy += (output.argmax(dim=1) == labels).float().mean().item()
        val_loss /= len(test_dataset)
        test_losses.append(val_loss)

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)
        print(f"Epoch {epoch + 1}, Train Loss: {total_loss}, Val Loss: {val_loss}")
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Grid Search eseguita in {execution_time:.4f} secondi.")
    torch.save(model.state_dict(), 'model/efficientkan/1final_model_' + kinematics + '.pth')
    np.savez('model/metrics/1final_efficient-' + kinematics + '_training_metrics.npz', train_losses=train_losses,
             val_losses=test_losses, execution_time=execution_time)
    plot_metric(train_losses, test_losses, 'Loss',
                'model/efficientkan/plot/1training_effkan_' + kinematics + '_loss.png')

    return train_losses, test_losses


def continual_learning():
    pass


def inference():
    pass


def main():
    kinematics = 'inverse'
    ######## TRAINING ###################
    X, Y = read_data('datasets/new_dataset.txt', kinematics)
    train_model(X, Y, 200, kinematics)

    ########## CONTINUAL LEARNING #################

    ########## INFERENCE #########################

    print("Test Results:")
    print(f"True actuation: [0.161 0.867 0.313]")
    # print(f"Predicted actuation: {y_pred}")


if __name__ == "__main__":
    main()
