import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn as nn
import torch.nn.init as init
from utils.utils import read_data, prepare_data, plot_metric
from sklearn.model_selection import train_test_split


class RBFKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_centers, alpha=1.0):
        super(RBFKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_centers = num_centers
        self.alpha = alpha

        self.centers = nn.Parameter(torch.empty(num_centers, input_dim))
        init.xavier_uniform_(self.centers)

        self.weights = nn.Parameter(torch.empty(num_centers, output_dim))
        init.xavier_uniform_(self.weights)

    def multiquadratic_rbf(self, distances):
        return (1 + (self.alpha * distances) ** 2) ** 0.5

    def gaussian_rbf(self, distances):
        return torch.exp(-self.alpha * distances ** 2)

    def forward(self, x):
        distances = torch.cdist(x, self.centers)
        # basis_values = self.multiquadratic_rbf(distances)
        basis_values = self.multiquadratic_rbf(distances)
        output = torch.sum(basis_values.unsqueeze(2) * self.weights.unsqueeze(0), dim=1)
        return output


class RBFKAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_centers):
        super(RBFKAN, self).__init__()
        self.rbf_kan_layer = RBFKANLayer(input_dim, hidden_dim, num_centers)
        self.output_weights = nn.Parameter(torch.empty(hidden_dim, output_dim))
        init.xavier_uniform_(self.output_weights)

    def forward(self, x):
        x = self.rbf_kan_layer(x)
        x = torch.relu(x)
        x = torch.matmul(x, self.output_weights)
        return x

kinematics = 'inverse'
X, Y = read_data('datasets/dataset.txt', kinematics)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

train_dataset = prepare_data(X_train, Y_train, batch_size=256, shuffle=True)
test_dataset = prepare_data(X_test, Y_test, batch_size=256)

model = RBFKAN(3, 21, 3, num_centers=5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
criterion = nn.MSELoss()

# Define ReduceLROnPlateau scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)
train_losses=[]
test_losses=[]
start_time = time.perf_counter()
for epoch in range(200):
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
            # accuracy = (output.argmax(dim=1) == labels).float().mean()
            total_loss += loss.item()
            # total_accuracy += accuracy.item()
            pbar.set_postfix(loss=loss.item())
    total_loss /= len(train_dataset)
    train_losses.append(total_loss)
    # total_accuracy /= len(trainloader)

    # Validation
    model.eval()
    val_loss = 0
    # val_accuracy = 0
    with torch.no_grad():
        for inputs, labels in test_dataset:
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)
            val_loss += criterion(output, labels).item()
            #val_accuracy += (output.argmax(dim=1) == labels).float().mean().item()
    val_loss /= len(test_dataset)
    test_losses.append(val_loss)
    #val_accuracy /= len(valloader)

    # Step the scheduler based on validation loss
    scheduler.step(val_loss)
    print(f"Epoch {epoch + 1}, Train Loss: {total_loss}, Val Loss: {val_loss}")
end_time=time.perf_counter()
execution_time=end_time-start_time
np.savez('model/metrics/final_kan-rbf-inverse_training_metrics.npz', train_losses=train_losses,
             val_losses=test_losses, train_execution_time=execution_time)
plot_metric(train_losses, test_losses, 'Loss', 'model/rbfkan/plot/training_effkan_' + kinematics + '_loss.png')

model.eval()
actuator = np.array([
    [.1, .1, .1],
    #     # [.2, .18, .2],
    #     # [.2, .2, .18]
])
poses = np.array([[.0, .0, .1]])
X_test_tensor = torch.FloatTensor(poses)
# Esegui la predizione
with torch.no_grad():
    y_pred = model(X_test_tensor).numpy()

print("Test Results:")
print(f"True actuation: {actuator}")
print(f"Predicted actuation: {y_pred}")

