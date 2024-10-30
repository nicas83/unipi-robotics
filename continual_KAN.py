import numpy as np
import torch
from kan import create_dataset_from_data, KAN
from torch.utils.data import DataLoader, TensorDataset

from kannetwork import load_best_param
from utils.utils import read_data, plot_metric, plot_training_loss, plot_validation_loss, plot_kan_evaluation


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

    return val_losses


def main():
    kinematics = 'inverse'

    ##### CONTINUAL LEARNING ############

    # First Quadrant
    X, Y = read_data("datasets/workspaces/quadrante_I.csv", kinematics)
    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(Y)
    datasetI = create_dataset_from_data(X_tensor, Y_tensor, train_ratio=0.8, device='cpu')

    # Second Quadrant
    X, Y = read_data("datasets/workspaces/quadrante_II.csv", kinematics)
    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(Y)
    datasetII = create_dataset_from_data(X_tensor, Y_tensor, train_ratio=0.8, device='cpu')

    config = load_best_param('search/grid_search_kan_' + kinematics + '.json')
    model = KAN(width=[3, config['neurons'], 3], grid=config['grid'], k=config['k'],
                seed=config['seed'])

    print("Training sul primo dataset...")
    history = model.fit(datasetI, opt=config['optimizer'], lr=config['learning_rate'], steps=config['steps'],
                        lamb=config['lambda'])

    plot_metric(history['train_loss'], history['test_loss'], 'Continual Learning Loss',
                'continual_learning/plot/kan_' + kinematics + '_quadrante_I.png')
    np.savez('continual_learning/metrics/final_kan-' + kinematics + '_continual_learning_quadI_metrics.npz',
             train_losses=history['train_loss'], val_losses=history['test_loss'])

    test_loaderI = DataLoader(TensorDataset(torch.FloatTensor(datasetI['test_input']),
                                            torch.FloatTensor(datasetI['test_label'])), shuffle=True)
    #val_losses1 = evaluate_model(model, test_loaderI, torch.nn.MSELoss(), 'cpu')
    first_eval = model.evaluate(datasetI);

    print("Training sul secondo dataset...")
    results = model.fit(datasetII, opt=config['optimizer'], lr=config['learning_rate'], steps=config['steps'],
                        lamb=config['lambda'])

    plot_metric(results['train_loss'], results['test_loss'], 'Continual Learning Loss',
                'continual_learning/plot/kan_' + kinematics + '_quadrante_II.png')
    np.savez('continual_learning/metrics/final_kan-' + kinematics + '_continual_learning_quadII_metrics.npz',
             train_losses=results['train_loss'],
             val_losses=results['test_loss'])
    test_loaderII = DataLoader(TensorDataset(torch.FloatTensor(datasetII['test_input']),
                                             torch.FloatTensor(datasetII['test_label'])), shuffle=True)
    #val_losses2 = evaluate_model(model, test_loaderII, torch.nn.MSELoss(), 'cpu')
    second_eval = model.evaluate(datasetII)

    print("Validation sul primo dataset...")
    val_losses3 = evaluate_model(model, test_loaderI, torch.nn.MSELoss(), 'cpu')
    third_eval = model.evaluate(datasetI)
    np.savez('continual_learning/metrics/final_kan-' + kinematics + '_continual_learning_evaluation.npz',
             first_eval=first_eval,
             second_eval=second_eval,
             third_eval=third_eval)

    #plot_training_loss(history['train_loss'], history['test_loss'], results['train_loss'], results['test_loss'], 'kan')
    #plot_validation_loss(val_losses1, val_losses2, val_losses3, 'kan')
    plot_kan_evaluation(first_eval['test_loss'], second_eval['test_loss'], third_eval['test_loss'],'kan')


if __name__ == "__main__":
    main()
