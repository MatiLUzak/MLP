import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

train_losses = {}
test_losses = {}
distributions = {}
predictions = {}

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(MLP, self).__init__()
        layers = []
        prev = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev, hidden_size))
            layers.append(nn.ReLU())
            prev = hidden_size
        layers.append(nn.Linear(prev, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)

def load_data(base_path):
    X_train_list, y_train_list, X_test_list, y_test_list = [], [], [], []

    def load_csv_files(path, X_list, y_list):
        for file in os.listdir(path):
            if file.endswith('.csv'):
                file_path = os.path.join(path, file)
                data = pd.read_csv(file_path, header=None)
                data = data.dropna().loc[data.apply(lambda x: len(x) == 4, axis=1)]
                if data.isnull().values.any():
                    print(f'NaN values found in {file}')
                X_list.append(data.iloc[:, :2].values)
                y_list.append(data.iloc[:, 2:].values)
                print(f'Loaded {file} with shape {data.shape}')

    for part in ['f8', 'f10']:
        stat_path = os.path.join(base_path, part, 'stat')
        load_csv_files(stat_path, X_train_list, y_train_list)

    for part in ['f8', 'f10']:
        dyn_path = os.path.join(base_path, part, 'dyn')
        load_csv_files(dyn_path, X_test_list, y_test_list)

    X_train = np.vstack(X_train_list)
    y_train = np.vstack(y_train_list)
    X_test = np.vstack(X_test_list)
    y_test = np.vstack(y_test_list)

    print(f'X_train length: {len(X_train)}, y_train length: {len(y_train)}')
    print(f'X_test length: {len(X_test)}, y_test length: {len(y_test)}')

    if np.isnan(X_train).any() or np.isnan(y_train).any() or np.isnan(X_test).any() or np.isnan(y_test).any():
        print('NaN values found in data. Removing...')
        X_train = X_train[~np.isnan(X_train).any(axis=1)]
        y_train = y_train[~np.isnan(y_train).any(axis=1)]
        X_test = X_test[~np.isnan(X_test).any(axis=1)]
        y_test = y_test[~np.isnan(y_test).any(axis=1)]

    print(f'X_train length after NaN removal: {len(X_train)}, y_train length after NaN removal: {len(y_train)}')
    print(f'X_test length after NaN removal: {len(X_test)}, y_test length after NaN removal: {len(y_test)}')

    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_test shape: {y_test.shape}')
    print(f'X_train range: {X_train.min()} to {X_train.max()}')
    print(f'X_test range: {X_test.min()} to {X_test.max()}')

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, y_train, X_test, y_test

def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=100, config_name=''):
    epoch_train_losses = []
    epoch_test_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        epoch_train_losses.append(epoch_loss)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        epoch_test_losses.append(test_loss)

        if test_loss < 1e-4:
            break
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}')

    train_losses[config_name] = epoch_train_losses
    test_losses[config_name] = epoch_test_losses

    # Przechowywanie predykcji do dystrybuanty
    all_predictions = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            all_predictions.extend(outputs.numpy())
            all_targets.extend(targets.numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    errors = np.sqrt(np.sum((all_predictions - all_targets) ** 2, axis=1))
    distributions[config_name] = errors

    # Store the full test set predictions
    predictions[config_name] = all_predictions

    return model

def plot_graphs(train_losses, test_losses, distributions, predictions, X_test, y_test, test_mse):
    # Plotting training loss
    plt.figure(figsize=(12, 6))
    for config_name, losses in train_losses.items():
        plt.plot(losses, label=config_name)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plotting test loss with reference line
    plt.figure(figsize=(12, 6))
    for config_name, losses in test_losses.items():
        plt.plot(losses, label=config_name)
    plt.axhline(y=test_mse, color='r', linestyle='--', label='Test set MSE')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Test Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plotting error distributions
    plt.figure(figsize=(12, 6))
    for config_name, errors in distributions.items():
        sorted_errors = np.sort(errors)
        cdf = np.arange(len(sorted_errors)) / float(len(sorted_errors))
        plt.plot(sorted_errors, cdf, label=config_name)
    plt.xlabel('Error')
    plt.ylabel('Cumulative Distribution')
    plt.title('Error Distributions (CDF)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Scatter plot for the best performing network variant
    best_config = min(test_losses, key=lambda k: test_losses[k][-1])
    best_model_predictions = predictions[best_config]

    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], color='red', label='Zmierzona')
    plt.scatter(y_test[:, 0], y_test[:, 1], color='green', label='Rzeczywista')
    plt.scatter(best_model_predictions[:, 0], best_model_predictions[:, 1], color='blue',
                label='Skorygowana przez sieć')
    plt.title('Pozycje: rzeczywista, zmierzona i skorygowana przez sieć')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig('positions.png')
    plt.show()

def main():
    base_path = 'C:/Users/mluza/PycharmProjects/MLP/dane/'
    X_train, y_train, X_test, y_test = load_data(base_path)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    input_size = 2
    output_size = 2

    hidden_layer_configs = [
        [15, 15]
    ]

    for hidden_layers in hidden_layer_configs:
        for run in range(3):
            config_name = f'{hidden_layers}_run_{run+1}'
            print(f'Training model with hidden layers: {hidden_layers}, run: {run + 1}')
            model = MLP(input_size, hidden_layers, output_size)
            model.apply(initialize_weights)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            trained_model = train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=10, config_name=config_name)

            example_input = torch.tensor([[7526.0, 602.0]], dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                prediction = trained_model(example_input)
                print(f'Predykcja dla {[7526.0, 602.0]}: {prediction.numpy()}')

    # Compute test set MSE reference line
    test_mse = criterion(y_test.clone().detach(), y_test.clone().detach()).item()

    # Draw plots
    plot_graphs(train_losses, test_losses, distributions, predictions, X_test, y_test, test_mse)

if __name__ == "__main__":
    main()
