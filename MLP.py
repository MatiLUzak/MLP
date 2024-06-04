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
test_mse_values = {}

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
        #model.train()
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

        #model.eval()
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

    test_mse_values[config_name] = test_loss

    all_predictions = []
    all_targets = []
    #model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            all_predictions.extend(outputs.numpy())
            all_targets.extend(targets.numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    errors = np.sqrt(np.sum((all_predictions - all_targets) ** 2, axis=1))
    distributions[config_name] = errors

    predictions[config_name] = all_predictions

    return model

def plot_graphs(best_train_losses, best_test_losses, best_distributions, best_predictions, X_test, y_test, reference_test_mse, best_model_config):
    plt.figure(figsize=(12, 6))
    for config_name, losses in best_train_losses.items():
        plt.plot(range(1, len(losses) + 1), losses, label=config_name)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_loss2_all_variants.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    for config_name, losses in best_test_losses.items():
        plt.plot(range(1, len(losses) + 1), losses, label=config_name)
    plt.axhline(y=reference_test_mse, color='r', linestyle='--', label='Reference Test set MSE')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Test Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('test_loss2_all_variants.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    for config_name, errors in best_distributions.items():
        sorted_errors = np.sort(errors)
        cdf = np.arange(len(sorted_errors)) / float(len(sorted_errors))
        plt.plot(sorted_errors, cdf, label=config_name)
    plt.xlabel('Error')
    plt.ylabel('Cumulative Distribution')
    plt.title('Error Distributions (CDF)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('error_distribution2_all_variants.png')
    plt.close()

    best_model_predictions = best_predictions[best_model_config]

    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], color='red', label='Zmierzona')
    #plt.scatter(y_test[:, 0], y_test[:, 1], color='green', label='Rzeczywista')
    plt.scatter(best_model_predictions[:, 0], best_model_predictions[:, 1], color='blue', label='Skorygowana przez sieć')
    plt.scatter(y_test[:, 0], y_test[:, 1], color='green', label='Rzeczywista')
    plt.title(f'Pozycje: rzeczywista, zmierzona i skorygowana przez sieć - {best_model_config}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig(f'positions2_{best_model_config}.png')
    plt.close()


def main():
    global criterion
    base_path = 'C:/Users/mluza/PycharmProjects/MLP/dane/'
    X_train, y_train, X_test, y_test = load_data(base_path)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    input_size = 2
    output_size = 2

    hidden_layer_configs = [
        [16],
        [16, 8],
        [32, 16, 8]
    ]

    learning_rate = 0.001
    best_train_losses = {}
    best_test_losses = {}
    best_distributions = {}
    best_predictions = {}

    for hidden_layers in hidden_layer_configs:
        best_config = None
        best_test_loss = float('inf')
        for run in range(3):
            config_name = f'{hidden_layers}_lr_{learning_rate}_run_{run+1}'
            print(f'Training model with hidden layers: {hidden_layers}, learning rate: {learning_rate}, run: {run + 1}')
            model = MLP(input_size, hidden_layers, output_size)
            model.apply(initialize_weights)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=25, config_name=config_name)
            if train_losses[config_name][-1] < best_test_loss:
                best_test_loss = train_losses[config_name][-1]
                best_config = config_name

        best_train_losses[best_config] = train_losses[best_config]
        best_test_losses[best_config] = test_losses[best_config]
        best_distributions[best_config] = distributions[best_config]
        best_predictions[best_config] = predictions[best_config]

    reference_test_mse = criterion(X_test, y_test).item()
    print(f'Reference Test MSE: {reference_test_mse}')
    best_model_config = min(best_test_losses, key=lambda k: best_test_losses[k][-1])

    plot_graphs(best_train_losses, best_test_losses, best_distributions, best_predictions, X_test, y_test, reference_test_mse, best_model_config)

if __name__ == "__main__":
    main()
