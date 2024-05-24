import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

    # Wczytywanie danych statycznych
    for part in ['f8', 'f10']:
        stat_path = os.path.join(base_path, part, 'stat')
        for file in os.listdir(stat_path):
            if file.endswith('.csv'):
                file_path = os.path.join(stat_path, file)
                data = pd.read_csv(file_path, header=None)
                if data.isnull().values.any():
                    print(f'NaN values found in {file}')  # Sprawdzenie NaN
                X_train_list.append(data.iloc[:, :2].values)
                y_train_list.append(data.iloc[:, 2:].values)
                print(f'Loaded {file} with shape {data.shape}')  # Debugowanie

    # Wczytywanie danych dynamicznych
    for part in ['f8', 'f10']:
        dyn_path = os.path.join(base_path, part, 'dyn')
        for file in os.listdir(dyn_path):
            if file.endswith('.csv'):
                file_path = os.path.join(dyn_path, file)
                data = pd.read_csv(file_path, header=None)
                if data.isnull().values.any():
                    print(f'NaN values found in {file}')  # Sprawdzenie NaN
                X_test_list.append(data.iloc[:, :2].values)
                y_test_list.append(data.iloc[:, 2:].values)
                print(f'Loaded {file} with shape {data.shape}')  # Debugowanie

    # Łączenie list numpy.ndarray do jednego numpy.ndarray
    X_train = np.vstack(X_train_list)
    y_train = np.vstack(y_train_list)
    X_test = np.vstack(X_test_list)
    y_test = np.vstack(y_test_list)

    # Debugowanie: porównanie długości X_train i y_train przed usunięciem wartości NaN
    print(f'X_train length: {len(X_train)}, y_train length: {len(y_train)}')
    print(f'X_test length: {len(X_test)}, y_test length: {len(y_test)}')

    # Sprawdzanie brakujących wartości i ich usunięcie
    if np.isnan(X_train).any() or np.isnan(y_train).any() or np.isnan(X_test).any() or np.isnan(y_test).any():
        print('NaN values found in data. Removing...')
        X_train = X_train[~np.isnan(X_train).any(axis=1)]
        y_train = y_train[~np.isnan(y_train).any(axis=1)]
        X_test = X_test[~np.isnan(X_test).any(axis=1)]
        y_test = y_test[~np.isnan(y_test).any(axis=1)]

    # Debugowanie: porównanie długości X_train i y_train po usunięciu wartości NaN
    print(f'X_train length after NaN removal: {len(X_train)}, y_train length after NaN removal: {len(y_train)}')
    print(f'X_test length after NaN removal: {len(X_test)}, y_test length after NaN removal: {len(y_test)}')

    # Normalizacja danych
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Debugowanie rozmiarów i zakresów wartości
    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_test shape: {y_test.shape}')
    print(f'X_train range: {X_train.min()} to {X_train.max()}')
    print(f'X_test range: {X_test.min()} to {X_test.max()}')

    # Konwersja na tensory
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, y_train, X_test, y_test

def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=100):
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

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

        test_loss /= len(test_loader)

        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}')

        if test_loss < 1e-4:
            print(f'Training stopped at epoch {epoch + 1} due to low test loss: {test_loss:.4f}')
            break

def main():
    base_path = 'C:/Users/mluza/PycharmProjects/MLP/dane/'  # Ustaw właściwą ścieżkę do katalogu 'dane'
    X_train, y_train, X_test, y_test = load_data(base_path)

    # Dodaj debugowanie, aby sprawdzić rozmiary tensorów
    print(f'X_train tensor shape: {X_train.shape}')
    print(f'y_train tensor shape: {y_train.shape}')
    print(f'X_test tensor shape: {X_test.shape}')
    print(f'y_test tensor shape: {y_test.shape}')

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    input_size = 2
    output_size = 2

    hidden_layer_configs = [
        [32],
        [32, 16],
        [32, 16, 8]
    ]

    for hidden_layers in hidden_layer_configs:
        for run in range(3):
            model = MLP(input_size, hidden_layers, output_size)
            model.apply(initialize_weights)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            print(f'Training model with hidden layers: {hidden_layers}, run: {run + 1}')
            train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=100)

    example_input = torch.tensor([[6000, 7000]], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        prediction = model(example_input)
        print(f'Predykcja dla {[6000, 7000]}: {prediction.numpy()}')

if __name__ == "__main__":
    main()
