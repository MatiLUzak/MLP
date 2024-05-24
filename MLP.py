import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

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

def generate_data(n_samples=100):
    X = np.linspace(-10, 10, n_samples).reshape(-1, 1)
    y = 2 * X + 1 + np.random.normal(0, 2, (n_samples, 1))  # Dodajemy szum dla realizmu
    return X, y

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
    input_size = 1
    hidden_layers = [10, 5]
    output_size = 1

    model = MLP(input_size, hidden_layers, output_size)
    model.apply(initialize_weights)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    X_train, y_train = generate_data(80)
    X_test, y_test = generate_data(20)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=100)

    example_input = torch.tensor([[6]], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        prediction = model(example_input)
        print(f'Predykcja dla {[6]}: {prediction.numpy()}')

if __name__ == "__main__":
    main()
