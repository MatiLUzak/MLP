import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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


def train_model(model, criterion, optimizer, data_loader, num_epochs=1000, stop_criteria=1e-4):
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(data_loader)

        if epoch % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        if epoch_loss < stop_criteria:
            print(f'Training stopped at epoch {epoch + 1} due to low loss: {epoch_loss:.4f}')
            break


def main():
    input_size = 2
    hidden_layers = [32,16,8,4]
    output_size = 2

    model = MLP(input_size, hidden_layers, output_size)
    model.apply(initialize_weights)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    X_train = torch.tensor([[i, i + 1] for i in range(1, 51)], dtype=torch.float32)
    y_train = torch.tensor([[i *i, (i + 1)*(i+1)] for i in range(1, 51)], dtype=torch.float32)

    dataset = TensorDataset(X_train, y_train)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    train_model(model, criterion, optimizer, data_loader, num_epochs=1000, stop_criteria=1e-4)

    X_test = torch.tensor([[6, 7]], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        prediction = model(X_test)
        print(f'Predykcja dla {[6, 7]}: {prediction.numpy()}')


if __name__ == "__main__":
    main()
