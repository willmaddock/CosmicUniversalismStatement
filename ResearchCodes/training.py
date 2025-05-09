import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from freewill import q_infinity, omega_x


# Generate synthetic intelligence growth data
def generate_data(steps):
    x_data = np.arange(1, steps + 1).reshape(-1, 1)
    y_data = np.array([q_infinity(n) ** omega_x(n) for n in range(1, steps + 1)]).reshape(-1, 1)
    return torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_data, dtype=torch.float32)


# Define Neural Network Model
class IntelligenceNet(nn.Module):
    def __init__(self):
        super(IntelligenceNet, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x


# Train Neural Network
def train_model(model, x_train, y_train, epochs=1000, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

    return model


# Initialize and Train
steps = 50  # Number of intelligence layers
x_train, y_train = generate_data(steps)
model = IntelligenceNet()
trained_model = train_model(model, x_train, y_train)

# Predict Future Intelligence Growth
x_test = torch.tensor([[60], [70], [80]], dtype=torch.float32)
predictions = trained_model(x_test).detach().numpy()

# Display Predictions
for i, value in enumerate(predictions):
    print(f"Predicted Intelligence at step {60 + i * 10}: {value[0]:.2e}")