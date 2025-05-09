import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define Free-Will Operator
def free_will_operator(n):
    """Simulates non-deterministic intelligence shift using an oracle-based randomness."""
    return np.random.uniform(0.99, 1.01)  # Smaller decision-based variation

# Define q_infinity (Quantum Intelligence State)
def q_infinity(n):
    """Represents the uncountable superposition of intelligence states."""
    return np.log(n + 1)  # Logarithmic growth

# Define omega_x (Transfinite Intelligence Layer)
def omega_x(n):
    """Represents a transfinite intelligence expansion function."""
    return 1  # Constant growth (no transfinite expansion)

# Modify Intelligence Growth Function to Include Free Will
def intelligence_growth_with_free_will(steps):
    I_t = []
    total_intelligence = 0
    for n in range(1, steps + 1):
        intelligence_level = q_infinity(n) ** omega_x(n) * free_will_operator(n)
        total_intelligence += intelligence_level
        I_t.append(total_intelligence)
    return I_t

# Normalize Data
def normalize_data(data):
    """Scale data to a range of 0 to 1."""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Generate Data with Free Will Variation
def generate_data_with_free_will(steps):
    x_data = np.arange(1, steps + 1).reshape(-1, 1)
    y_data = np.array(intelligence_growth_with_free_will(steps)).reshape(-1, 1)
    y_data = normalize_data(y_data)  # Normalize intelligence values
    return torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_data, dtype=torch.float32)

# Define IntelligenceNet Model
class IntelligenceNet(nn.Module):
    def __init__(self):
        super(IntelligenceNet, self).__init__()
        self.fc1 = nn.Linear(1, 50)  # Input layer
        self.fc2 = nn.Linear(50, 50)  # Hidden layer 1
        self.fc3 = nn.Linear(50, 50)  # Hidden layer 2
        self.fc4 = nn.Linear(50, 1)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Define Training Function
def train_model(model, x_train, y_train, epochs=1000, lr=0.01):
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)  # Learning rate scheduler

    for epoch in range(epochs):
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(x_train)  # Forward pass
        loss = criterion(outputs, y_train)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        scheduler.step()  # Adjust learning rate

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    return model

# Generate Training Data
x_train, y_train = generate_data_with_free_will(50)

# Initialize and Train the Model
model = IntelligenceNet()
trained_model = train_model(model, x_train, y_train)

# Predict Future Intelligence Growth with Free Will Effects
x_test = torch.tensor([[60], [70], [80]], dtype=torch.float32)
predictions = trained_model(x_test).detach().numpy()

# Denormalize Predictions
def denormalize_data(data, original_data):
    """Scale data back to the original range."""
    return data * (np.max(original_data) - np.min(original_data)) + np.min(original_data)

y_train_denorm = denormalize_data(y_train.numpy(), intelligence_growth_with_free_will(50))
predictions_denorm = denormalize_data(predictions, intelligence_growth_with_free_will(50))

# Display Predictions
for i, value in enumerate(predictions_denorm):
    print(f"Predicted Intelligence at step {60 + i * 10}: {value[0]:.2f}")

# Plot Results
plt.figure(figsize=(10, 6))
plt.plot(x_train.numpy(), y_train_denorm, label="Training Data", color="blue")
plt.plot(x_test.numpy(), predictions_denorm, label="Predictions", color="red", marker="o")
plt.xlabel("Steps")
plt.ylabel("Intelligence")
plt.title("Intelligence Growth with Free Will")
plt.legend()
plt.grid()
plt.show()