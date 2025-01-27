# PYTORCH CODE
import torch
import torch.nn as nn
import torch.optim as optim

# Simple Neural Network
model = nn.Sequential(
    nn.Linear(784, 64),  # Input layer: 784 -> Hidden layer: 64
    nn.ReLU(),           # Activation function
    nn.Linear(64, 10)    # Hidden layer: 64 -> Output layer: 10
)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy data
train_data = torch.randn(100, 784)  # 100 samples, 784 features
train_labels = torch.randint(0, 10, (100,))  # 100 labels (0-9)

# Training loop
for epoch in range(10):  # 5 epochs
    optimizer.zero_grad()
    loss = criterion(model(train_data), train_labels)  # Forward + Loss
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")