import torch
import torch.nn as nn
import torch.optim as optim
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        x = torch.relu(self.fc1(x)) 
        x = self.fc2(x)
        return x
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
from torch.utils.data import DataLoader, TensorDataset
train_data = torch.randn(1000, 784)  # 1000 samples, 784 features each
train_labels = torch.randint(0, 10, (1000,))  # 1000 labels (10 classes)
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
for epoch in range(10):  # 10 epochs
    running_loss = 0.0
    for data, labels in train_loader:
        optimizer.zero_grad()  # Clear gradients
        outputs = model(data)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")