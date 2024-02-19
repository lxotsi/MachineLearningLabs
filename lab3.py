from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
from torch import nn
from torch.utils.data import DataLoader

# Load FashionMNIST dataset for training and testing
training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

# Create data loaders for training and testing
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# Define neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Flatten layer to convert 2D input images to 1D tensors
        self.flatten = nn.Flatten()
        # Sequential stack of linear and ReLU layers
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Initialize loss function
loss_function = nn.CrossEntropyLoss()

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork().to(device)

# Define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Training function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss.current = loss.item(), batch * len(X)
            print("Loss", loss, "Current", batch, "of", size / 64)

# Testing function
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print("Accuracy", 100 * correct, "%")

# Number of epochs for training
epoch = 10

# Training loop
for t in range(epoch):
    print("Epoch", t)
    train(train_dataloader, model, loss_function, optimizer)
    test(test_dataloader, model, loss_function)

# Save model state
torch.save(model.state_dict(), "model.pth")
