#  imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transfroms

# Create Fully connected network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameter
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5


# load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transfroms.ToTensor(),download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transfroms.ToTensor(),download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # get data to cuda
        data = data.to(device=device)
        targets = targets.to(device=device)
        # get  to correct shape
        data = data.reshape(data.shape[0], -1)
        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient desent or adam step
        optimizer.step()

# check  accuracy on training and test to how good our model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking acc on the training data")
    else:
        print("Checking acc on the test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuarcy {float(num_correct)/float(num_samples)*100:.2f}')
    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

