import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
device = "cpu"
EPOCHS = 250
PATH = f"model_{EPOCHS}_single.pt"
class NeuralNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten()

        self.conv_stack_1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.conv_stack_2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.linear_stack = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.conv_stack_1(x)
        x = self.conv_stack_2(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear_stack(x)
        return x

print("Loading data...")
data = pd.read_csv('train.csv')
xtrain = data.copy()
ytrain = xtrain.pop('label')
xtrain = xtrain.to_numpy(dtype=np.float64)

xtrain = torch.from_numpy(xtrain.reshape((-1, 1, 28, 28))).float()
ytrain = torch.from_numpy(ytrain.to_numpy()).long()

xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.2, random_state=42)
print("Data loaded")

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
opt_fn = torch.optim.Adam(model.parameters(), lr=0.001)

print("Starting training...")

def acc(pred, true):
    return np.sum(np.argmax(pred.detach().numpy(), axis=1) == true.detach().numpy())/len(true)

for epoch in range(EPOCHS):

    model.train(True)

    opt_fn.zero_grad()

    outputs = model(xtrain)
    loss = loss_fn(outputs, ytrain)
    loss.backward()
    opt_fn.step()

    model.train(False)    

    print(f"EPOCH {epoch + 1}:\tLOSS: {loss}\t\t\t Training Accuracy: {acc(outputs, ytrain)}")

    if (epoch + 1) % 10 == 0:
        vout = model(xval)
        print("-"*50)
        print(f"\nValidation Accuracy: {acc(vout, yval)}\n")
        print("-"*50)

print(f"Training Complete, saving model to ./{PATH}")

torch.save(model.state_dict(), PATH)