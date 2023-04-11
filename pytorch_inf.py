import torch
import numpy as np
import pandas as pd
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten()

        self.conv_stack_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )

        self.conv_stack_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )
        self.linear_stack = nn.Sequential(
            nn.Linear(576, 240),
            nn.ReLU(),
            nn.Linear(240, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        x = self.conv_stack_1(x)
        x = self.conv_stack_2(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear_stack(x)
        return x
    

PATH = "model_250_single_large.pt"
OUT = 'submission_model_250_single_large.csv'
device = "cpu"


model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(PATH))
model.eval()

data = pd.read_csv("test.csv").to_numpy(dtype=np.float64)
data = torch.from_numpy(data.reshape((-1, 1, 28, 28))).float()

preds = np.argmax(model(data).detach().numpy(), axis=1)

with open(OUT, 'w') as f:
    f.write('ImageId,Label\n')
    for i, p in enumerate(preds):
        f.write(f'{i+1},{p}\n')

print(f"Wrote {len(preds)} predictions to {OUT}")