import torch
from route_model import create_model
import numpy as np


def load_model(model, path='your model pth'):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model

def predict(model, x1, x2, x3):
    model.eval()
    with torch.no_grad():
        outputs = model(x1, x2, x3)
    return outputs

model = load_model(create_model(), path="new-result/1-model.pth")

x_1 = torch.from_numpy(np.load("salt_features.npy")).float() # 14
x_2 = torch.from_numpy(np.load("solvent_features.npy")).float() # 14
x_3 = torch.from_numpy(np.load("condition_features.npy")).float() # 6


predictions = predict(model, x_1, x_2, x_3)
for i in predictions:
    print(i)


