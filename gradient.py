import torch
import numpy as np
from route_model import create_model


model = create_model()
model.load_state_dict(torch.load("your model.pth")) 

feature_1 = torch.from_numpy(np.load("data/salt_features.npy")).float() 
feature_2 = torch.from_numpy(np.load("data/solvent_features.npy")).float() 
feature_3 = torch.from_numpy(np.load("data/condition_features.npy")).float() 

model.eval()

inputs = [feature_1, feature_2, feature_3]

inputs = [inp.requires_grad_(True) for inp in inputs]


output = model(*inputs)

output.backward(torch.ones_like(output))

feature_importance = [inp.grad for inp in inputs]

importance_1 = feature_importance[0].mean(axis=0)
importance_2 = feature_importance[1].mean(axis=0)
importance_3 = feature_importance[2].mean(axis=0)

for i in importance_1:
    print(i.item())


