import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split



def load_data():
    feature_1 = torch.from_numpy(np.load("data/salt_features.npy")).float() # 14
    feature_2 = torch.from_numpy(np.load("data/solvent_features.npy")).float() # 14
    feature_3 = torch.from_numpy(np.load("data/condition_features.npy")).float() # 5
    target = torch.from_numpy(np.loadtxt("data/conductivity_target.txt")).float().unsqueeze(1)  # target
    batch_size = 320


    # Create dataset
    dataset = TensorDataset(feature_1, feature_2, feature_3, target)

    # Split dataset into training, validation, test sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    #
    # Create DataLoader for batch training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
