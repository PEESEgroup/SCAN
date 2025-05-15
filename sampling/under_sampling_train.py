import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torchmetrics.regression import PearsonCorrCoef
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import create_model, create_optimizer, create_criterion
from sklearn.neighbors import NearestNeighbors

def discretize_labels(labels):
    return np.digitize(labels, [1.0, 4.0])


def split_main_train_val(A_data, A_labels, n_splits=5):
    A_labels_discretized = discretize_labels(A_labels) 
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(A_labels, A_labels_discretized): 
        yield [data[train_idx] for data in A_data], A_labels[train_idx], [data[val_idx] for data in A_data], A_labels[val_idx]


def smoter_focus_low_y(X_list, y, k_neighbors=5, reduction_ratio=0.1, y_threshold=10):
    mask = y <= y_threshold
    X_low_list = [X[mask] for X in X_list]
    y_low = y[mask]
    X_high_list = [X[~mask] for X in X_list]
    y_high = y[~mask]

    X_low_combined = np.hstack(X_low_list)

    knn = NearestNeighbors(n_neighbors=k_neighbors)
    knn.fit(X_low_combined)
    neighbors = knn.kneighbors(X_low_combined, return_distance=False)

    num_samples = int(len(y_low) * reduction_ratio)
    sampled_indices = []
    for idx in range(len(neighbors)):
        sampled_indices.append(np.random.choice(neighbors[idx], size=1)[0])
    sampled_indices = np.unique(np.random.choice(sampled_indices, num_samples, replace=False))

    X_resampled_low_list = [X[sampled_indices] for X in X_low_list]
    y_resampled_low = y_low[sampled_indices]

    X_resampled_list = [
        np.vstack([X_high, X_low]) for X_high, X_low in zip(X_high_list, X_resampled_low_list)
    ]
    y_resampled = np.hstack([y_high, y_resampled_low])

    return X_resampled_list, y_resampled

def train_model(train_data, train_labels, val_data, val_labels, epochs=10, type=False):

    model = create_model().to(device)
    optimizer = create_optimizer(model)
    criterion = create_criterion()
    
    train_data_balanced, train_labels_balanced = smoter_focus_low_y(train_data, train_labels, k_neighbors=5, reduction_ratio=your ratio, y_threshold=your cutoff)

    train_dataset = TensorDataset(torch.FloatTensor(train_data_balanced[0]), torch.FloatTensor(train_data_balanced[1]),
                                  torch.FloatTensor(train_data_balanced[2]), torch.FloatTensor(train_labels_balanced))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    val_dataset = TensorDataset(torch.FloatTensor(val_data[0]), torch.FloatTensor(val_data[1]),
                                  torch.FloatTensor(val_data[2]), torch.FloatTensor(val_labels))
    val_loader = DataLoader(val_dataset, batch_size=256)

    for epoch in range(epochs):
        model.train()
        for x_1, x_2, x_3, y in train_loader:
            x_1, x_2, x_3, y = x_1.to(device), x_2.to(device), x_3.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x_1, x_2, x_3)
            loss = criterion(outputs.squeeze(), y)
            loss.backward()
            optimizer.step()

        if type:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_1, x_2, x_3, y in val_loader:
                    x_1, x_2, x_3, y = x_1.to(device), x_2.to(device), x_3.to(device), y.to(device)
                    outputs = model(x_1, x_2, x_3, y)
                    val_loss += criterion(outputs.squeeze(), y).item()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss / len(val_loader):.4f}")
    return model


def print_results(labels, preds, name):
    print(f"{name} Results:")
    print("MSE:", mean_squared_error(labels, preds))
    print("MAE:", mean_absolute_error(labels, preds))
    # pearson = PearsonCorrCoef()
    # labels = labels.reshape(-1,1)
    preds = preds.squeeze()
    print("R2:", np.corrcoef(labels, preds)[0, 1])
    results = np.vstack((labels, preds)).T
    np. savetxt(name+"-result.txt", results)


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to the {path}")


def main(A_data, A_labels):
    flag = 1
    for S, S_labels, T, T_labels in split_main_train_val(A_data, A_labels):
        model = train_model(S, S_labels, T, T_labels, epochs=1000, type=True)

        model.eval()
        T_1, T_2, T_3 = torch.FloatTensor(T[0]), torch.FloatTensor(T[1]), torch.FloatTensor(T[2])
        T_1, T_2, T_3 = T_1.to(device), T_2.to(device), T_3.to(device)
        with torch.no_grad():
            predictions = np.array(model(T_1, T_2, T_3).cpu())
        print_results(T_labels, predictions, name=str(flag)+'-train')
        save_model(model, path=str(flag)+"-model.pth")
        flag += 1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
A_feature_1 = np.load("../../salt_features.npy") 
A_feature_2 = np.load("../../solvent_features.npy") 
A_feature_3 = np.load("../../condition_features.npy") 
A_target = np.loadtxt("../../conductivity_target.txt")
A_dataset = [A_feature_1, A_feature_2, A_feature_3]

main(A_dataset, A_target)
