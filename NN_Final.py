import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


# 🔥 Loss Functions
def nlinex_loss_2_0_0_4(pred, target):
    D = pred - target
    return (2.0 * (torch.exp(0.4 * D) + 0.4 * D ** 2 - 0.4 * D - 1)).mean()


def nlinex_loss_2_5_0_5(pred, target):
    D = pred - target
    return (2.5 * (torch.exp(0.5 * D) + 0.5 * D ** 2 - 0.5 * D - 1)).mean()


def nlinex_loss_1_5_0_6(pred, target):
    D = pred - target
    return (1.5 * (torch.exp(0.6 * D) + 0.6 * D ** 2 - 0.6 * D - 1)).mean()


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    bce = -(target * torch.log(pred + 1e-8) + (1 - target) * torch.log(1 - pred + 1e-8))
    pt = torch.where(target == 1, pred, 1 - pred)
    return (alpha * (1 - pt) ** gamma * bce).mean()


def lovasz_hinge_loss(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    errors = (1.0 - target) * pred + target * (1.0 - pred)
    errors_sorted, perm = torch.sort(errors, descending=True)
    target_sorted = target[perm]
    intersection = target_sorted.cumsum(0)
    union = target_sorted.cumsum(0) + (1.0 - target_sorted).cumsum(0)
    jaccard = 1.0 - intersection / (union + 1e-8)
    return torch.mean((1.0 - target_sorted) * errors_sorted * torch.cumsum(jaccard, 0))


# 🧠 Neural Net
class NeuroNet(nn.Module):
    def __init__(self, input_dim=100):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x): return self.network(x)


# 📦 Load Composite Class Data from local directory
def load_custom_binary_data(data_dir, image_size=(64, 64), n_features=100):
    config = {
        'NonDemented': {'label': 0, 'count': float('inf')},  # Use all available
        'MildDemented': {'label': 1, 'count': float('inf')},
        'VeryMildDemented': {'label': 1, 'count': float('inf')},
        'ModerateDemented': {'label': 1, 'count': float('inf')}
    }

    X, y = [], []
    for cls, spec in config.items():
        cls_dir = os.path.join(data_dir, cls)
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.seed(42)
        random.shuffle(files)
        selected = files  # Use all files without capping
        for file in selected:
            img = Image.open(file).convert('L').resize(image_size)
            X.append(np.array(img).flatten())
            y.append(spec['label'])

    X = np.array(X)
    y = np.array(y).astype(np.float32)
    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=n_features).fit_transform(X_scaled)

    X_train, X_val, y_train, y_val = train_test_split(
        X_pca, y, test_size=0.2, stratify=y, random_state=42
    )

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1),
        torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
    )


# ⚙️ Train Loop
def train_model(model, criterion, X_train, y_train, X_val, y_val, epochs=50, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses, train_accs, val_accs, val_aucs = [], [], [], [], []

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        train_accs.append(accuracy_score(y_train.numpy(), (preds >= 0.5).float().numpy()))

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val).clamp(1e-7, 1 - 1e-7)
            val_losses.append(criterion(val_preds, y_val).item())
            val_accs.append(accuracy_score(y_val.numpy(), (val_preds >= 0.5).float().numpy()))
            val_aucs.append(
                roc_auc_score(y_val.numpy(), val_preds.numpy()) if len(np.unique(y_val.numpy())) > 1 else float('nan'))

    return train_losses, val_losses, train_accs, val_accs, val_aucs


# 🚀 Main Execution
if __name__ == '__main__':
    torch.manual_seed(42);
    np.random.seed(42)
    data_dir = r'C:\Users\skgtas8\OneDrive - University College London\AI & ML\Kaggle_Mood\Alzheimers'

    loss_functions = {
        'NLINEX (k=2.0, c=0.4)': nlinex_loss_2_0_0_4,
        'NLINEX (k=2.5, c=0.5)': nlinex_loss_2_5_0_5,
        'NLINEX (k=1.5, c=0.6)': nlinex_loss_1_5_0_6,
        'Focal': focal_loss,
        'Lovasz': lovasz_hinge_loss,
        'BCE': nn.BCELoss(),
        'MSE': nn.MSELoss(),
        'MAE': nn.L1Loss()
    }

    all_results = {'all': {}}
    print("\n📥 Loading all available composite binary samples...")
    X_train, X_val, y_train, y_val = load_custom_binary_data(data_dir)
    for name, criterion in loss_functions.items():
        print(f"\n🔧 Training with {name} Loss...")
        model = NeuroNet()
        all_results['all'][name] = train_model(model, criterion, X_train, y_train, X_val, y_val)

    # 📊 Plotting
    fig, axs = plt.subplots(3, 1, figsize=(12, 16))
    metrics = ['Val Loss', 'Val Accuracy', 'Val ROC-AUC']
    indices = [1, 3, 4]

    for row, idx in enumerate(indices):
        for name in loss_functions:
            axs[row].plot(all_results['all'][name][idx], label=name)
        axs[row].set_title(f'{metrics[row]} – All Samples')
        axs[row].set_xlabel('Epoch')
        axs[row].set_ylabel(metrics[row])
        if metrics[row] == 'Val ROC-AUC': axs[row].set_ylim(0.0, 1.0)
        axs[row].legend()

    plt.tight_layout()
    plt.show()