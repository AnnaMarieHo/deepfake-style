import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from utils.dataset_pure_style import PureStyleDataset
from utils.balanced_sampler import BalancedBatchSampler
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class PureStyleClassifier(nn.Module):
    def __init__(self, style_dim=25, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, style_features):
        return self.net(style_features)

dataset = PureStyleDataset("openfake-annotation/datasets/combined/cache/pure_style_embeddings.npz")

style_dim = dataset.style.shape[1]
print(f"Style dimension: {style_dim}")

model = PureStyleClassifier(style_dim=style_dim).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

labels = dataset.label.numpy()
print(f"Dataset: {len(dataset)} samples | Real: {(labels==1).sum()} | Fake: {(labels==0).sum()}")

real_idx = np.where(labels == 1)[0]
fake_idx = np.where(labels == 0)[0]

np.random.seed(42)
np.random.shuffle(real_idx)
np.random.shuffle(fake_idx)

train_real = real_idx[:int(0.8 * len(real_idx))]
val_real = real_idx[int(0.8 * len(real_idx)):]
train_fake = fake_idx[:int(0.8 * len(fake_idx))]
val_fake = fake_idx[int(0.8 * len(fake_idx)):]

train_idx = np.concatenate([train_real, train_fake])
val_idx = np.concatenate([val_real, val_fake])

train_ds = Subset(dataset, train_idx)
val_ds = Subset(dataset, val_idx)

train_labels = labels[train_idx]
print(f"Train: {len(train_ds)} samples | Real: {(train_labels==1).sum()} | Fake: {(train_labels==0).sum()}")
print(f"Val: {len(val_idx)} samples | Real: {(labels[val_idx]==1).sum()} | Fake: {(labels[val_idx]==0).sum()}")

batch_size = 128
train_sampler = BalancedBatchSampler(train_labels, batch_size)
train_dl = DataLoader(train_ds, batch_sampler=train_sampler)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for style, y, sim, cluster in dataloader:
        style, y = style.to(device), y.to(device)
        
        logits = model(style)
        loss = nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0

def validate_epoch(model, dataloader, device):
    model.eval()
    y_true, y_prob = [], []
    
    with torch.no_grad():
        for style, y, sim, cluster in dataloader:
            style, y = style.to(device), y.to(device)
            
            logits = model(style)
            probs = torch.sigmoid(logits)
            
            y_true.extend(y.cpu().numpy().flatten())
            y_prob.extend(probs.cpu().numpy().flatten())
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob > 0.5).astype(int)
    
    from sklearn.metrics import accuracy_score, roc_auc_score
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    return acc, auc

best_auc = 0.0
for epoch in range(30):
    loss_val = train_epoch(model, train_dl, opt, device)
    acc, auc = validate_epoch(model, val_dl, device)
    
    print(f"Epoch {epoch+1:2d} | Loss {loss_val:.4f} | Acc {acc:.3f} | AUC {auc:.3f}", end="")
    
    if auc > best_auc:
        best_auc = auc
        torch.save({
            "model": model.state_dict(),
            "style_dim": style_dim,
        }, "checkpoints/pure_style.pt")
        print(" Best!")
    else:
        print()

print(f"\nSaved to checkpoints/pure_style.pt")
print(f"Best AUC: {best_auc:.4f}")

