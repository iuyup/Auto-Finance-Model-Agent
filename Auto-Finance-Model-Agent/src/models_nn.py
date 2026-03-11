import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.metrics import calc_metrics


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(256, 128), dropout=0.2):
        super().__init__()
        layers = []
        dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            dim = h
        layers += [nn.Linear(dim, 1)]  # 输出logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)  # (B,)


def train_mlp(
    X_sub_train, y_sub_train,
    X_val, y_val,
    X_test, y_test,
    epochs=30,
    batch_size=1024,
    lr=1e-3,
    weight_decay=1e-4,
    patience=5,
    device=None
):
    """
    返回：
    - y_prob_test: 测试集预测概率
    - metrics_test: 测试集指标dict
    - info: 训练信息（best_val_auc等）
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # numpy -> torch
    Xtr = torch.tensor(X_sub_train, dtype=torch.float32)
    ytr = torch.tensor(y_sub_train, dtype=torch.float32)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    yva = torch.tensor(y_val, dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=batch_size,
        shuffle=True
    )

    model = MLP(input_dim=X_sub_train.shape[1]).to(device)

    # 类别不平衡：pos_weight = neg/pos（跟xgb的scale_pos_weight类似）
    pos = float((y_sub_train == 1).sum())
    neg = float((y_sub_train == 0).sum())
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auc = -1.0
    best_state = None
    bad_epochs = 0

    def predict_prob(X):
        model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, X.shape[0], 4096):
                xb = X[i:i+4096].to(device)
                logits = model(xb)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.append(p)
        return np.concatenate(probs)

    # 训练循环 + 早停（用val AUC）
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 每个epoch评估一次val AUC
        y_prob_val = predict_prob(Xva)
        metrics_val = calc_metrics(y_val, y_prob_val, threshold=0.5)

        val_auc = metrics_val["auc"]
        print(f"[MLP] epoch={epoch:02d} val_auc={val_auc:.4f} val_f1={metrics_val['f1']:.4f}")

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[MLP] Early stop at epoch={epoch}, best_val_auc={best_val_auc:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # test
    y_prob_test = predict_prob(Xte)
    metrics_test = calc_metrics(y_test, y_prob_test, threshold=0.5)

    info = {
        "device": device,
        "best_val_auc": best_val_auc,
        "pos_weight": float((neg / pos) if pos > 0 else 1.0),
    }
    return y_prob_test, metrics_test, info