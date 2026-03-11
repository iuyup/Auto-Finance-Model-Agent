"""
扩展神经网络模型：LSTM + TabTransformer
TabTransformer 为自定义深度学习架构（加分项）
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.metrics import calc_metrics


# ============================================================
# LSTM：将 300 维特征视为长度 300 的序列
# ============================================================
class FeatureLSTM(nn.Module):
    """
    把每个样本的 D 维特征看作长度为 D 的序列，每步输入 1 维。
    用 LSTM 提取序列依赖后取最后隐藏状态做分类。
    """
    def __init__(self, input_dim: int, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        # x: (B, D) -> (B, D, 1)
        x = x.unsqueeze(-1)
        # lstm_out: (B, D, H)
        lstm_out, _ = self.lstm(x)
        # 取最后一步的隐藏状态
        last_hidden = lstm_out[:, -1, :]  # (B, H)
        return self.head(last_hidden).squeeze(1)  # (B,)


def train_lstm(
    X_sub_train, y_sub_train,
    X_val, y_val,
    X_test, y_test,
    hidden_size=64,
    num_layers=2,
    epochs=30,
    batch_size=512,
    lr=1e-3,
    weight_decay=1e-4,
    patience=5,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr = torch.tensor(X_sub_train, dtype=torch.float32)
    ytr = torch.tensor(y_sub_train, dtype=torch.float32)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

    model = FeatureLSTM(
        input_dim=X_sub_train.shape[1],
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device)

    pos = float((y_sub_train == 1).sum())
    neg = float((y_sub_train == 0).sum())
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auc = -1.0
    best_state = None
    bad_epochs = 0

    def predict_prob(X_tensor):
        model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, X_tensor.shape[0], 2048):
                xb = X_tensor[i:i+2048].to(device)
                logits = model(xb)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.append(p)
        return np.concatenate(probs)

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        y_prob_val = predict_prob(Xva)
        metrics_val = calc_metrics(y_val, y_prob_val, threshold=0.5)
        val_auc = metrics_val["auc"]
        print(f"[LSTM] epoch={epoch:02d} val_auc={val_auc:.4f} val_f1={metrics_val['f1']:.4f}")

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[LSTM] Early stop at epoch={epoch}, best_val_auc={best_val_auc:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_prob_test = predict_prob(Xte)
    metrics_test = calc_metrics(y_test, y_prob_test, threshold=0.5)

    info = {
        "device": device,
        "best_val_auc": best_val_auc,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "pos_weight": float(neg / pos) if pos > 0 else 1.0,
    }
    return y_prob_test, metrics_test, info


# ============================================================
# TabTransformer：自定义深度学习架构（加分项）
# 将连续特征分组 -> 线性嵌入 -> Transformer Encoder -> 分类头
# ============================================================
class TabTransformer(nn.Module):
    """
    Tabular Transformer:
    - 将 D 维连续特征分成 num_groups 组
    - 每组通过线性层映射到 d_model 维嵌入
    - 加上可学习的位置编码
    - 经过 Transformer Encoder
    - 对所有组的输出做 mean pooling -> 分类头
    """
    def __init__(
        self,
        input_dim: int,
        num_groups: int = 30,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.d_model = d_model

        # 每个组的特征数
        self.group_size = input_dim // num_groups
        self.remainder = input_dim - self.group_size * num_groups

        # 每组的嵌入层
        self.embeddings = nn.ModuleList([
            nn.Linear(self.group_size + (1 if i < self.remainder else 0), d_model)
            for i in range(num_groups)
        ])

        # 可学习位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, num_groups, d_model) * 0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        # x: (B, D) -> 分成 num_groups 组
        B = x.shape[0]
        chunks = []
        start = 0
        for i in range(self.num_groups):
            size = self.group_size + (1 if i < self.remainder else 0)
            chunk = x[:, start:start + size]  # (B, group_size)
            emb = self.embeddings[i](chunk)  # (B, d_model)
            chunks.append(emb)
            start += size

        # (B, num_groups, d_model)
        tokens = torch.stack(chunks, dim=1)
        tokens = tokens + self.pos_embedding

        # Transformer
        tokens = self.transformer(tokens)  # (B, num_groups, d_model)

        # Mean pooling
        pooled = tokens.mean(dim=1)  # (B, d_model)

        return self.head(pooled).squeeze(1)  # (B,)


def train_tab_transformer(
    X_sub_train, y_sub_train,
    X_val, y_val,
    X_test, y_test,
    num_groups=30,
    d_model=64,
    nhead=4,
    num_layers=2,
    epochs=30,
    batch_size=512,
    lr=1e-3,
    weight_decay=1e-4,
    patience=5,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr = torch.tensor(X_sub_train, dtype=torch.float32)
    ytr = torch.tensor(y_sub_train, dtype=torch.float32)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

    model = TabTransformer(
        input_dim=X_sub_train.shape[1],
        num_groups=num_groups,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
    ).to(device)

    pos = float((y_sub_train == 1).sum())
    neg = float((y_sub_train == 0).sum())
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = -1.0
    best_state = None
    bad_epochs = 0

    def predict_prob(X_tensor):
        model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, X_tensor.shape[0], 2048):
                xb = X_tensor[i:i+2048].to(device)
                logits = model(xb)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.append(p)
        return np.concatenate(probs)

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        y_prob_val = predict_prob(Xva)
        metrics_val = calc_metrics(y_val, y_prob_val, threshold=0.5)
        val_auc = metrics_val["auc"]
        print(f"[TabTransformer] epoch={epoch:02d} val_auc={val_auc:.4f} val_f1={metrics_val['f1']:.4f}")

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[TabTransformer] Early stop at epoch={epoch}, best_val_auc={best_val_auc:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_prob_test = predict_prob(Xte)
    metrics_test = calc_metrics(y_test, y_prob_test, threshold=0.5)

    info = {
        "device": device,
        "best_val_auc": best_val_auc,
        "num_groups": num_groups,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "pos_weight": float(neg / pos) if pos > 0 else 1.0,
    }
    return y_prob_test, metrics_test, info
