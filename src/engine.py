import torch
import numpy as np
import random
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def set_seed(seed):
    """固定所有随机种子以保证可重复性"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 使用多GPU时

    # 固定 CuDNN 的算法底层实现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def evaluate(model, dataset, criterion, device):
    model.eval()
    X, y = dataset.tensors
    X, y = X.to(device), y.to(device)
    with torch.no_grad():
        outputs = model(X)
        loss = criterion(outputs, y)
    return loss.item(), outputs.cpu().numpy(), y.cpu().numpy()

def get_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) #
    mae = mean_absolute_error(y_true, y_pred) #
    r2 = r2_score(y_true, y_pred) #
    return rmse, mae, r2