import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from . import config


def get_data_loaders():
    df = pd.read_csv(config.DATA_PATH)

    X = df.drop(config.DROP_COLUMNS, axis=1)  #
    y = df[config.TARGET]

    # 两次拆分得到 Train, Val, Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=config.VAL_SIZE, random_state=config.RANDOM_STATE
    )

    # 标准化
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # 转换为 Tensor
    def to_tensor(X, y):
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32)
        return X_t, y_t

    train_ds = TensorDataset(*to_tensor(X_train_s, y_train))
    val_ds = TensorDataset(*to_tensor(X_val_s, y_val))
    test_ds = TensorDataset(*to_tensor(X_test_s, y_test))

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)

    return train_loader, val_ds, test_ds, X_train_s.shape[1]