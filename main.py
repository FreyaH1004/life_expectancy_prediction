import torch
import torch.nn as nn
import torch.optim as optim
import copy
from src import config, model, dataset, engine, visualize


def run():
    engine.set_seed(config.SEED)

    # 1. 准备数据
    train_loader, val_ds, test_ds, input_dim = dataset.get_data_loaders()

    # 2. 初始化模型与优化器
    net = model.LifeExpectancyNN(input_dim).to(config.DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=150)

    # 用于绘图的数据记录
    train_losses_history = []
    val_losses_history = []

    # 3. 训练循环
    best_val_loss = float('inf')
    best_wts = None
    early_stop_count = 0

    print(f"Start Training on {config.DEVICE}...")
    for epoch in range(config.NUM_EPOCHS):
        # 训练并记录
        train_loss = engine.train_one_epoch(net, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, _, _ = engine.evaluate(net, val_ds, criterion, config.DEVICE)

        train_losses_history.append(train_loss)
        val_losses_history.append(val_loss)

        scheduler.step(val_loss)

        # 早停与权重保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_wts = copy.deepcopy(net.state_dict())
            early_stop_count = 0
        else:
            early_stop_count += 1

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if early_stop_count >= config.PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}!")
            break

    # 4. 训练过程可视化
    visualize.plot_loss_curve(train_losses_history, val_losses_history)

    # 5. 最终评估与预测可视化
    net.load_state_dict(best_wts)
    _, y_pred, y_true = engine.evaluate(net, test_ds, criterion, config.DEVICE)

    rmse, mae, r2 = engine.get_metrics(y_true, y_pred)
    print(f"\n========= Model Evaluation Report =========")
    print(f"RMSE: {rmse:.4f} years")
    print(f"MAE:  {mae:.4f} years")
    print(f"R²:   {r2:.4f}")
    print(f"===========================================")

    visualize.plot_predictions(y_true, y_pred, r2)


if __name__ == "__main__":
    run()