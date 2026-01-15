import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss_curve(train_losses, val_losses):
    """绘制训练与验证集的损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.title('Training and Validation Loss Over Epochs')
    plt.ylim(0, 100) # 参考原代码的坐标范围
    plt.grid(True)
    plt.show()

def plot_predictions(y_true, y_pred, r2):
    """绘制真实值 vs 预测值的散点图"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color='blue')
    # 绘制 45 度参考线
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Life Expectancy')
    plt.ylabel('Predicted Life Expectancy')
    plt.title(f'Actual vs Predicted (R² = {r2:.4f})')
    plt.grid(True)
    plt.show()