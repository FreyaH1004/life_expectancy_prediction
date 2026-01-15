import random
import torch

# 路径配置
DATA_PATH = "data/LifeExpectancyData_cleaned.csv"
MODEL_SAVE_PATH = "best_life_expectancy_model.pth"

# 数据配置
TARGET = 'Life_expectancy'
DROP_COLUMNS = ['Country', 'Life_expectancy'] #
TEST_SIZE = 0.2
VAL_SIZE = 0.2
RANDOM_STATE = 42

# 训练超参数
BATCH_SIZE = 64
LR = 0.005
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 5000
PATIENCE = 600
MIN_DELTA = 1e-4

# 随机种子配置
SEED = 42 # or random.randint(0,10000)

# 硬件设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")