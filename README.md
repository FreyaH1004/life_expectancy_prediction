This project is based on **PyTorch** and builds a three-layer neural network model to predict life expectancy based on various socio-economic and health indicators.

## Project Structure
- `src/config.py`: Hyperparameters and global configuration.
- `src/model.py`: Neural network architecture (128 -> 32 -> 1).
- `src/dataset.py`: Data preprocessing, standardization, and loading logic.
- `src/engine.py`: Training, validation, and evaluation metric calculation.
- `src/visualize.py`: Loss curves and prediction distribution visualization.
- `main.py`: Project entry point.

## Experimental Setup
- **Optimizer**: AdamW
- **Loss Function**: MSELoss
- **Learning Rate Scheduler**: ReduceLROnPlateau
- **Early Stopping Mechanism**: Patience = 600
- **Random Seed**: 42 (to ensure experimental reproducibility)

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the main program: `python main.py`

## Results
The model performs stably on the test set, with an $R^2$ index of approximately 0.9478.