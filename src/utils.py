import numpy as np
from typing import Tuple, Optional

# evaluation metrics
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

# data split
def train_test_split_by_index(X: np.ndarray, y: np.ndarray, 
                               test_size: float = 0.2, 
                               random_state: Optional[int] = None) -> Tuple:
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split_idx = int(n_samples * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# save figure
def save_figure(fig, filename: str, figures_dir: str = None):
    from pathlib import Path
    from src.config import FIGURES_DIR
    save_dir = Path(figures_dir) if figures_dir else FIGURES_DIR
    save_path = save_dir / filename
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")

# print metrics
def print_metrics(metrics: dict, model_name: str = "Model"):
    print(f"\n{model_name} Performance:")
    
    for metric, value in metrics.items():
        print(f"  {metric:10s}: {value:.6f}")

