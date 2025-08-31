import os
import logging
import random
import numpy as np
import torch

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_I")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str) -> None:
    """Create directory if not exists."""
    os.makedirs(path, exist_ok=True)
