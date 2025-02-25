import os
import torch

# Root and dataset paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(ROOT_DIR, "datasets")
CHECKPOINT_PATH = os.path.join(ROOT_DIR, "checkpoints")
OUTPUT_PATH = os.path.join(ROOT_DIR, "outputs")

# Ensure required directories exist
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Training Parameters
BATCH_SIZE = 4  # Increase if GPU memory allows
EPOCHS = 200
LR_G = 0.0001  # Lower learning rate for generator
LR_D = 0.0001  # Lower learning rate for discriminator
BETA1 = 0.5
BETA2 = 0.999
LAMBDA_CYCLE = 10.0  # Increase cycle consistency loss weight
LAMBDA_IDENTITY = 10.0  # Increase identity loss weight
SAVE_INTERVAL = 10  # Save checkpoint every 10 epochs
IMG_SIZE = 32  # Tile size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
