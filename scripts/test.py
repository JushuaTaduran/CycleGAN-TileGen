import os
import sys
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# Add the root directory to the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cycle_gan import CycleGAN
from dataset import ForestDataset
import config

# Load the model
device = config.DEVICE
model = CycleGAN().to(device)
checkpoint_path = os.path.join(config.CHECKPOINT_PATH, "forest_gan_latest.pth")  # Adjust the checkpoint path as needed
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Load the test datasets
testA_dataset = ForestDataset(config.DATASET_PATH, "testA")
testB_dataset = ForestDataset(config.DATASET_PATH, "testB")
testA_loader = DataLoader(testA_dataset, batch_size=1, shuffle=False)
testB_loader = DataLoader(testB_dataset, batch_size=1, shuffle=False)

# Create output directory if it doesn't exist
os.makedirs(config.OUTPUT_PATH, exist_ok=True)

# Generate and save images for testA (only 10 samples)
with torch.no_grad():
    for i, real_A in enumerate(testA_loader):
        if i >= 10:
            break
        real_A = real_A.to(device)
        fake_B = model.gen_AtoB(real_A)
        recov_A = model.gen_BtoA(fake_B)

        # Save the images
        save_image(fake_B, os.path.join(config.OUTPUT_PATH, f"fake_B_from_A_{i+1}.png"))
        save_image(recov_A, os.path.join(config.OUTPUT_PATH, f"recov_A_from_B_{i+1}.png"))

# Generate and save images for testB (only 10 samples)
with torch.no_grad():
    for i, real_B in enumerate(testB_loader):
        if i >= 5:
            break
        real_B = real_B.to(device)
        fake_A = model.gen_BtoA(real_B)
        recov_B = model.gen_AtoB(fake_A)

        # Save the images
        save_image(fake_A, os.path.join(config.OUTPUT_PATH, f"fake_A_from_B_{i+1}.png"))
        save_image(recov_B, os.path.join(config.OUTPUT_PATH, f"recov_B_from_A_{i+1}.png"))

print("Image generation complete!")