import os
import sys
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from pytorch_fid import fid_score

# Add the root directory to the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cycle_gan import CycleGAN
from dataset import ForestDataset
import config

device = config.DEVICE
print("Using", device)

trainA_dataset = ForestDataset(config.DATASET_PATH, "trainA")
trainB_dataset = ForestDataset(config.DATASET_PATH, "trainB")

trainA_loader = DataLoader(trainA_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
trainB_loader = DataLoader(trainB_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

model = CycleGAN().to(device)
optimizer_G = torch.optim.Adam(
    list(model.gen_AtoB.parameters()) + list(model.gen_BtoA.parameters()), 
    lr=config.LR_G, betas=(config.BETA1, config.BETA2)
)
optimizer_D = torch.optim.Adam(
    list(model.disc_A.parameters()) + list(model.disc_B.parameters()), 
    lr=config.LR_D, betas=(config.BETA1, config.BETA2)
)

# Load checkpoint if exists
start_epoch = 0
checkpoint_path = os.path.join(config.CHECKPOINT_PATH, "forest_gan_latest.pth")
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}")

# Lists to store losses
losses_G = []
losses_D = []

# Ensure analytics directories exist
analytics_loss_path = os.path.join(config.ROOT_DIR, "analytics", "losscurve")
analytics_fid_path = os.path.join(config.ROOT_DIR, "analytics", "FID")
os.makedirs(analytics_loss_path, exist_ok=True)
os.makedirs(analytics_fid_path, exist_ok=True)

for epoch in range(start_epoch, config.EPOCHS):
    for i, (real_A, real_B) in enumerate(zip(trainA_loader, trainB_loader)):
        real_A, real_B = real_A.to(device), real_B.to(device)

        loss_G, loss_D = model.train_step(real_A, real_B, optimizer_G, optimizer_D)

        losses_G.append(loss_G)
        losses_D.append(loss_D)

        if i % 100 == 0:
            print(f"Epoch [{epoch}/{config.EPOCHS}], Step [{i}/{len(trainA_loader)}], Loss G: {loss_G:.4f}, Loss D: {loss_D:.4f}")

    if (epoch + 1) % config.SAVE_INTERVAL == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
        }, checkpoint_path)

# Plot the losses
plt.figure(figsize=(10, 5))
plt.plot(losses_G, label='Generator Loss')
plt.plot(losses_D, label='Discriminator Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Losses')
plt.savefig(os.path.join(analytics_loss_path, 'training_losses.png'))
plt.show()

print("Training complete!")

# Calculate FID
def calculate_fid(real_dir, fake_dir):
    fid_value = fid_score.calculate_fid_given_paths([real_dir, fake_dir], batch_size=config.BATCH_SIZE, device=device, dims=2048)
    return fid_value

# Generate samples for FID calculation
def generate_samples(loader, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for i, real_img in enumerate(loader):
            real_img = real_img.to(device)
            if prefix == "AtoB":
                fake_img = model.gen_AtoB(real_img)
            else:
                fake_img = model.gen_BtoA(real_img)
            save_image(fake_img, os.path.join(output_dir, f"{prefix}_fake_{i+1}.png"))

# Generate samples for FID calculation
generate_samples(trainA_loader, os.path.join(analytics_fid_path, "AtoB"), "AtoB")
generate_samples(trainB_loader, os.path.join(analytics_fid_path, "BtoA"), "BtoA")

# Calculate FID for both directions
fid_AtoB = calculate_fid(os.path.join(config.DATASET_PATH, "trainB"), os.path.join(analytics_fid_path, "AtoB"))
fid_BtoA = calculate_fid(os.path.join(config.DATASET_PATH, "trainA"), os.path.join(analytics_fid_path, "BtoA"))

# Save FID scores
with open(os.path.join(analytics_fid_path, "fid_scores.txt"), "w") as f:
    f.write(f"FID AtoB: {fid_AtoB}\n")
    f.write(f"FID BtoA: {fid_BtoA}\n")

print(f"FID AtoB: {fid_AtoB}")
print(f"FID BtoA: {fid_BtoA}")
