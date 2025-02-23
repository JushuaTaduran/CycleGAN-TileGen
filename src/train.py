import torch
import itertools
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import save_image
from config import config
from data_load import get_dataloader
from generator_model import Generator
from discriminator_model import Discriminator
from utils import ReplayBuffer, LambdaLR, weights_init_normal
import matplotlib.pyplot as plt

def plot_losses(epoch, G_losses, D_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{config.output_path}/loss_plot_epoch_{epoch}.png")
    plt.close()

def train():
    # Initialize models
    netG_A2B = Generator(input_nc=3, output_nc=3).to(config.device)
    netG_B2A = Generator(input_nc=3, output_nc=3).to(config.device)
    netD_A = Discriminator(input_nc=3).to(config.device)
    netD_B = Discriminator(input_nc=3).to(config.device)

    # Initialize weights
    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # Losses
    criterion_GAN = torch.nn.MSELoss().to(config.device)
    criterion_cycle = torch.nn.L1Loss().to(config.device)
    criterion_identity = torch.nn.L1Loss().to(config.device)

    # Optimizers
    optimizer_G = Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=config.learning_rate, betas=(config.beta1, config.beta2))
    optimizer_D_A = Adam(netD_A.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2))
    optimizer_D_B = Adam(netD_B.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2))

    # Learning rate schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(config.num_epochs, 0, config.num_epochs // 2).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(config.num_epochs, 0, config.num_epochs // 2).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(config.num_epochs, 0, config.num_epochs // 2).step)

    # Data loaders
    dataloader_A = get_dataloader(config.default_tile_path, config.batch_size, config.image_size)
    dataloader_B = get_dataloader(config.forest_tiles_path, config.batch_size, config.image_size)

    # Buffers
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    G_losses = []
    D_losses = []

    real_label = 0.9  # One-sided label smoothing
    fake_label = 0.0  # Fake remains 0

    for epoch in range(config.num_epochs):
        for i, (real_A, real_B) in enumerate(zip(dataloader_A, dataloader_B)):
            real_A = real_A.to(config.device)
            real_B = real_B.to(config.device)

            # Generators A2B and B2A
            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(netG_B2A(real_A), real_A) * config.lambda_identity
            loss_id_B = criterion_identity(netG_A2B(real_B), real_B) * config.lambda_identity

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, torch.full_like(pred_fake, real_label, device=config.device))

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, torch.full_like(pred_fake, real_label, device=config.device))

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * config.lambda_cycle

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * config.lambda_cycle

            # Total loss
            loss_G = loss_id_A + loss_id_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()
            optimizer_G.step()

            # Discriminator A
            optimizer_D_A.zero_grad()

            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, torch.full_like(pred_real, real_label, device=config.device))

            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.full_like(pred_fake, fake_label, device=config.device))

            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            # Discriminator B
            optimizer_D_B.zero_grad()

            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, torch.full_like(pred_real, real_label, device=config.device))

            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.full_like(pred_fake, fake_label, device=config.device))

            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()

            G_losses.append(loss_G.item())
            D_losses.append(loss_D_A.item() + loss_D_B.item())

            print(f"[Epoch {epoch}/{config.num_epochs}] [Batch {i}/{len(dataloader_A)}] [D loss: {loss_D_A.item() + loss_D_B.item()}] [G loss: {loss_G.item()}]")

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Save sample images
        if epoch % config.save_sample_interval == 0:
            save_image(fake_A, f"{config.output_path}/fake_A_{epoch}.png", normalize=True)
            save_image(fake_B, f"{config.output_path}/fake_B_{epoch}.png", normalize=True)
            plot_losses(epoch, G_losses, D_losses)

if __name__ == "__main__":
    train()