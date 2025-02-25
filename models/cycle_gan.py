import torch
import torch.nn as nn
import config  # Import the config module

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, img_channels=3, num_residuals=6):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.downsampling = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.residuals = nn.Sequential(*[ResidualBlock(256) for _ in range(num_residuals)])

        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, img_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.upsampling(self.residuals(self.downsampling(self.initial(x))))

class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class CycleGAN(nn.Module):
    def __init__(self):
        super(CycleGAN, self).__init__()
        self.gen_AtoB = Generator()
        self.gen_BtoA = Generator()
        self.disc_A = Discriminator()
        self.disc_B = Discriminator()
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

    def train_step(self, real_A, real_B, optimizer_G, optimizer_D):
        self.train()

        optimizer_G.zero_grad()

        loss_id_A = self.criterion_identity(self.gen_BtoA(real_A), real_A) * config.LAMBDA_IDENTITY
        loss_id_B = self.criterion_identity(self.gen_AtoB(real_B), real_B) * config.LAMBDA_IDENTITY
        loss_identity = loss_id_A + loss_id_B

        fake_B = self.gen_AtoB(real_A)
        pred_fake_B = self.disc_B(fake_B)
        loss_GAN_AtoB = self.criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))

        fake_A = self.gen_BtoA(real_B)
        pred_fake_A = self.disc_A(fake_A)
        loss_GAN_BtoA = self.criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))

        loss_GAN = loss_GAN_AtoB + loss_GAN_BtoA

        recov_A = self.gen_BtoA(fake_B)
        loss_cycle_A = self.criterion_cycle(recov_A, real_A) * config.LAMBDA_CYCLE

        recov_B = self.gen_AtoB(fake_A)
        loss_cycle_B = self.criterion_cycle(recov_B, real_B) * config.LAMBDA_CYCLE

        loss_cycle = loss_cycle_A + loss_cycle_B

        loss_G = loss_identity + loss_GAN + loss_cycle
        loss_G.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        pred_real_A = self.disc_A(real_A)
        loss_D_real_A = self.criterion_GAN(pred_real_A, torch.ones_like(pred_real_A))

        pred_real_B = self.disc_B(real_B)
        loss_D_real_B = self.criterion_GAN(pred_real_B, torch.ones_like(pred_real_B))

        pred_fake_A = self.disc_A(fake_A.detach())
        loss_D_fake_A = self.criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))

        pred_fake_B = self.disc_B(fake_B.detach())
        loss_D_fake_B = self.criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))

        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
        loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5

        loss_D = loss_D_A + loss_D_B
        loss_D.backward()
        optimizer_D.step()

        return loss_G.item(), loss_D.item()
