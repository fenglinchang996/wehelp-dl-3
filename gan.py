import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as T

from utils import get_device


def plot_images(
    images: torch.Tensor,
    save_path: Path | None = None,
    show=False,
):
    images_with_grid = vutils.make_grid(
        images, nrow=10, normalize=True, value_range=(-1, 1)
    ).detach()

    plt.figure(figsize=(10, 10))
    plt.imshow(images_with_grid.cpu().permute(1, 2, 0))
    plt.axis("off")
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def generate_samples(generator: nn.Module, save_dir: Path, num_images=80):
    save_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    generator.to(device)
    generator.eval()

    with torch.inference_mode():
        noise = torch.randn(num_images, nz, 1, 1, device=device)
        fake_images = generator(noise)
        plot_images(fake_images, save_path=save_dir / "sample.png", show=False)


dataroot = Path("data/animefaces/")
batch_size = 128
workers = 4

image_size = 64
nc = 3

# size of generator input
nz = 100
# size of feature maps in generator
ngf = 64
# size of feature maps in discriminator
ndf = 64

epochs = 50

# optimizer
lr = 0.0002
betas = (0.5, 0.999)

generator_path = "gan_generator.pth"
discriminator_path = "gan_discriminator.pth"


# Weights init based on DCGAN
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # ngf x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
            # nc x 64 x 64
        )

    def forward(self, input):
        return self.net(input)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.net(input)


def get_transform():
    transform = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.uint8, scale=True),
            T.Resize(size=[64, 64], antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    return transform


def train():
    device = get_device()

    dataset = ImageFolder(dataroot, transform=get_transform())
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=workers)

    netG = Generator().to(device)
    netG.apply(weights_init)
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=betas)

    netD = Discriminator().to(device)
    netD.apply(weights_init)
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=betas)

    criterion = nn.BCELoss()

    real_label = 1.0
    fake_label = 0.0

    fixed_noise = torch.randn(40, nz, 1, 1, device=device)
    G_losses = []
    D_losses = []

    start = time.time()
    for epoch in range(epochs):
        netG.train()
        netD.train()
        for i, (batchX, _) in enumerate(dataloader, 0):
            # Discriminator
            ## Train with real batch
            netD.zero_grad()
            batchX = batchX.to(device)
            b_size = batchX.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(batchX).view(-1)
            lossD_real = criterion(output, label)
            lossD_real.backward()
            D_x = output.mean().item()
            ## Train with fake batch
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            lossD_fake = criterion(output, label)
            lossD_fake.backward()
            D_G_z1 = output.mean().item()
            lossD = lossD_real + lossD_fake
            optimizerD.step()

            # Generator
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            lossG = criterion(output, label)
            lossG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            G_losses.append(lossG.item())
            D_losses.append(lossD.item())

            if ((epoch + 1) % 5 == 0 or epoch == epochs - 1) and (
                i == len(dataloader) - 1
            ):
                print(
                    f"[{epoch + 1:2d}/{epochs}][{i + 1:3d}/{len(dataloader)}]",
                    " ",
                    f"Loss_D: {lossD.item():.4f} Loss_G: {lossG.item():.4f}",
                    f"D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}",
                )
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            netG.eval()
            with torch.inference_mode():
                plot_images(
                    netG(fixed_noise), save_path=Path(f"progress_epoch_{epoch + 1}.png")
                )
    end = time.time()
    print(f"duration: {end - start}")

    torch.save(netG.state_dict(), Path(generator_path))
    torch.save(netD.state_dict(), Path(discriminator_path))
    print("Models saved successfully")
    return netG


def main():
    is_training = False

    if is_training or not Path(generator_path).exists():
        train()

    if Path(generator_path).exists():
        generator = Generator()
        generator.load_state_dict(
            torch.load(
                Path(generator_path), map_location=get_device(), weights_only=True
            )
        )
        generate_samples(generator, save_dir=Path("gan_test"), num_images=3)


if __name__ == "__main__":
    main()
