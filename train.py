import os
import torch
import torch.optim as optim
from models import Generator, Critic
from utils import compute_gradient_penalty, critic_accuracy
from dataset import dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Hyperparameters
batch_size = 64
latent_dim = 100
lr = 1e-4
n_critic = 5
lambda_gp = 10
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
generator = Generator().to(device)
critic = Critic().to(device)
opt_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
opt_C = optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

writer = SummaryWriter()
j = 0
if not os.path.exists("checkpoint"):
    os.makedirs("checkpoint")

for epoch in range(num_epochs):
    with tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as data_loader_tqdm:
        for i, (real_imgs, _) in enumerate(data_loader_tqdm):
            real_imgs = real_imgs.to(device)

            # Train Critic
            for _ in range(n_critic):
                z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
                fake_imgs = generator(z).detach()
                critic_real = critic(real_imgs)
                critic_fake = critic(fake_imgs)
                gp = compute_gradient_penalty(critic, real_imgs, fake_imgs, device)
                loss_C = -torch.mean(critic_real) + torch.mean(critic_fake) + lambda_gp * gp
                opt_C.zero_grad()
                loss_C.backward()
                opt_C.step()

            # Train Generator
            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            gen_imgs = generator(z)
            loss_G = -critic(gen_imgs).mean()
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # Log progress
            data_loader_tqdm.set_postfix(loss_C=loss_C.item(), loss_G=loss_G.item(), C_acc=critic_accuracy)
            writer.add_scalar("Loss/Critic", loss_C.item(), j)
            writer.add_scalar("Loss/Generator", loss_G.item(), j)
            writer.add_scalar("Critic Accuracy", critic_accuracy(critic_real, critic_fake), j)
            j += 1

    # epoch end
    # log generated images with writer and save model
    with torch.no_grad():
        z = torch.randn(9, latent_dim, 1, 1, device=device)
        fake_imgs = generator(fixed_noise).detach().cpu().numpy()
    fake_imgs = (fake_imgs + 1) / 2
    writer.add_images("Generated Images", fake_imgs, epoch)

    # Save model
    torch.save(generator.state_dict(), f"checkpoint/generator.pth")
    torch.save(critic.state_dict(), f"checkpoint/critic.pth")

writer.close()
