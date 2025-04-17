from models import Generator
import torch
import math
import matplotlib.pyplot as plt

generator = Generator(checkpoint_path="checkpoint/generator.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator.to(device)

num_images = 24
n_rows = math.ceil(num_images / 4)
n_cols = num_images // n_rows
z = torch.randn(num_images, 100, 1, 1).to(device)
fake_images = generator(z)
fake_images = fake_images.detach().cpu().numpy()
fake_images = (fake_images + 1) / 2  # Rescale from [-1, 1] to [0, 1]

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(fake_images[i].squeeze(), cmap='gray', vmin=0.0, vmax=1.0)
    ax.axis('off')

plt.tight_layout()
plt.show()
