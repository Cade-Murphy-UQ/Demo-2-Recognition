import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image  
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Hyper-parameters
num_epochs = 8
learning_rate = 1e-3

class FlatPngDataset(Dataset):
    def __init__(self, root, size=(28, 28)):
        self.paths = [os.path.join(root, f)
                      for f in os.listdir(root)
                      if f.lower().endswith(".png")]
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # read_image -> uint8 [C,H,W], C can be 1 (grayscale) or 3/4 (RGB/RGBA)
        x = read_image(self.paths[idx]).float() / 255.0  # -> [C,H,W] in [0,1]
        if x.size(0) == 3:           # RGB -> gray
            x = x.mean(0, keepdim=True)
        elif x.size(0) == 4:         # RGBA -> RGB -> gray
            x = x[:3].mean(0, keepdim=True)
        # ensure 28x28
        x = F.interpolate(x.unsqueeze(0), size=self.size, mode="bilinear",
                          align_corners=False).squeeze(0)
        return x, 0                   # dummy label
    

train_root = "../keras_png_slices_data/keras_png_slices_train"
val_root   = "../keras_png_slices_data/keras_png_slices_validate"

trainset = FlatPngDataset(train_root, size=(28, 28))
testset  = FlatPngDataset(val_root,   size=(28, 28))

train_loader = DataLoader(trainset, batch_size=128, shuffle=True)
test_loader  = DataLoader(testset,  batch_size=128, shuffle=False)

class CNNVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder (same as before but outputs mean and log_var)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 28x28 -> 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 14x14 -> 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 7x7 -> 7x7
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # 7x7 -> 4x4

            nn.Flatten(),
        )

        # Latent space parameters
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)      # Mean
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)  # Log variance

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),  # Reshape to (batch, 128, 4, 4)

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4x4 -> 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),  # 16x16 -> 16x16
            nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False),  # 16x16 -> 28x28
            nn.Sigmoid()  # Output in [0, 1] for image reconstruction
        )

    def encode(self, x):
        """Encode input to latent parameters"""
        h = self.encoder_conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: sample from N(mu, var) using N(0,1)"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # Use mean for inference. Using mu (mean) rather than sampling from the full distribution because:
                        # It provides deterministic, reproducible results
                        # The mean represents the "most likely" latent representation
                        # It avoids noise that could make interpolation less smooth

    def decode(self, z):
        """Decode latent variable to reconstruction"""
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction loss + KL divergence
    beta: weight for KL divergence (beta-VAE)
    """
    # Reconstruction loss (Binary Cross Entropy)
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence loss
    # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + beta * KLD, BCE, KLD


import matplotlib.pyplot as plt
# Train the VAE
vae = CNNVAE(latent_dim=32).to(device)
vae_optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

# Training loop
num_vae_epochs = 10
beta = 1.0  # Beta parameter for beta-VAE (1.0 = standard VAE)

print("Training VAE...")
for epoch in range(num_vae_epochs):
    vae.train()
    total_loss = 0
    total_bce = 0
    total_kld = 0

    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)

        # Forward pass
        recon_images, mu, logvar = vae(images)

        # Calculate loss
        loss, bce, kld = vae_loss_function(recon_images, images, mu, logvar, beta)

        # Backward pass
        vae_optimizer.zero_grad()
        loss.backward()
        vae_optimizer.step()

        total_loss += loss.item()
        total_bce += bce.item()
        total_kld += kld.item()

    avg_loss = total_loss / len(train_loader.dataset)
    avg_bce = total_bce / len(train_loader.dataset)
    avg_kld = total_kld / len(train_loader.dataset)

    print(f'VAE Epoch [{epoch+1}/{num_vae_epochs}], Loss: {avg_loss:.4f}, BCE: {avg_bce:.4f}, KLD: {avg_kld:.4f}')


# Manifold visualisation


def plot_decoder_manifold_grid(model, latent_dim, device,
                               n=15, lim=3.0,
                               savepath="outputs/vae_manifold_grid.png",
                               title="Decoder samples over 2D latent grid"):
    """
    Create an n x n grid by sweeping the first two latent dims in [-lim, lim],
    fixing all other latent dims to 0, and decoding each point.
    """
    model.eval()
    os.makedirs("outputs", exist_ok=True)

    grid_x = torch.linspace(-lim, lim, n)
    grid_y = torch.linspace(-lim, lim, n)

    fig, axes = plt.subplots(n, n, figsize=(n, n))
    with torch.no_grad():
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z = torch.zeros(1, latent_dim, device=device)   # [1, D]
                z[0, 0] = xi
                z[0, 1] = yi
                xhat = model.decode(z).cpu().squeeze()          # [H,W] since 1-channel
                axes[i, j].imshow(xhat.numpy(), cmap="gray", vmin=0, vmax=1)
                axes[i, j].axis("off")

    plt.suptitle(title, y=1.02, fontsize=12)
    plt.tight_layout(pad=0.05)
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    print("Saved:", savepath)
    plt.show()


plot_decoder_manifold_grid(vae, vae.latent_dim, device,
                           n=15, lim=3.0,
                           savepath="outputs/vae_manifold_grid.png",
                           title="VAE manifold (dims 0 & 1; others=0)")