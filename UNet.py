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

class FlatPngSegDataset(Dataset):
    def __init__(self, img_root, mask_root, size=(256,256)):
        self.imgs  = sorted([os.path.join(img_root,  f) for f in os.listdir(img_root)  if f.endswith(".png")])
        self.masks = sorted([os.path.join(mask_root, f) for f in os.listdir(mask_root) if f.endswith(".png")])
        assert len(self.imgs) == len(self.masks), "mismatch images/masks"
        self.size = size

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        x = read_image(self.imgs[i]).float()/255.0          # [C,H,W], 0..1
        if x.size(0) >= 3: x = x[:3].mean(0, keepdim=True)  # grayscale
        x = F.interpolate(x.unsqueeze(0), self.size, mode="bilinear", align_corners=False).squeeze(0)

        y = read_image(self.masks[i])[0].long()             # take single channel as labels
        y = F.interpolate(y[None,None].float(), self.size, mode="nearest").squeeze(0).squeeze(0).long()
        return x, y
    

img_tr = "../keras_png_slices_data/keras_png_slices_train"
msk_tr = "../keras_png_slices_data/keras_png_slices_seg_train"
img_va = "../keras_png_slices_data/keras_png_slices_validate"
msk_va = "../keras_png_slices_data/keras_png_slices_seg_validate"

trainset = FlatPngSegDataset(img_tr, msk_tr, size=(256,256))
valset   = FlatPngSegDataset(img_va, msk_va, size=(256,256))
train_loader = DataLoader(trainset, batch_size=8, shuffle=True)
val_loader   = DataLoader(valset,   batch_size=8, shuffle=False)

xb, yb = next(iter(train_loader))
n_classes = int(yb.max().item() + 1)
print("n_classes:", n_classes)

class UNetCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Dropout2d(0.2)
        )                       # 256 -> 256
        self.pool1 = nn.MaxPool2d(2)   # 256 -> 128

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Dropout2d(0.2)
        )                       # 128 -> 128
        self.pool2 = nn.MaxPool2d(2)   # 128 -> 64

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Dropout2d(0.2)
        )                       # 64 -> 64
        self.pool3 = nn.MaxPool2d(2)   # 64 -> 32

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Dropout2d(0.2)
        ) 

        # Decoder
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)  # 32 -> 64
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Dropout2d(0.2)
        )

        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)   # 64 -> 128
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Dropout2d(0.2)
        )

        self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)    # 128 -> 256
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Dropout2d(0.2)
        )

        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)    # final logits
        
    #  encode returns feature maps
    def encode(self, x):
        x1 = self.enc1(x)                 # [B, 32, 256, 256]
        x2 = self.enc2(self.pool1(x1))    # [B, 64, 128, 128]
        x3 = self.enc3(self.pool2(x2))    # [B,128,  64,  64]
        xb = self.bottleneck(self.pool3(x3))  # [B,256, 32, 32]
        return x1, x2, x3, xb
    
    def decode(self, xb):
        y = self.up3(xb); y = self.dec3(y)      # -> [B,128, 64, 64]
        y = self.up2(y);  y = self.dec2(y)      # -> [B, 64,128,128]
        y = self.up1(y);  y = self.dec1(y)      # -> [B, 32,256,256]
        return self.outc(y)                     # -> [B,C,256,256]

    # forward: head at 32×32, then upsample back to H×W
    def forward(self, x):
        return 


ce_loss = nn.CrossEntropyLoss()

def dice_loss(logits, target, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    tgt1h = torch.nn.functional.one_hot(target, probs.shape[1])
    tgt1h = tgt1h.permute(0,3,1,2).float()
    dims = (0,2,3)
    inter = (probs * tgt1h).sum(dims)
    union = probs.sum(dims) + tgt1h.sum(dims)
    dice  = (2*inter + eps) / (union + eps)
    return 1.0 - dice.mean()


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