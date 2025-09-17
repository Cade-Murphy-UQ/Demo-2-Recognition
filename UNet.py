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
        x1, x2, x3, xb = self.encode(x)
        return self.decode(xb)


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


# Training loop
net = UNetCNN(n_classes).to(device)
opt = torch.optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(10):
    net.train(); running=0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = net(x)
        loss = ce_loss(logits, y) + 0.5 * dice_loss(logits, y)
        loss.backward(); opt.step()
        running += loss.item() * x.size(0)
    print(f"epoch {epoch+1}: train_loss={running/len(trainset):.4f}")