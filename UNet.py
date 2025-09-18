import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image  
import os
import matplotlib.pyplot as plt

DEMO = True
CKPT = "unet_best.pt"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Hyper-parameters
learning_rate = 1e-3

class FlatPngSegDataset(Dataset):
    def __init__(self, img_root, mask_root, size=(256,256)):
        self.imgs  = sorted([os.path.join(img_root,  f) for f in os.listdir(img_root)  if f.endswith(".png")])
        self.masks = sorted([os.path.join(mask_root, f) for f in os.listdir(mask_root) if f.endswith(".png")])
        assert len(self.imgs) == len(self.masks)
        self.size = size

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        x = read_image(self.imgs[i]).float() / 255.0
        if x.size(0) >= 3:
            x = x[:3].mean(0, keepdim=True)
        x = F.interpolate(x.unsqueeze(0), self.size, mode="bilinear", align_corners=False).squeeze(0)

        lab = read_image(self.masks[i])[0:1].long()
        lab = F.interpolate(lab.float().unsqueeze(0), self.size, mode="nearest").squeeze(0).long()
        
        # remap values {0,85,170,255} -> {0,1,2,3}
        mapping = {0:0, 85:1, 170:2, 255:3}
        y = torch.zeros_like(lab)
        for k,v in mapping.items():
            y[lab==k] = v

        return x, y.squeeze(0)   # ensure shape [H,W]


    
root = "../keras_png_slices_data"

img_tr = f"{root}/keras_png_slices_train"
msk_tr = f"{root}/keras_png_slices_seg_train"
img_va = f"{root}/keras_png_slices_validate"
msk_va = f"{root}/keras_png_slices_seg_validate"

trainset = FlatPngSegDataset(img_tr, msk_tr, size=(256,256))
valset   = FlatPngSegDataset(img_va, msk_va, size=(256,256))
train_loader = DataLoader(trainset, batch_size=8, shuffle=True)
val_loader   = DataLoader(valset,   batch_size=8, shuffle=False)


n_classes = 4 #since tehre are 4 labels

for x, y in train_loader:
    print(torch.unique(y))
    break

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
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)                  # 32→64
        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Dropout2d(0.2)
        )

        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)                   # 64→128
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Dropout2d(0.2)
        )

        self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)                    # 128→256
        self.dec1 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
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
    
    def decode(self, x1, x2, x3, xb):
        y = self.up3(xb)                            # [B,128, 64, 64]
        y = torch.cat([y, x3], dim=1)               # [B,256, 64, 64]
        y = self.dec3(y)                            # [B,128, 64, 64]

        y = self.up2(y)                             # [B, 64,128,128]
        y = torch.cat([y, x2], dim=1)               # [B,128,128,128]
        y = self.dec2(y)                            # [B, 64,128,128]

        y = self.up1(y)                             # [B, 32,256,256]
        y = torch.cat([y, x1], dim=1)               # [B, 64,256,256]
        y = self.dec1(y)                            # [B, 32,256,256]

        return self.outc(y)                         # [B,C,256,256]

    # forward: head at 32×32, then upsample back to H×W
    def forward(self, x):
        x1, x2, x3, xb = self.encode(x)
        return self.decode(x1, x2, x3, xb)


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

def dice_per_class(logits: torch.Tensor, y: torch.Tensor, n_classes: int, eps: float = 1e-6):
    probs = torch.softmax(logits, dim=1)                       # [B,C,H,W]
    y1h   = torch.nn.functional.one_hot(y, n_classes).permute(0,3,1,2).float()  # [B,C,H,W]
    dims  = (0,2,3)                                            # sum over batch & pixels
    inter = (probs * y1h).sum(dims)                            # [C]
    union = probs.sum(dims) + y1h.sum(dims)                    # [C]
    return (2*inter + eps) / (union + eps)                     # [C]

@torch.inference_mode()
def evaluate(net: nn.Module, loader: DataLoader, n_classes: int):
    net.eval()

    ce_sum = 0.0
    px_correct = 0
    px_total   = 0

    inter_sum = torch.zeros(n_classes, device='cpu')   # keep accumulators on CPU
    union_sum = torch.zeros(n_classes, device='cpu')

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = net(x)                                # no graph is created
        ce_sum += ce_loss(logits, y).item() * x.size(0)

        # pixel accuracy
        pred = logits.argmax(1)
        px_correct += (pred == y).sum().item()
        px_total   += y.numel()

        # per-class Dice (do math on GPU, then move the small result to CPU)
        probs = logits.softmax(1)
        y1h   = torch.nn.functional.one_hot(y, n_classes).permute(0,3,1,2).to(probs.dtype)

        dims  = (0,2,3)
        inter = (probs * y1h).sum(dims).detach().cpu()
        union = (probs.sum(dims) + y1h.sum(dims)).detach().cpu()

        inter_sum += inter
        union_sum += union

    ce_avg  = ce_sum / len(loader.dataset)
    acc_px  = px_correct / px_total
    dice_pc = (2*inter_sum / union_sum.clamp_min(1e-6)).tolist()
    return ce_avg, acc_px, dice_pc

@torch.inference_mode()
def save_segmentation_figures(net, dataset, indices=(0,5,12), out_dir="figures", prefix="val"):
    os.makedirs(out_dir, exist_ok=True)
    net.eval()
    for idx in indices:
        x, y = dataset[idx]                             # x:[1,H,W], y:[H,W]
        pred = net(x.unsqueeze(0).to(device)).argmax(1).squeeze(0).cpu()

        # Triplet
        fig, axs = plt.subplots(1, 3, figsize=(12,4))
        axs[0].imshow(x.squeeze(0), cmap="gray");           axs[0].set_title("Input");        axs[0].axis("off")
        axs[1].imshow(y, cmap="nipy_spectral");             axs[1].set_title("Ground Truth"); axs[1].axis("off")
        axs[2].imshow(pred, cmap="nipy_spectral");          axs[2].set_title("Predicted");    axs[2].axis("off")
        fig.tight_layout()
        fig.savefig(f"{out_dir}/{prefix}_triplet_{idx}.png", dpi=150)
        plt.close(fig)

        # Overlay
        fig = plt.figure(figsize=(4,4))
        plt.imshow(x.squeeze(0), cmap="gray")
        plt.imshow(pred, cmap="nipy_spectral", alpha=0.45, interpolation="nearest")
        plt.title(f"{prefix} idx={idx}"); plt.axis("off")
        fig.savefig(f"{out_dir}/{prefix}_overlay_{idx}.png", dpi=150)
        plt.close(fig)


if not DEMO:
    best_mDSC = -1.0
    # Training loop
    net = UNetCNN(n_classes).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(20):
        net.train(); running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = net(x)
            loss = ce_loss(logits, y) + 0.5 * dice_loss(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)

        ce_v, acc_v, dice_pc = evaluate(net, val_loader, n_classes)
        mDSC = sum(dice_pc)/len(dice_pc)
        print(f"epoch {epoch+1:02d} | train_loss {running/len(trainset):.4f} "
            f"| val_CE {ce_v:.4f} | val_pxAcc {acc_v:.3f} | val_DSC per-class {dice_pc} | mDSC {mDSC:.3f}")

        if mDSC > best_mDSC:
            best_mDSC = mDSC
            torch.save(net.state_dict(), CKPT)
            print(f"[ckpt] saved new best -> {CKPT} (mDSC={best_mDSC:.3f})")

    save_segmentation_figures(net, valset, indices=(0,5,12,20), out_dir="figures", prefix="val")


img_te = f"{root}/keras_png_slices_test"
msk_te = f"{root}/keras_png_slices_seg_test"
testset = FlatPngSegDataset(img_te, msk_te, size=(256,256))
test_loader = DataLoader(testset, batch_size=8, shuffle=False)

if DEMO:
    net = UNetCNN(n_classes).to(device)
    net.load_state_dict(torch.load(CKPT, map_location=device))
    net.eval()
    print(f"loaded checkpoint from {CKPT}")

    ce_t, acc_t, dice_pc_t = evaluate(net, test_loader, n_classes)
    mDSC_t = sum(dice_pc_t)/len(dice_pc_t)
    print("test_pxAcc:", f"{acc_t:.3f}")
    print("test_DSC per-class:", [f"{d:.3f}" for d in dice_pc_t])



