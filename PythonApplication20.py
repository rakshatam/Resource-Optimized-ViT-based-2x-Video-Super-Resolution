import os
import glob
import random
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from patchify import patchify
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

# --- Configuration ---
TRAIN_HR_DIR = "C:/Users/Lenovo/Downloads/archive (9)/Dataset/DIV2K_train_HR"
TRAIN_LR_DIR = "C:/Users/Lenovo/Downloads/archive (9)/Dataset/DIV2K_train_LR_bicubic/X2"
VALID_HR_DIR = "C:/Users/Lenovo/Downloads/archive (9)/Dataset/DIV2K_valid_HR"
VALID_LR_DIR = "C:/Users/Lenovo/Downloads/archive (9)/Dataset/DIV2K_valid_LR_bicubic/X2"

UPSCALE_FACTOR = 2
LR_PATCH_SIZE = 64   # patch size
HR_PATCH_SIZE = LR_PATCH_SIZE * UPSCALE_FACTOR
PATCHES_PER_IMAGE = 4  # number of patches to sample per image

# --- ViT Model Dimensions ---
MODEL_EMBED_DIM = 128
MODEL_NUM_LAYERS = 4
MODEL_NUM_HEADS = 4

# --- Training Params ---
BATCH_SIZE_IMAGES = 4           # number of images per batch
# effective patches per batch = BATCH_SIZE_IMAGES * PATCHES_PER_IMAGE
GRADIENT_ACCUM_STEPS = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True

# --- Output Paths ---
OUTPUT_DIR = "C:/Users/Lenovo/Downloads/TRAINING"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Checkpointed Transformer ---
class CheckpointedTransformer(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])
    def forward(self, x):
        for lyr in self.layers:
            x = checkpoint(lyr, x)
        return x

# --- Model Definition ---
class ViTSuperResolutionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, MODEL_EMBED_DIM, kernel_size=3, padding=1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=MODEL_EMBED_DIM, nhead=MODEL_NUM_HEADS,
            dim_feedforward=MODEL_EMBED_DIM*4, batch_first=True, dropout=0.1
        )
        self.transformer = CheckpointedTransformer(enc_layer, MODEL_NUM_LAYERS)
        self.upsample = nn.Sequential(
            nn.Conv2d(MODEL_EMBED_DIM, 3*UPSCALE_FACTOR**2, kernel_size=3, padding=1),
            nn.PixelShuffle(UPSCALE_FACTOR),
            nn.Conv2d(3, 3, kernel_size=3, padding=1)
        )
    def forward(self, x):
        b, c, h, w = x.shape
        feat = self.patch_embed(x)
        seq = feat.flatten(2).transpose(1,2)
        out = self.transformer(seq)
        img = out.transpose(1,2).view(b, MODEL_EMBED_DIM, h, w)
        return torch.clamp(self.upsample(img), 0, 1)

# --- Dataset: multiple patches per image ---
class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, lr_size, hr_size, upscale, patches_per_image, is_train=True):
        self.hr_files = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
        self.lr_files = sorted(glob.glob(os.path.join(lr_dir, "*.png")))
        assert len(self.hr_files)==len(self.lr_files), "HR/LR mismatch"
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.up = upscale
        self.k = patches_per_image
        self.is_train = is_train
    def __len__(self): return len(self.hr_files)
    def __getitem__(self, idx):
        hr = Image.open(self.hr_files[idx]).convert('RGB')
        lr = Image.open(self.lr_files[idx]).convert('RGB')
        exp = (hr.width//self.up, hr.height//self.up)
        if lr.size!=exp: lr=lr.resize(exp,Image.BICUBIC)
        hr_np = np.array(hr, np.float32)/255.
        lr_np = np.array(lr, np.float32)/255.
        # full-grid patchify
        lr_p = patchify(lr_np, (self.lr_size,self.lr_size,3), step=self.lr_size)
        hr_p = patchify(hr_np, (self.hr_size,self.hr_size,3), step=self.hr_size)
        lr_p = lr_p.reshape(-1,self.lr_size,self.lr_size,3)
        hr_p = hr_p.reshape(-1,self.hr_size,self.hr_size,3)
        # sample k indices
        if self.is_train:
            inds = random.choices(range(lr_p.shape[0]), k=self.k)
        else:
            center = lr_p.shape[0]//2
            inds = [center]*self.k
        # build stacks
        lr_t = torch.stack([TF.to_tensor(lr_p[i]) for i in inds], dim=0)
        hr_t = torch.stack([TF.to_tensor(hr_p[i]) for i in inds], dim=0)
        # augment each
        if self.is_train:
            for j in range(self.k):
                if random.random()>0.5:
                    lr_t[j],hr_t[j]=TF.hflip(lr_t[j]),TF.hflip(hr_t[j])
                if random.random()>0.5:
                    lr_t[j],hr_t[j]=TF.vflip(lr_t[j]),TF.vflip(hr_t[j])
        return lr_t, hr_t

# --- Training & Validation ---
def train_one_epoch(model, loader, opt, crit, scaler, dev, epoch, epochs, grad_steps):
    model.train(); total=0; opt.zero_grad()
    for lr_imgs,hr_imgs in tqdm(loader, desc=f"Train {epoch+1}/{epochs}"):
        # lr_imgs: [B_img, k, 3, H, W]
        B,k,_,H,W = lr_imgs.shape
        lr = lr_imgs.view(B*k,3,H,W).to(dev)
        hr = hr_imgs.view(B*k,3,H*UPSCALE_FACTOR,W*UPSCALE_FACTOR).to(dev) if False else hr_imgs.view(B*k,3,HR_PATCH_SIZE,HR_PATCH_SIZE).to(dev)
        with torch.cuda.amp.autocast(USE_AMP): out=model(lr); loss=crit(out,hr)/grad_steps
        scaler.scale(loss).backward()
        if (total+1)%grad_steps==0: scaler.step(opt); scaler.update(); opt.zero_grad()
        total+=1
        total_loss = loss.item()*grad_steps
    return total_loss/total

def validate_one_epoch(model, loader, crit, dev, epoch, epochs):
    model.eval(); sum_loss=0; sum_psnr=0;count=0
    with torch.no_grad():
        for lr_imgs,hr_imgs in tqdm(loader, desc=f"Val   {epoch+1}/{epochs}"):
            B,k,_,H,W = lr_imgs.shape
            lr = lr_imgs.view(B*k,3,H,W).to(dev)
            hr = hr_imgs.view(B*k,3,HR_PATCH_SIZE,HR_PATCH_SIZE).to(dev)
            with torch.cuda.amp.autocast(USE_AMP): out=model(lr); loss=crit(out,hr)
            sum_loss+=loss.item()
            mse=torch.mean((out-hr)**2,dim=[1,2,3])
            sum_psnr+=(10*torch.log10(1./torch.clamp(mse,1e-10))).sum().item()
            count+=B*k
    return sum_loss/len(loader), sum_psnr/count

# --- Main Execution ---
if __name__=="__main__":
    # diagnostics
    print("CUDA available:",torch.cuda.is_available())
    if DEVICE.type=='cuda': print(torch.cuda.get_device_name(0))
    # loaders
    train_ds=SRDataset(TRAIN_HR_DIR,TRAIN_LR_DIR,LR_PATCH_SIZE,HR_PATCH_SIZE,UPSCALE_FACTOR,PATCHES_PER_IMAGE,True)
    val_ds=SRDataset(VALID_HR_DIR,VALID_LR_DIR,LR_PATCH_SIZE,HR_PATCH_SIZE,UPSCALE_FACTOR,PATCHES_PER_IMAGE,False)
    train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE_IMAGES,shuffle=True,num_workers=4,pin_memory=True)
    val_loader=DataLoader(val_ds,batch_size=BATCH_SIZE_IMAGES,shuffle=False,num_workers=4,pin_memory=True)

    model=ViTSuperResolutionModel().to(DEVICE)
    crit=nn.L1Loss(); opt=optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=1e-3)
    sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=NUM_EPOCHS)
    scaler=torch.cuda.amp.GradScaler(enabled=USE_AMP)
    best_psnr=0
    for ep in range(NUM_EPOCHS):
        tr_loss=train_one_epoch(model,train_loader,opt,crit,scaler,DEVICE,ep,NUM_EPOCHS,GRADIENT_ACCUM_STEPS)
        val_loss,val_psnr=validate_one_epoch(model,val_loader,crit,DEVICE,ep,NUM_EPOCHS)
        sched.step()
        print(f"Epoch {ep+1}: train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}, val_psnr={val_psnr:.2f}")
        # save checkpoint
        ckpt_path=os.path.join(CHECKPOINT_DIR,f"ckpt_{ep+1}.pth")
        torch.save({'epoch':ep+1,'model':model.state_dict(),'opt':opt.state_dict(),'psnr':val_psnr}, ckpt_path)
        if val_psnr>best_psnr:
            best_psnr=val_psnr
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved with PSNR {best_psnr:.2f}")
    print("Training complete. Best PSNR:", best_psnr)
