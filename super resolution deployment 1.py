import cv2
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.checkpoint import checkpoint
import numpy as np
import time

# --- 1) ViTVideoSR definition matching your trained model ---
MODEL_EMBED_DIM  = 128
MODEL_NUM_HEADS  = 4
MODEL_NUM_LAYERS = 4    # ← must match your checkpoint
UPSCALE_FACTOR   = 2

class CheckpointedTransformer(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])
    def forward(self, x):
        for lyr in self.layers:
            x = checkpoint(lyr, x)
        return x

class ViTVideoSR(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_ch, MODEL_EMBED_DIM, kernel_size=3, padding=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=MODEL_EMBED_DIM,
            nhead=MODEL_NUM_HEADS,
            dim_feedforward=MODEL_EMBED_DIM * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = CheckpointedTransformer(encoder_layer, MODEL_NUM_LAYERS)
        self.upsample = nn.Sequential(
            nn.Conv2d(MODEL_EMBED_DIM, in_ch * (UPSCALE_FACTOR**2), kernel_size=3, padding=1),
            nn.PixelShuffle(UPSCALE_FACTOR),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1)
        )

    def forward(self, lr):
        b, c, h, w = lr.shape
        x = self.patch_embed(lr)                   # [B, E, H, W]
        x = x.flatten(2).transpose(1, 2)           # [B, H*W, E]
        x = self.transformer(x)                    # [B, H*W, E]
        x = x.transpose(1, 2).view(b, MODEL_EMBED_DIM, h, w)
        return torch.clamp(self.upsample(x), 0.0, 1.0)  # [B, C, 2H, 2W]

# --- 2) Video SR pipeline with progress & AMP ---
def process_video(input_path, output_path, model_path, device):
    # Load model
    model = ViTVideoSR().to(device)
    state = torch.load(model_path, map_location=device)
    if 'model' in state:
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)
    model.eval()

    # OpenCV I/O
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file {input_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    w_in         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_out, h_out = w_in * UPSCALE_FACTOR, h_in * UPSCALE_FACTOR

    # MP4 writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w_out, h_out))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    print(f"Processing {total_frames} frames...")

    start = time.time()
    frame_idx = 0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # to torch tensor
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = TF.to_tensor(img).unsqueeze(0).to(device)

            # SR with AMP
            with torch.cuda.amp.autocast():
                sr = model(t)

            # back to BGR uint8
            sr_img = (sr.squeeze().clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
            sr_img = cv2.cvtColor(sr_img.transpose(1,2,0), cv2.COLOR_RGB2BGR)
            writer.write(sr_img)

            # free GPU memory
            torch.cuda.empty_cache()

            # print progress
            print(f"\rFrame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}% )", end='', flush=True)

    elapsed = time.time() - start
    cap.release()
    writer.release()
    print(f"\n✅  Done in {elapsed:.1f}s. Saved upscaled video to {output_path}")

# --- 3) Interactive entrypoint ---
if __name__ == "__main__":
    input_path  = input("Path to input video: ").strip().strip('"')
    output_path = input("Path to save upscaled video (filename.mp4): ").strip().strip('"')
    model_path  = input("Path to your .pth model file: ").strip().strip('"')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    process_video(input_path, output_path, model_path, device)
