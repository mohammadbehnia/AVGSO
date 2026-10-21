"""
transnav_gso_maps.py

Multi-task transformer for:
  - next-frame prediction
  - next-state prediction (lat, lon, HEIGHT, zoom, PITCH, heading)
using map screenshots from:

  D:\\PhD\\Code\\gso\\maps_screenshots

Filenames must look like:
  frame_<id>_<lat>_<lon>_<HEIGHT>_<zoom>_<PITCH>_<heading>.png

Example:
  frame_0001_36.0089_-79.6937_90_18_60_120.png
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List

import os
import csv
import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    img_size: Tuple[int, int] = (224, 224)
    seq_len: int = 8                    # number of past frames
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    state_dim: int = 6                  # [lat, lon, HEIGHT, zoom, PITCH, heading]
    lambda_state: float = 1.0           # weight for state loss
    max_seq_len: int = 64               # max temporal length for embeddings
    max_tokens: int = 1024              # max spatial tokens (h*w)
    max_grad_norm: float = 1.0
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 20                # bump later if needed
    num_workers: int = 4


# ---------------------------------------------------------------------------
# 2. Conv stem (VGG16-BN) -> feature tokens
# ---------------------------------------------------------------------------

class VGGConvStem(nn.Module):
    """
    VGG16-BN conv stem, projects to d_model.
    Input:  [B*T, 3, H, W]
    Output: [B*T, d_model, h, w]
    """

    def __init__(self, d_model: int):
        super().__init__()
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(*list(vgg.features.children())[:-2])
        self.proj = nn.Conv2d(512, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.stem(x)           # [B*T, 512, h, w]
        f = self.proj(f)           # [B*T, d_model, h, w]
        return f


# ---------------------------------------------------------------------------
# 3. Spatio-temporal transformer with dual heads
# ---------------------------------------------------------------------------

class SpatioTemporalTransNav(nn.Module):
    """
    Shared transformer encoder.

    Input:
        frames: [B, T, 3, H, W]
    Output:
        pred_frame:      [B, 3, H, W]
        pred_state_norm: [B, state_dim]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.conv_stem = VGGConvStem(cfg.d_model)

        self.time_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.spatial_embed = nn.Embedding(cfg.max_tokens, cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        # Navigation regressor head
        self.nav_head = nn.Sequential(
            nn.Linear(cfg.d_model, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, cfg.state_dim),
        )

        # Token-space image decoder head
        self.token_decoder = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # Reconstruction from feature map to RGB
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(cfg.d_model, cfg.d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.d_model // 2, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),   # output in [0, 1]
        )

    def forward(self, frames: torch.Tensor):
        """
        frames: [B, T, 3, H, W]
        """
        B, T, C, H, W = frames.shape
        assert T <= self.cfg.max_seq_len

        # Conv stem
        x = frames.view(B * T, C, H, W)
        feat = self.conv_stem(x)          # [B*T, D, h, w]
        BT, D, h, w = feat.shape
        assert BT == B * T
        L = h * w
        assert L <= self.cfg.max_tokens

        # Reshape to [B, T, L, D]
        feat = feat.view(B, T, D, h, w)
        tokens = feat.flatten(3)               # [B, T, D, L]
        tokens = tokens.permute(0, 1, 3, 2)    # [B, T, L, D]

        device = tokens.device

        # Positional encodings
        time_ids = torch.arange(T, device=device).view(1, T, 1).expand(B, T, L)
        space_ids = torch.arange(L, device=device).view(1, 1, L).expand(B, T, L)
        pos = self.time_embed(time_ids) + self.spatial_embed(space_ids)
        tokens = tokens + pos

        # Flatten (T, L) into sequence dimension
        tokens_flat = tokens.reshape(B, T * L, D)  # [B, T*L, D]
        enc = self.encoder(tokens_flat)           # [B, T*L, D]

        # Navigation head: global average pool
        enc_pooled = enc.mean(dim=1)              # [B, D]
        pred_state_norm = self.nav_head(enc_pooled)

        # Image head: use tokens for last frame
        start = (T - 1) * L
        end = T * L
        last_tokens = enc[:, start:end, :]        # [B, L, D]
        decoded_tokens = self.token_decoder(last_tokens)
        feat_pred = decoded_tokens.permute(0, 2, 1).reshape(B, D, h, w)

        # Upsample to original size and reconstruct
        feat_pred_up = F.interpolate(
            feat_pred, size=(H, W), mode="bilinear", align_corners=False
        )
        pred_frame = self.reconstruction_head(feat_pred_up)

        return pred_frame, pred_state_norm


# ---------------------------------------------------------------------------
# 4. Dataset for maps screenshots (from filenames)
# ---------------------------------------------------------------------------

class GreensboroMapsDataset(Dataset):
    """
    Uses files like:
        frame_<id>_<lat>_<lon>_<HEIGHT>_<zoom>_<PITCH>_<heading>.png

    Example:
        frame_00001_36.0089_-79.6937_90_18_60_120.png

    We treat all frames in the folder as one long sequence.
    """

    def __init__(
        self,
        root_dir: str,
        seq_len: int = 8,
        img_size: Tuple[int, int] = (224, 224),
        train: bool = True,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        pattern: str = "frame_*.*",  # matches png/jpg
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.seq_len = seq_len
        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),   # [0, 1]
        ])

        # 1) Find files
        all_paths = sorted(self.root_dir.rglob(pattern))

        items: List[tuple[int, np.ndarray, Path]] = []

        # Enumerate to give a temporal index
        for t_idx, p in enumerate(all_paths):
            state_vec = self._parse_filename(p)
            if state_vec is not None:
                items.append((t_idx, state_vec, p))

        if len(items) <= seq_len:
            raise ValueError(
                f"Not enough frames in {self.root_dir} for seq_len={seq_len}."
            )

        # Single sequence ID
        self.groups = {"gso": items}
        self.samples: List[tuple[str, List[int], int]] = []

        # 2) Build sliding windows: past T frames -> predict next frame
        for seq_id, seq_items in self.groups.items():
            N = len(seq_items)
            for end_pos in range(self.seq_len - 1, N - 1):
                start_pos = end_pos - self.seq_len + 1
                window_indices = list(range(start_pos, end_pos + 1))
                next_pos = end_pos + 1
                self.samples.append((seq_id, window_indices, next_pos))

        if len(self.samples) == 0:
            raise ValueError(
                "No valid sequence windows constructed. Check seq_len and data."
            )

        # 3) Compute normalization from train set
        if train and (state_mean is None or state_std is None):
            all_states = []
            for seq_id, _, next_pos in self.samples:
                _, state_vec, _ = self.groups[seq_id][next_pos]
                all_states.append(state_vec)
            arr = np.stack(all_states, axis=0)
            self.state_mean = arr.mean(axis=0)
            self.state_std = arr.std(axis=0) + 1e-6
        else:
            if state_mean is None or state_std is None:
                raise ValueError(
                    "For val/test, pass state_mean and state_std from train set."
                )
            self.state_mean = np.array(state_mean, dtype="float32")
            self.state_std = np.array(state_std, dtype="float32")

    def _parse_filename(self, path: Path) -> Optional[np.ndarray]:
        """
        Parse:
          frame_<lat>_<lon>_<HEIGHT>_<zoom>_<PITCH>_<heading>.<ext>
        into:
          [lat, lon, HEIGHT, zoom, PITCH, heading]
        """
        stem = path.stem  # without extension
        parts = stem.split("_")

        # expected: ["frame", lat, lon, HEIGHT, zoom, PITCH, heading]
        if len(parts) != 7:
            return None

        try:
            lat     = float(parts[1])
            lon     = float(parts[2])
            height  = float(parts[3])
            zoom    = float(parts[4])
            pitch   = float(parts[5])
            heading = float(parts[6])
        except ValueError:
            return None

        state_vec = np.array(
            [lat, lon, height, zoom, pitch, heading],
            dtype="float32",
        )
        return state_vec

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx: int):
        seq_id, window_indices, next_pos = self.samples[idx]
        seq_items = self.groups[seq_id]

        # Past frames
        frame_tensors = []
        for pos in window_indices:
            _, _, path = seq_items[pos]
            frame_tensors.append(self._load_image(path))
        frames = torch.stack(frame_tensors, dim=0)  # [T, 3, H, W]

        # Next frame + state
        _, state_vec, next_path = seq_items[next_pos]
        next_frame = self._load_image(next_path)

        state = state_vec
        state_norm = (state - self.state_mean) / self.state_std

        return {
            "frames": frames,                                # [T, 3, H, W]
            "next_frame": next_frame,                        # [3, H, W]
            "next_state": torch.from_numpy(state),           # [6]
            "next_state_norm": torch.from_numpy(state_norm), # [6]
        }


# ---------------------------------------------------------------------------
# 5. Training & evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    cfg: ModelConfig,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        frames = batch["frames"].to(device)               # [B, T, 3, H, W]
        next_frame = batch["next_frame"].to(device)       # [B, 3, H, W]
        next_state_norm = batch["next_state_norm"].to(device)  # [B, 6]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            pred_frame, pred_state_norm = model(frames)
            loss_img = F.mse_loss(pred_frame, next_frame)
            loss_state = F.mse_loss(pred_state_norm, next_state_norm)
            loss = loss_img + cfg.lambda_state * loss_state

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        bs = frames.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: ModelConfig,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        frames = batch["frames"].to(device)
        next_frame = batch["next_frame"].to(device)
        next_state_norm = batch["next_state_norm"].to(device)

        pred_frame, pred_state_norm = model(frames)
        loss_img = F.mse_loss(pred_frame, next_frame)
        loss_state = F.mse_loss(pred_state_norm, next_state_norm)
        loss = loss_img + cfg.lambda_state * loss_state

        bs = frames.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / max(total_count, 1)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    cfg = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Your dataset location
    img_root = r'C:\Users\admin\Desktop\spiral_avgso\maps_screenshots_circle_28150'

    # Right now we use the same folder for train + val.
    train_ds = GreensboroMapsDataset(
        root_dir=img_root,
        seq_len=cfg.seq_len,
        img_size=cfg.img_size,
        train=True,
        pattern="frame_*.*",  # if only png: "frame_*.png"
    )

    val_ds = GreensboroMapsDataset(
        root_dir=img_root,
        seq_len=cfg.seq_len,
        img_size=cfg.img_size,
        train=False,
        state_mean=train_ds.state_mean,
        state_std=train_ds.state_std,
        pattern="frame_*.*",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = SpatioTemporalTransNav(cfg).to(device)
    
    # to create a diagram of the model
    dummy = torch.randn(1, cfg.seq_len, 3, 224, 224).to(device)  # [B, T, C, H, W]
    torch.onnx.export(model, dummy, "model.onnx")
    input("ONNX model exported. Press Enter to continue...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Initialize CSV files for logging
    train_csv_path = "logs/train_losses.csv"
    val_csv_path = "logs/val_losses.csv"
    
    # Write headers
    with open(train_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss'])
    
    with open(val_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'val_loss'])

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, cfg
        )
        val_loss = evaluate(
            model, val_loader, device, cfg
        )
        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )
        
        # Log losses to CSV files
        with open(train_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss])
        
        with open(val_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, val_loss])

        ckpt_path = f"checkpoints/transnav_gso_epoch_{epoch:03d}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "state_mean": train_ds.state_mean,
                "state_std": train_ds.state_std,
                "config": cfg.__dict__,
            },
            ckpt_path,
        )


if __name__ == "__main__":
    main()
