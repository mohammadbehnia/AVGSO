# -*- coding: utf-8 -*-
"""
GSO-Data Multiframe Single-View Localization (ONE FOLDER, sliding windows) — VS Code ready

Dataset assumptions (per your update):
- All frames are in ONE folder (no subfolders).
- Filenames follow:
    frame_<FRAMEID>_<LAT>_<LON>_<HEADING>.<ext>
  Example:
    frame_000001_36.045944_-79.960988_90.jpg
  Where:
    FRAMEID = integer (used for ordering)
    LAT, LON = decimal degrees
    HEADING = degrees (or radians; auto-detected)

Your requested sampling:
- window 1: frames 1,2,3,4
- window 2: frames 2,3,4,5
- window 3: frames 3,4,5,6
...and so on.

Tasks:
1) Predict aggregated / overall location for the 4-frame window (XY meters) from ALL 4 frames.
2) Predict 4th-frame location and frame3->frame4 displacement from ONLY the first 3 frames.
   Heading is calculated from the predicted displacement instead of being learned directly.

Model (Spatio-Temporal Transformer style from your project):
- VGG16 feature maps Ft ∈ R^{C×h×w} per frame
- Each spatial cell -> token x_{t,ℓ} ∈ R^C  => T×L tokens
- Add learned time embedding e_t^time and spatial embedding e_ℓ^space
- Transformer encoder
- Two passes:
    Pass A (T=4): overall location via CLS
    Pass B (T=3): frame4 pose via CLS (context only)

Loss (kept minimal):
- Huber for XY regression
- Heading via calculated frame3->frame4 motion direction, not a separate learned heading head
- Optional InfoNCE (recommended if you want stronger place embeddings)

Run:
  pip install -r requirements.txt
  python gso_multiframe_stt.py --data_root "D:\\PhD\\Code\\gso\\New_GSO\\maps_screenshots" --epochs 20 --use_infonce 1
"""

import os
import csv
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import torchvision.transforms as T
from torchvision.models import vgg16, VGG16_Weights, resnet18, ResNet18_Weights

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    data_root: str

    image_size: int = 224
    T_frames: int = 4
    stride: int = 1

    # Split strategy (single continuous route):
    # Use chronological split by window start index to avoid leakage.
    train_frac: float = 0.80
    eval_frac: float = 0.10
    test_frac: float = 0.10

    seed: int = 42

    # Training
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 10
    lr: float = 5e-5
    weight_decay: float = 0.05

    # Minimal multi-task weights
    w_overall_xy: float = 1.0
    w_xy4: float = 1.0
    w_head: float = 0.05
    w_delta_xy: float = 0.30

    # Regularization
    dropout: float = 0.25

    # Optional InfoNCE
    use_infonce: bool = True
    w_nce: float = 0.15
    temperature: float = 0.07

    # Backbone
    use_pretrained: bool = False

BACKBONE = 'vgg' # resnet

# -----------------------------
# Geo helpers (lat/lon -> local meters)
# -----------------------------
EARTH_R = 6378137.0  # meters

def ll_to_local_xy(lat_deg: torch.Tensor, lon_deg: torch.Tensor, lat0_deg: float, lon0_deg: float):
    """Local tangent approximation: x east, y north in meters."""
    lat = torch.deg2rad(lat_deg)
    lon = torch.deg2rad(lon_deg)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)

    x = (lon - lon0) * math.cos(lat0) * EARTH_R
    y = (lat - lat0) * EARTH_R
    return x, y

def heading_to_rad(h: torch.Tensor):
    """If heading looks like degrees (> 2pi), convert to radians."""
    if torch.max(torch.abs(h)).item() > 6.5:
        return torch.deg2rad(h)
    return h


# -----------------------------
# File parsing
# -----------------------------
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

def parse_frame_name(fname: str) -> Optional[Tuple[int, float, float, float]]:
    """
    Parse: frame_<FRAMEID>_<LAT>_<LON>_<HEADING>
    Returns (frame_id, lat, lon, heading)
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    if len(parts) < 5:
        return None
    if parts[0].lower() != "frame":
        return None
    try:
        frame_id = int(parts[1])
        lat = float(parts[2])
        lon = float(parts[3])
        heading = float(parts[4])
        return frame_id, lat, lon, heading
    except ValueError:
        return None

def list_frames_one_folder(data_root: str) -> List[Tuple[str, int, float, float, float]]:
    """
    Returns list of tuples:
      (full_path, frame_id, lat, lon, heading)
    sorted by frame_id.
    """
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data_root not found: {data_root}")

    items = []
    for fn in os.listdir(data_root):
        ext = os.path.splitext(fn)[1].lower()
        if ext not in IMG_EXT:
            continue
        if not fn.lower().startswith("frame_"):
            continue
        parsed = parse_frame_name(fn)
        if parsed is None:
            continue
        frame_id, lat, lon, heading = parsed
        items.append((os.path.join(data_root, fn), frame_id, lat, lon, heading))

    items.sort(key=lambda x: x[1])
    if len(items) < 4:
        raise RuntimeError(
            f"Found only {len(items)} valid frames in {data_root}. "
            "Need at least 4. Ensure filenames match: frame_<id>_<lat>_<lon>_<heading>.<ext>"
        )
    return items


# -----------------------------
# Dataset (sliding windows)
# -----------------------------
class GSOSlidingWindowDataset(Dataset):
    """
    One continuous sequence in one folder -> sliding window dataset.

    Sample i returns frames [i, i+1, i+2, i+3] (stride=1 by default).

    Returns:
      images: (4,3,H,W)
      gt_overall_xy: mean(xy over 4)
      gt_xy4: xy of 4th frame
      gt_heading4: heading of 4th frame (rad)
    """
    def __init__(
        self,
        frames: List[Tuple[str, int, float, float, float]],
        cfg: CFG,
        start_idx: int,
        end_idx: int,
        lat0: Optional[float] = None,
        lon0: Optional[float] = None,
        xy_mean: Optional[torch.Tensor] = None,
        xy_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.frames = frames
        self.cfg = cfg
        self.start_idx = start_idx
        self.end_idx = end_idx  # exclusive over possible window starts

        # origin for local XY
        if lat0 is None or lon0 is None:
            lat0 = frames[0][2]
            lon0 = frames[0][3]
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self.xy_mean = xy_mean.float() if xy_mean is not None else torch.zeros(2, dtype=torch.float32)
        self.xy_std = xy_std.float() if xy_std is not None else torch.ones(2, dtype=torch.float32)

        self.tf = T.Compose([
            T.Resize((cfg.image_size, cfg.image_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return max(0, self.end_idx - self.start_idx)

    def _load_img(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.tf(img)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        idx = self.start_idx + i
        Tframes = self.cfg.T_frames
        stride = self.cfg.stride

        chosen = [self.frames[idx + k*stride] for k in range(Tframes)]
        paths = [c[0] for c in chosen]

        imgs = torch.stack([self._load_img(p) for p in paths], dim=0)  # (4,3,H,W)

        lat = torch.tensor([c[2] for c in chosen], dtype=torch.float32)
        lon = torch.tensor([c[3] for c in chosen], dtype=torch.float32)
        head = torch.tensor([c[4] for c in chosen], dtype=torch.float32)
        head = heading_to_rad(head)

        x, y = ll_to_local_xy(lat, lon, self.lat0, self.lon0)
        xy = torch.stack([x, y], dim=-1)

        gt_overall_xy = (xy.mean(dim=0) - self.xy_mean) / self.xy_std
        gt_xy4 = (xy[3] - self.xy_mean) / self.xy_std
        # Auxiliary relative displacement target, normalized by the same common radius scale.
        # This preserves motion magnitude/direction without subtracting the center.
        gt_delta34 = (xy[3] - xy[2]) / self.xy_std

        return {
            "images": imgs,
            "gt_overall_xy": gt_overall_xy,
            "gt_xy4": gt_xy4,
            "gt_delta34": gt_delta34,
            "gt_heading4": head[3],
            "window_start_idx": torch.tensor(idx, dtype=torch.long),
            "frame_ids": torch.tensor([c[1] for c in chosen], dtype=torch.long),
            "paths": paths,
        }


def chronological_split(num_windows: int, train_frac: float, eval_frac: float, test_frac: float):
    assert abs(train_frac + eval_frac + test_frac - 1.0) < 1e-6
    n_train = int(num_windows * train_frac)
    n_eval = int(num_windows * eval_frac)
    tr = (0, n_train)
    ev = (n_train, n_train + n_eval)
    te = (n_train + n_eval, num_windows)
    return tr, ev, te


def compute_xy_normalization(
    frames: List[Tuple[str, int, float, float, float]],
    cfg: CFG,
    start_idx: int,
    end_idx: int,
    lat0: float,
    lon0: float,
):
    """
    Compute center/radius normalization from training windows only, without loading images.

    Instead of separate X/Y standard deviations, this uses one shared radius scale:
        xy_norm = (xy_meters - xy_center) / xy_radius

    This better preserves geometry for spiral/circular trajectories because X and Y keep
    the same physical scale.
    """
    values = []
    for idx in range(start_idx, end_idx):
        chosen = [frames[idx + k * cfg.stride] for k in range(cfg.T_frames)]
        lat = torch.tensor([c[2] for c in chosen], dtype=torch.float32)
        lon = torch.tensor([c[3] for c in chosen], dtype=torch.float32)
        x, y = ll_to_local_xy(lat, lon, lat0, lon0)
        xy = torch.stack([x, y], dim=-1)
        values.append(xy.mean(dim=0))
        values.append(xy[2])
        values.append(xy[3])

    vals = torch.stack(values, dim=0)
    xy_center = vals.mean(dim=0)
    distances = torch.norm(vals - xy_center, dim=-1)
    xy_radius = torch.quantile(distances, 0.95).clamp_min(1.0)

    # Keep a 2-vector scale so the rest of the code can denormalize with broadcasting:
    # pred_xy_m = pred_xy_norm * xy_scale + xy_center
    xy_scale = torch.tensor([xy_radius.item(), xy_radius.item()], dtype=torch.float32)
    return xy_center, xy_scale


def move_tensor_batch_to_device(batch: Dict[str, object], device: str):
    """Move only tensor values to device; leave paths/metadata strings on CPU."""
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


# -----------------------------
# Spatio-Temporal Transformer (VGG16 feature maps -> spatial tokens)
# -----------------------------
class SpatioTemporalTransformer(nn.Module):
    """
    Spatio–temporal tokens:
      VGG16 => Ft ∈ R^{C×h×w}
      spatial cell => token x_{t,ℓ} ∈ R^C
      flatten => T×L tokens
      + learned time embedding e_t^time
      + learned spatial embedding e_ℓ^space
      transformer encoder

    Two passes:
      Pass A (T=4): overall XY from CLS
      Pass B (T=3): frame4 pose from first 3 frames only (CLS)
    """
    def __init__(self, cfg: CFG, embed_dim: int = 256, nhead: int = 8, layers: int = 4, dropout: float = 0.1, backbone = 'vgg'):
        super().__init__()
        self.cfg = cfg
        self.T = cfg.T_frames
        backbone = backbone.lower()

        if backbone == 'vgg':
            weights = VGG16_Weights.IMAGENET1K_V1 if cfg.use_pretrained else None
            base = vgg16(weights=weights)

            self.backbone = base.features
            self.in_channels = 512

        elif backbone == 'resnet':
            weights = ResNet18_Weights.IMAGENET1K_V1 if cfg.use_pretrained else None
            base = resnet18(weights=weights)
        
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            self.in_channels = base.fc.in_features

        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use 'vgg' or 'resnet'.")

        # For 224 input, usually 7x7. We'll adapt if needed.
        self.h = 7
        self.w = 7
        self.L = self.h * self.w

        self.proj = nn.Linear(self.in_channels, embed_dim)

        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls, std=0.02)

        self.time_emb_4 = nn.Embedding(4, embed_dim)
        self.time_emb_3 = nn.Embedding(3, embed_dim)
        self.space_emb = nn.Embedding(self.L, embed_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=4*embed_dim, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)

        self.head_overall = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2),
        )
        self.head_frame4 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2),  # x4,y4 only; heading is calculated from predicted delta34
        )
        self.head_delta34 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2),  # delta x,y from frame 3 to frame 4
        )

    def _frame_tokens(self, images_btchw: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(images_btchw)  # (B*T,512,h,w)
        B_T, C, Hf, Wf = feat.shape
        if (Hf, Wf) != (self.h, self.w):
            self.h, self.w = Hf, Wf
            new_L = Hf * Wf
            if new_L != self.L:
                self.L = new_L
                self.space_emb = nn.Embedding(self.L, self.space_emb.embedding_dim).to(feat.device)

        feat = feat.flatten(2).transpose(1, 2)  # (B*T, L, C)
        tok = self.proj(feat)                   # (B*T, L, D)
        return tok

    def _add_positional(self, tok: torch.Tensor, Tctx: int) -> torch.Tensor:
        """
        tok: (B, Tctx*L, D) time-major then space
        """
        B, TL, D = tok.shape
        L = TL // Tctx
        t_idx = torch.arange(Tctx, device=tok.device).repeat_interleave(L)
        s_idx = torch.arange(L, device=tok.device).repeat(Tctx)

        if Tctx == 4:
            tok = tok + self.time_emb_4(t_idx)[None, :, :] + self.space_emb(s_idx)[None, :, :]
        else:
            tok = tok + self.time_emb_3(t_idx)[None, :, :] + self.space_emb(s_idx)[None, :, :]
        return tok

    def forward(self, images: torch.Tensor):
        """
        images: (B,4,3,H,W)
        """
        B, T, C, H, W = images.shape
        assert T == 4, "Expected exactly 4 input frames."

        x = images.view(B*T, C, H, W)
        tok_bt = self._frame_tokens(x)                 # (B*T, L, D)
        L = tok_bt.size(1)

        # Pass A (4 frames)
        tok_all = tok_bt.view(B, T*L, -1)
        tok_all = self._add_positional(tok_all, Tctx=4)
        cls = self.cls.expand(B, -1, -1)
        seq_all = torch.cat([cls, tok_all], dim=1)
        out_all = self.encoder(seq_all)
        overall_xy = self.head_overall(out_all[:, 0])

        # Pass B (first 3 only)
        Tctx = 3
        tok_ctx = tok_bt.view(B, T, L, -1)[:, :Tctx]
        tok_ctx = tok_ctx.reshape(B, Tctx*L, -1)
        tok_ctx = self._add_positional(tok_ctx, Tctx=3)
        cls2 = self.cls.expand(B, -1, -1)
        seq_ctx = torch.cat([cls2, tok_ctx], dim=1)
        out_ctx = self.encoder(seq_ctx)

        pred_xy4 = self.head_frame4(out_ctx[:, 0])      # (B,2): x4,y4
        pred_delta34 = self.head_delta34(out_ctx[:, 0])  # (B,2): dx,dy from frame 3 to frame 4

        # Heading is NOT learned as a separate output.
        # Because x=east and y=north, a unit displacement [dx, dy]
        # is equivalent to [sin(heading), cos(heading)].
        pred_sincos4 = F.normalize(pred_delta34, dim=-1, eps=1e-6)

        # embeddings for optional InfoNCE
        tok_frames = tok_bt.view(B, T, L, -1)
        z4 = F.normalize(tok_frames[:, 3].mean(dim=1), dim=-1)
        zpos = F.normalize(tok_frames[:, :3].mean(dim=(1, 2)), dim=-1)

        return {
            "overall_xy": overall_xy,
            "pred_xy4": pred_xy4,
            "pred_delta34": pred_delta34,
            "pred_sincos4": pred_sincos4,
            "z4": z4,
            "zpos": zpos,
        }


# -----------------------------
# Losses (minimal + optional InfoNCE)
# -----------------------------
def huber(pred: torch.Tensor, gt: torch.Tensor, delta: float = 1.0):
    return F.smooth_l1_loss(pred, gt, beta=delta)

def heading_loss_sincos(pred_sincos: torch.Tensor, gt_heading_rad: torch.Tensor):
    gt = torch.stack([torch.sin(gt_heading_rad), torch.cos(gt_heading_rad)], dim=-1)
    gt = F.normalize(gt, dim=-1)
    cos_sim = (pred_sincos * gt).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cos_sim).mean()


def heading_loss_from_delta(pred_delta: torch.Tensor, gt_delta: torch.Tensor):
    """
    Calculate heading loss from motion direction instead of learning heading directly.

    pred_delta and gt_delta are [dx, dy] where:
      dx = east displacement
      dy = north displacement

    Since compass-style heading uses atan2(dx, dy), the normalized [dx, dy]
    vector is equivalent to [sin(heading), cos(heading)].
    """
    pred_dir = F.normalize(pred_delta, dim=-1, eps=1e-6)
    gt_dir = F.normalize(gt_delta, dim=-1, eps=1e-6)
    cos_sim = (pred_dir * gt_dir).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cos_sim).mean()

def info_nce(z_anchor: torch.Tensor, z_pos: torch.Tensor, temperature: float = 0.07):
    logits = (z_anchor @ z_pos.t()) / temperature
    labels = torch.arange(z_anchor.size(0), device=z_anchor.device)
    return F.cross_entropy(logits, labels)

def total_loss(out: dict, batch: dict, cfg: CFG):
    l_overall = huber(out["overall_xy"], batch["gt_overall_xy"])
    l_xy4 = huber(out["pred_xy4"], batch["gt_xy4"])
    l_delta = huber(out["pred_delta34"], batch["gt_delta34"])

    # Heading is calculated from the predicted frame3->frame4 displacement.
    # The target heading is also calculated from the ground-truth frame3->frame4 displacement.
    l_head = heading_loss_from_delta(out["pred_delta34"], batch["gt_delta34"])

    loss = cfg.w_overall_xy*l_overall + cfg.w_xy4*l_xy4 + cfg.w_delta_xy*l_delta + cfg.w_head*l_head
    logs = {
        "l_overall": l_overall.detach(),
        "l_xy4": l_xy4.detach(),
        "l_delta": l_delta.detach(),
        "l_head": l_head.detach(),
    }

    if cfg.use_infonce:
        l_nce = info_nce(out["z4"], out["zpos"], temperature=cfg.temperature)
        loss = loss + cfg.w_nce * l_nce
        logs["l_nce"] = l_nce.detach()

    return loss, logs


# -----------------------------
# Train / Eval
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device, cfg: CFG):
    model.eval()
    tot = 0.0
    n = 0
    for batch in loader:
        batch = move_tensor_batch_to_device(batch, device)
        out = model(batch["images"])
        loss, _ = total_loss(out, batch, cfg)
        tot += loss.item() * batch["images"].size(0)
        n += batch["images"].size(0)
    model.train()
    return tot / max(n, 1)

@torch.no_grad()
def evaluate_meters(model, loader, device, cfg: CFG, xy_mean: torch.Tensor, xy_std: torch.Tensor):
    """
    Evaluate the trained model in real meter units.

    The model predicts normalized XY values because training uses:
        xy_norm = (xy_meters - xy_mean) / xy_std

    This function converts predictions and labels back to meters, then reports
    Euclidean localization error. These are the values to present to an audience.
    """
    model.eval()

    xy_mean = xy_mean.to(device)
    xy_std = xy_std.to(device)

    overall_errors = []
    frame4_errors = []
    heading_errors_deg = []

    gt_overall_all = []
    pred_overall_all = []
    gt_frame4_all = []
    pred_frame4_all = []
    frame4_ids_all = []

    for batch in loader:
        batch = move_tensor_batch_to_device(batch, device)
        out = model(batch["images"])

        pred_overall_m = out["overall_xy"] * xy_std + xy_mean
        gt_overall_m = batch["gt_overall_xy"] * xy_std + xy_mean

        pred_xy4_m = out["pred_xy4"] * xy_std + xy_mean
        gt_xy4_m = batch["gt_xy4"] * xy_std + xy_mean

        overall_err = torch.norm(pred_overall_m - gt_overall_m, dim=-1)
        frame4_err = torch.norm(pred_xy4_m - gt_xy4_m, dim=-1)

        # Motion-based heading: heading is calculated from frame3->frame4 displacement.
        # x=east, y=north, so compass heading is atan2(dx, dy).
        pred_heading = torch.atan2(out["pred_delta34"][:, 0], out["pred_delta34"][:, 1])
        gt_heading = torch.atan2(batch["gt_delta34"][:, 0], batch["gt_delta34"][:, 1])
        heading_diff = torch.atan2(torch.sin(pred_heading - gt_heading), torch.cos(pred_heading - gt_heading))
        heading_err_deg = torch.rad2deg(torch.abs(heading_diff))

        overall_errors.append(overall_err.detach().cpu())
        frame4_errors.append(frame4_err.detach().cpu())
        heading_errors_deg.append(heading_err_deg.detach().cpu())

        gt_overall_all.append(gt_overall_m.detach().cpu())
        pred_overall_all.append(pred_overall_m.detach().cpu())
        gt_frame4_all.append(gt_xy4_m.detach().cpu())
        pred_frame4_all.append(pred_xy4_m.detach().cpu())

        if torch.is_tensor(batch.get("frame_ids")):
            frame4_ids_all.append(batch["frame_ids"][:, -1].detach().cpu())

    overall_errors = torch.cat(overall_errors)
    frame4_errors = torch.cat(frame4_errors)
    heading_errors_deg = torch.cat(heading_errors_deg)

    gt_overall_all = torch.cat(gt_overall_all)
    pred_overall_all = torch.cat(pred_overall_all)
    gt_frame4_all = torch.cat(gt_frame4_all)
    pred_frame4_all = torch.cat(pred_frame4_all)
    frame4_ids_all = torch.cat(frame4_ids_all) if frame4_ids_all else torch.arange(frame4_errors.numel())

    metrics = {
        "overall_mean_m": overall_errors.mean().item(),
        "overall_median_m": overall_errors.median().item(),
        "overall_rmse_m": torch.sqrt((overall_errors ** 2).mean()).item(),
        "overall_p90_m": torch.quantile(overall_errors, 0.90).item(),
        "frame4_mean_m": frame4_errors.mean().item(),
        "frame4_median_m": frame4_errors.median().item(),
        "frame4_rmse_m": torch.sqrt((frame4_errors ** 2).mean()).item(),
        "frame4_p90_m": torch.quantile(frame4_errors, 0.90).item(),
        "heading_mean_deg": heading_errors_deg.mean().item(),
        "heading_median_deg": heading_errors_deg.median().item(),
        "heading_rmse_deg": torch.sqrt((heading_errors_deg ** 2).mean()).item(),
        "heading_p90_deg": torch.quantile(heading_errors_deg, 0.90).item(),
    }

    details = {
        "overall_errors": overall_errors,
        "frame4_errors": frame4_errors,
        "heading_errors_deg": heading_errors_deg,
        "gt_overall_xy_m": gt_overall_all,
        "pred_overall_xy_m": pred_overall_all,
        "gt_frame4_xy_m": gt_frame4_all,
        "pred_frame4_xy_m": pred_frame4_all,
        "frame4_ids": frame4_ids_all,
    }

    model.train()
    return metrics, details


def save_final_metrics_csv(metrics: Dict[str, float], output_dir: str):
    path = os.path.join(output_dir, "final_meter_metrics.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, f"{v:.6f}"])
    return path


def save_test_error_csv(details: Dict[str, torch.Tensor], output_dir: str):
    path = os.path.join(output_dir, "test_per_sample_errors.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame4_id",
            "overall_error_m",
            "frame4_error_m",
            "heading_error_deg",
            "gt_frame4_x_m",
            "gt_frame4_y_m",
            "pred_frame4_x_m",
            "pred_frame4_y_m",
        ])
        for i in range(details["frame4_errors"].numel()):
            writer.writerow([
                int(details["frame4_ids"][i].item()),
                f"{details['overall_errors'][i].item():.6f}",
                f"{details['frame4_errors'][i].item():.6f}",
                f"{details['heading_errors_deg'][i].item():.6f}",
                f"{details['gt_frame4_xy_m'][i, 0].item():.6f}",
                f"{details['gt_frame4_xy_m'][i, 1].item():.6f}",
                f"{details['pred_frame4_xy_m'][i, 0].item():.6f}",
                f"{details['pred_frame4_xy_m'][i, 1].item():.6f}",
            ])
    return path


def plot_loss_curves(history: List[Tuple[int, float, float]], output_dir: str):
    if plt is None or not history:
        return None
    epochs = [h[0] for h in history]
    train = [h[1] for h in history]
    val = [h[2] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.plot(epochs, train, marker="o", linewidth=2, label="Train loss")
    ax.plot(epochs, val, marker="o", linewidth=2, label="Validation loss")
    ax.set_title("Normalized Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized Huber Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, "loss_curve.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_meter_summary(metrics: Dict[str, float], output_dir: str):
    if plt is None:
        return None
    labels = ["Overall\nMean", "Overall\nMedian", "Overall\nRMSE", "Frame-4\nMean", "Frame-4\nMedian", "Frame-4\nRMSE"]
    values = [
        metrics["overall_mean_m"],
        metrics["overall_median_m"],
        metrics["overall_rmse_m"],
        metrics["frame4_mean_m"],
        metrics["frame4_median_m"],
        metrics["frame4_rmse_m"],
    ]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    ax.bar(labels, values)
    ax.set_title("Test Localization Error in Meters")
    ax.set_ylabel("Error (meters)")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()

    path = os.path.join(output_dir, "meter_error_summary.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_error_histograms(details: Dict[str, torch.Tensor], output_dir: str):
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.hist(details["overall_errors"].numpy(), bins=40, alpha=0.55, label="Overall/window")
    ax.hist(details["frame4_errors"].numpy(), bins=40, alpha=0.55, label="Frame-4")
    ax.set_title("Distribution of Test Localization Errors")
    ax.set_xlabel("Error (meters)")
    ax.set_ylabel("Number of windows")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, "meter_error_histogram.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_trajectory_comparison(details: Dict[str, torch.Tensor], output_dir: str, max_points: int = 1200):
    if plt is None:
        return None

    gt = details["gt_frame4_xy_m"]
    pred = details["pred_frame4_xy_m"]
    n = gt.shape[0]
    step = max(1, n // max_points)
    gt = gt[::step]
    pred = pred[::step]

    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    ax.plot(gt[:, 0].numpy(), gt[:, 1].numpy(), linewidth=2, label="Ground truth")
    ax.plot(pred[:, 0].numpy(), pred[:, 1].numpy(), linewidth=2, linestyle="--", label="Predicted")
    ax.set_title("Frame-4 Trajectory: Ground Truth vs Prediction")
    ax.set_xlabel("Local X / East (meters)")
    ax.set_ylabel("Local Y / North (meters)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, "trajectory_prediction_vs_ground_truth.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_heading_summary(metrics: Dict[str, float], output_dir: str):
    if plt is None:
        return None
    labels = ["Mean", "Median", "RMSE", "90th %ile"]
    values = [
        metrics["heading_mean_deg"],
        metrics["heading_median_deg"],
        metrics["heading_rmse_deg"],
        metrics["heading_p90_deg"],
    ]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    ax.bar(labels, values)
    ax.set_title("Test Heading Error")
    ax.set_ylabel("Error (degrees)")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.1f}°", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()

    path = os.path.join(output_dir, "heading_error_summary.png")
    fig.savefig(path)
    plt.close(fig)
    return path

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=r"C:\Users\admin\Desktop\spiral_avgso\maps_screenshots_Lawn_31668")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--use_infonce", type=int, default=0)
    ap.add_argument("--use_pretrained", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--output_dir", type=str, default="results")
    args = ap.parse_args()

    cfg = CFG(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_infonce=bool(args.use_infonce),
        use_pretrained=bool(args.use_pretrained),
        num_workers=args.num_workers,
        dropout=args.dropout,
    )


    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    frames = list_frames_one_folder(cfg.data_root)
    total_frames = len(frames)

    # windows count for stride=1, T=4: N - 3
    num_windows = total_frames - (cfg.T_frames - 1) * cfg.stride
    if num_windows <= 0:
        raise RuntimeError(f"Not enough frames for windows. frames={total_frames}, need >= {cfg.T_frames}")

    (tr_s, tr_e), (ev_s, ev_e), (te_s, te_e) = chronological_split(
        num_windows, cfg.train_frac, cfg.eval_frac, cfg.test_frac
    )

    lat0 = float(frames[0][2])
    lon0 = float(frames[0][3])
    xy_mean, xy_std = compute_xy_normalization(frames, cfg, tr_s, tr_e, lat0, lon0)

    train_ds = GSOSlidingWindowDataset(frames, cfg, tr_s, tr_e, lat0=lat0, lon0=lon0, xy_mean=xy_mean, xy_std=xy_std)
    eval_ds  = GSOSlidingWindowDataset(frames, cfg, ev_s, ev_e, lat0=lat0, lon0=lon0, xy_mean=xy_mean, xy_std=xy_std)
    test_ds  = GSOSlidingWindowDataset(frames, cfg, te_s, te_e, lat0=lat0, lon0=lon0, xy_mean=xy_mean, xy_std=xy_std)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    eval_loader  = DataLoader(eval_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    model = SpatioTemporalTransformer(cfg, embed_dim=256, nhead=8, layers=4, dropout=cfg.dropout, backbone=BACKBONE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    best = 1e9
    history = []

    print(f"Data root: {cfg.data_root}")
    print(f"Frames: {total_frames} | Windows: {num_windows}")
    print(f"Split windows: train={len(train_ds)} eval={len(eval_ds)} test={len(test_ds)} (chronological)")
    print(f"Use InfoNCE: {cfg.use_infonce} (w={cfg.w_nce}, T={cfg.temperature})")
    print(f"XY center/radius normalization center={xy_mean.tolist()} scale={xy_std.tolist()}")
    print(f"Dropout: {cfg.dropout} | Relative delta loss weight: {cfg.w_delta_xy}")
    print("Heading strategy: calculated from predicted frame3->frame4 displacement")
    print("AMP/autocast: disabled")
    print(f"Saving graphs/metrics to: {args.output_dir}")
    print("Saving best model to: checkpoints/best.pt")

    with open(os.path.join(args.output_dir, 'loss_log.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'train_loss', 'val_loss'])

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running = 0.0
            seen = 0

            for batch in train_loader:
                batch = move_tensor_batch_to_device(batch, device)

                opt.zero_grad(set_to_none=True)
                out = model(batch["images"])
                loss, logs = total_loss(out, batch, cfg)

                if not torch.isfinite(loss):
                    print("NaN/Inf loss detected")
                    print("window_start_idx:", batch.get("window_start_idx"))
                    print("frame_ids:", batch.get("frame_ids"))
                    print("paths:", batch.get("paths"))
                    print("overall_xy finite:", torch.isfinite(out["overall_xy"]).all().item())
                    print("pred_xy4 finite:", torch.isfinite(out["pred_xy4"]).all().item())
                    print("pred_delta34 finite:", torch.isfinite(out["pred_delta34"]).all().item())
                    print("pred_sincos4 finite:", torch.isfinite(out["pred_sincos4"]).all().item())
                    print("gt_overall_xy finite:", torch.isfinite(batch["gt_overall_xy"]).all().item())
                    print("gt_xy4 finite:", torch.isfinite(batch["gt_xy4"]).all().item())
                    print("gt_delta34 finite:", torch.isfinite(batch["gt_delta34"]).all().item())
                    print("gt_heading4 finite:", torch.isfinite(batch["gt_heading4"]).all().item())
                    raise RuntimeError("Stopping because loss became NaN/Inf")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                running += loss.item() * batch["images"].size(0)
                seen += batch["images"].size(0)
                print(f"{running:.2f}    |    {loss.item():.2f}    |    {seen}    |    {(running / max(seen, 1)):.2f}", end='\r')

            train_loss = running / max(seen, 1)
            eval_loss = evaluate(model, eval_loader, device, cfg)
            print(f"Epoch {epoch:03d} | train={train_loss:.4f} | eval={eval_loss:.4f}")
            writer.writerow([f"{epoch:03d}" , f"{train_loss:.4f}" , f"{eval_loss:.4f}"])
            history.append((epoch, train_loss, eval_loss))

            if eval_loss < best:
                best = eval_loss
                torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "xy_center": xy_mean, "xy_scale": xy_std}, "checkpoints/best.pt")

    ckpt = torch.load("checkpoints/best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])

    test_loss = evaluate(model, test_loader, device, cfg)
    meter_metrics, meter_details = evaluate_meters(model, test_loader, device, cfg, xy_mean, xy_std)

    metrics_csv = save_final_metrics_csv(meter_metrics, args.output_dir)
    errors_csv = save_test_error_csv(meter_details, args.output_dir)

    graph_paths = []
    for path in [
        plot_loss_curves(history, args.output_dir),
        plot_meter_summary(meter_metrics, args.output_dir),
        plot_error_histograms(meter_details, args.output_dir),
        plot_trajectory_comparison(meter_details, args.output_dir),
        plot_heading_summary(meter_metrics, args.output_dir),
    ]:
        if path is not None:
            graph_paths.append(path)

    print(f"Best eval normalized loss: {best:.4f}")
    print(f"Test normalized loss: {test_loss:.4f}")
    print("Test meter errors:")
    print(f"  Overall/window mean error:   {meter_metrics['overall_mean_m']:.2f} m")
    print(f"  Overall/window median error: {meter_metrics['overall_median_m']:.2f} m")
    print(f"  Overall/window RMSE:         {meter_metrics['overall_rmse_m']:.2f} m")
    print(f"  Overall/window 90th %ile:    {meter_metrics['overall_p90_m']:.2f} m")
    print(f"  Frame-4 mean error:          {meter_metrics['frame4_mean_m']:.2f} m")
    print(f"  Frame-4 median error:        {meter_metrics['frame4_median_m']:.2f} m")
    print(f"  Frame-4 RMSE:                {meter_metrics['frame4_rmse_m']:.2f} m")
    print(f"  Frame-4 90th %ile:           {meter_metrics['frame4_p90_m']:.2f} m")
    print("Test heading errors:")
    print(f"  Heading mean error:          {meter_metrics['heading_mean_deg']:.2f} deg")
    print(f"  Heading median error:        {meter_metrics['heading_median_deg']:.2f} deg")
    print(f"  Heading RMSE:                {meter_metrics['heading_rmse_deg']:.2f} deg")
    print(f"  Heading 90th %ile:           {meter_metrics['heading_p90_deg']:.2f} deg")
    print(f"Saved metrics CSV: {metrics_csv}")
    print(f"Saved per-sample error CSV: {errors_csv}")
    if graph_paths:
        print("Saved graphs:")
        for path in graph_paths:
            print(f"  {path}")
    else:
        print("Graphs were not saved because matplotlib is not available.")

    print("Done.")

if __name__ == "__main__":
    main()
