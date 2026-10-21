# -*- coding: utf-8 -*-
"""
GSO-Data Multiframe Single-View Localization
V4-original-center-radius version

Base idea:
- Keep the original localization architecture and training flow:
  VGG/ResNet feature maps -> spatio-temporal transformer
  Pass A: 4 frames -> overall/window XY
  Pass B: first 3 frames -> frame-4 XY + heading sin/cos
- Apply ONLY center/radius normalization to XY targets.
- Add V3-style metric output and graphs so results are easy to compare.

Dataset assumptions:
- All frames are in ONE folder.
- Filenames follow:
    frame_<FRAMEID>_<LAT>_<LON>_<HEADING>.<ext>
  Example:
    frame_000001_36.045944_-79.960988_90.jpg

Run example from:
  C:\\Users\\admin\\Desktop\\spiral_avgso

python .\\gso_multiframe_spiral_V4_original_center_radius.py ^
  --data_root "C:\\Users\\admin\\Desktop\\spiral_avgso\\maps_screenshots_spiral_28150" ^
  --epochs 10 ^
  --batch_size 4 ^
  --use_infonce 0 ^
  --use_pretrained 1 ^
  --output_dir "C:\\Users\\admin\\Desktop\\spiral_avgso\\localization_spiral_vgg\\v4_original_center_radius_results"

Important:
- Checkpoints save inside output_dir\\checkpoints.
- Graphs/CSVs save directly inside output_dir.
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

    # Split strategy:
    # Use chronological split by window start index to avoid leakage.
    train_frac: float = 0.80
    eval_frac: float = 0.10
    test_frac: float = 0.10

    seed: int = 42

    # Training
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 0.05

    # Multi-task weights
    w_overall_xy: float = 1.0
    w_xy4: float = 1.0
    w_head: float = 0.05

    # Optional InfoNCE
    use_infonce: bool = False
    w_nce: float = 0.15
    temperature: float = 0.07

    # Backbone
    use_pretrained: bool = True

    # Regular dropout; default kept as original unless overridden
    dropout: float = 0.1


BACKBONE = "vgg"  # or "resnet"
EARTH_R = 6378137.0
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


# -----------------------------
# Geo helpers
# -----------------------------
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
    Returns:
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
            "Need at least 4. Ensure filenames match frame_<id>_<lat>_<lon>_<heading>.<ext>"
        )
    return items


# -----------------------------
# Center/radius normalization
# -----------------------------
def compute_center_radius_normalization(
    frames: List[Tuple[str, int, float, float, float]],
    cfg: CFG,
    start_idx: int,
    end_idx: int,
    lat0: float,
    lon0: float,
    use_start_as_center: bool = True,
):
    """
    Center/radius normalization for XY targets.

    Local XY is first computed relative to lat0/lon0.
    If use_start_as_center=True:
      center = [0,0] because lat0/lon0 is the start/center point.
    Radius is computed from the training-window targets only.

    If your plotted spiral starts at the center, this matches your trajectory design.
    No lat/lon conversion back is needed for training; denormalization returns local meters.
    """
    values = []
    for idx in range(start_idx, end_idx):
        chosen = [frames[idx + k * cfg.stride] for k in range(cfg.T_frames)]
        lat = torch.tensor([c[2] for c in chosen], dtype=torch.float32)
        lon = torch.tensor([c[3] for c in chosen], dtype=torch.float32)
        x, y = ll_to_local_xy(lat, lon, lat0, lon0)
        xy = torch.stack([x, y], dim=-1)
        values.append(xy.mean(dim=0))
        values.append(xy[3])

    vals = torch.stack(values, dim=0)

    if use_start_as_center:
        xy_center = torch.zeros(2, dtype=torch.float32)
    else:
        xy_center = vals.mean(dim=0)

    distances = torch.norm(vals - xy_center, dim=-1)

    # Use 95th percentile so a few far/outlier points do not dominate scaling.
    radius_scale = torch.quantile(distances, 0.95).clamp_min(1.0)

    return xy_center.float(), radius_scale.float()


def normalize_xy(xy: torch.Tensor, xy_center: torch.Tensor, radius_scale: torch.Tensor):
    return (xy - xy_center) / radius_scale


def denormalize_xy(xy_norm: torch.Tensor, xy_center: torch.Tensor, radius_scale: torch.Tensor):
    return xy_norm * radius_scale + xy_center


# -----------------------------
# Dataset
# -----------------------------
class GSOSlidingWindowDataset(Dataset):
    """
    One continuous sequence in one folder -> sliding window dataset.

    Sample i returns frames [i, i+1, i+2, i+3].
    Returns normalized XY labels using center/radius normalization.
    """
    def __init__(
        self,
        frames: List[Tuple[str, int, float, float, float]],
        cfg: CFG,
        start_idx: int,
        end_idx: int,
        lat0: Optional[float] = None,
        lon0: Optional[float] = None,
        xy_center: Optional[torch.Tensor] = None,
        radius_scale: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.frames = frames
        self.cfg = cfg
        self.start_idx = start_idx
        self.end_idx = end_idx

        if lat0 is None or lon0 is None:
            lat0 = frames[0][2]
            lon0 = frames[0][3]
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)

        if xy_center is None or radius_scale is None:
            raise ValueError("xy_center and radius_scale must be provided.")

        self.xy_center = xy_center.float()
        self.radius_scale = radius_scale.float()

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

        chosen = [self.frames[idx + k * stride] for k in range(Tframes)]
        paths = [c[0] for c in chosen]

        imgs = torch.stack([self._load_img(p) for p in paths], dim=0)

        lat = torch.tensor([c[2] for c in chosen], dtype=torch.float32)
        lon = torch.tensor([c[3] for c in chosen], dtype=torch.float32)
        head = heading_to_rad(torch.tensor([c[4] for c in chosen], dtype=torch.float32))

        x, y = ll_to_local_xy(lat, lon, self.lat0, self.lon0)
        xy = torch.stack([x, y], dim=-1)

        gt_overall_xy = normalize_xy(xy.mean(dim=0), self.xy_center, self.radius_scale)
        gt_xy4 = normalize_xy(xy[3], self.xy_center, self.radius_scale)

        return {
            "images": imgs,
            "gt_overall_xy": gt_overall_xy,
            "gt_xy4": gt_xy4,
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


# -----------------------------
# Model: original architecture style
# -----------------------------
class SpatioTemporalTransformer(nn.Module):
    def __init__(self, cfg: CFG, embed_dim: int = 256, nhead: int = 8, layers: int = 4, dropout: float = 0.1, backbone="vgg"):
        super().__init__()
        self.cfg = cfg
        self.T = cfg.T_frames
        backbone = backbone.lower()

        if backbone == "vgg":
            weights = VGG16_Weights.IMAGENET1K_V1 if cfg.use_pretrained else None
            base = vgg16(weights=weights)
            self.backbone = base.features
            self.in_channels = 512

        elif backbone == "resnet":
            weights = ResNet18_Weights.IMAGENET1K_V1 if cfg.use_pretrained else None
            base = resnet18(weights=weights)
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            self.in_channels = base.fc.in_features

        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use 'vgg' or 'resnet'.")

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
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
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
            nn.Linear(512, 4),  # x4, y4, sin, cos
        )

    def _frame_tokens(self, images_btchw: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(images_btchw)
        _, _, Hf, Wf = feat.shape
        if (Hf, Wf) != (self.h, self.w):
            self.h, self.w = Hf, Wf
            new_L = Hf * Wf
            if new_L != self.L:
                self.L = new_L
                self.space_emb = nn.Embedding(self.L, self.space_emb.embedding_dim).to(feat.device)

        feat = feat.flatten(2).transpose(1, 2)
        tok = self.proj(feat)
        return tok

    def _add_positional(self, tok: torch.Tensor, Tctx: int) -> torch.Tensor:
        _, TL, _ = tok.shape
        L = TL // Tctx
        t_idx = torch.arange(Tctx, device=tok.device).repeat_interleave(L)
        s_idx = torch.arange(L, device=tok.device).repeat(Tctx)

        if Tctx == 4:
            tok = tok + self.time_emb_4(t_idx)[None, :, :] + self.space_emb(s_idx)[None, :, :]
        else:
            tok = tok + self.time_emb_3(t_idx)[None, :, :] + self.space_emb(s_idx)[None, :, :]
        return tok

    def forward(self, images: torch.Tensor):
        B, Tn, C, H, W = images.shape
        assert Tn == 4, "Expected exactly 4 input frames."

        x = images.view(B * Tn, C, H, W)
        tok_bt = self._frame_tokens(x)
        L = tok_bt.size(1)

        # Pass A: 4 frames -> overall/window XY
        tok_all = tok_bt.view(B, Tn * L, -1)
        tok_all = self._add_positional(tok_all, Tctx=4)
        seq_all = torch.cat([self.cls.expand(B, -1, -1), tok_all], dim=1)
        out_all = self.encoder(seq_all)
        overall_xy = self.head_overall(out_all[:, 0])

        # Pass B: first 3 frames -> frame-4 pose
        Tctx = 3
        tok_ctx = tok_bt.view(B, Tn, L, -1)[:, :Tctx]
        tok_ctx = tok_ctx.reshape(B, Tctx * L, -1)
        tok_ctx = self._add_positional(tok_ctx, Tctx=3)
        seq_ctx = torch.cat([self.cls.expand(B, -1, -1), tok_ctx], dim=1)
        out_ctx = self.encoder(seq_ctx)

        pred = self.head_frame4(out_ctx[:, 0])
        pred_xy4 = pred[:, :2]
        pred_sincos4 = F.normalize(pred[:, 2:], dim=-1)

        tok_frames = tok_bt.view(B, Tn, L, -1)
        z4 = F.normalize(tok_frames[:, 3].mean(dim=1), dim=-1)
        zpos = F.normalize(tok_frames[:, :3].mean(dim=(1, 2)), dim=-1)

        return {
            "overall_xy": overall_xy,
            "pred_xy4": pred_xy4,
            "pred_sincos4": pred_sincos4,
            "z4": z4,
            "zpos": zpos,
        }


# -----------------------------
# Losses
# -----------------------------
def huber(pred: torch.Tensor, gt: torch.Tensor, delta: float = 1.0):
    return F.smooth_l1_loss(pred, gt, beta=delta)


def heading_loss_sincos(pred_sincos: torch.Tensor, gt_heading_rad: torch.Tensor):
    gt = torch.stack([torch.sin(gt_heading_rad), torch.cos(gt_heading_rad)], dim=-1)
    gt = F.normalize(gt, dim=-1)
    cos_sim = (pred_sincos * gt).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cos_sim).mean()


def info_nce(z_anchor: torch.Tensor, z_pos: torch.Tensor, temperature: float = 0.07):
    logits = (z_anchor @ z_pos.t()) / temperature
    labels = torch.arange(z_anchor.size(0), device=z_anchor.device)
    return F.cross_entropy(logits, labels)


def total_loss(out: dict, batch: dict, cfg: CFG):
    l_overall = huber(out["overall_xy"], batch["gt_overall_xy"])
    l_xy4 = huber(out["pred_xy4"], batch["gt_xy4"])
    l_head = heading_loss_sincos(out["pred_sincos4"], batch["gt_heading4"])

    loss = cfg.w_overall_xy * l_overall + cfg.w_xy4 * l_xy4 + cfg.w_head * l_head

    logs = {
        "l_overall": l_overall.detach(),
        "l_xy4": l_xy4.detach(),
        "l_head": l_head.detach(),
    }

    if cfg.use_infonce:
        l_nce = info_nce(out["z4"], out["zpos"], temperature=cfg.temperature)
        loss = loss + cfg.w_nce * l_nce
        logs["l_nce"] = l_nce.detach()

    return loss, logs


# -----------------------------
# Evaluation and metrics
# -----------------------------
def move_tensor_batch_to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


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
def evaluate_meters(model, loader, device, cfg: CFG, xy_center, radius_scale, save_rows=False):
    model.eval()

    xy_center = xy_center.to(device)
    radius_scale = radius_scale.to(device)

    overall_errors = []
    frame4_errors = []
    heading_errors_deg = []
    rows = []

    for batch in loader:
        batch = move_tensor_batch_to_device(batch, device)
        out = model(batch["images"])

        pred_overall_m = denormalize_xy(out["overall_xy"], xy_center, radius_scale)
        gt_overall_m = denormalize_xy(batch["gt_overall_xy"], xy_center, radius_scale)

        pred_xy4_m = denormalize_xy(out["pred_xy4"], xy_center, radius_scale)
        gt_xy4_m = denormalize_xy(batch["gt_xy4"], xy_center, radius_scale)

        overall_err = torch.norm(pred_overall_m - gt_overall_m, dim=-1)
        frame4_err = torch.norm(pred_xy4_m - gt_xy4_m, dim=-1)

        pred_heading = torch.atan2(out["pred_sincos4"][:, 0], out["pred_sincos4"][:, 1])
        gt_heading = batch["gt_heading4"]
        heading_diff = torch.atan2(torch.sin(pred_heading - gt_heading), torch.cos(pred_heading - gt_heading))
        heading_err_deg = torch.rad2deg(torch.abs(heading_diff))

        overall_errors.append(overall_err.cpu())
        frame4_errors.append(frame4_err.cpu())
        heading_errors_deg.append(heading_err_deg.cpu())

        if save_rows:
            B = batch["images"].shape[0]
            for i in range(B):
                frame_ids = batch["frame_ids"][i].detach().cpu().tolist()
                rows.append({
                    "window_start_idx": int(batch["window_start_idx"][i].item()),
                    "frame1_id": int(frame_ids[0]),
                    "frame2_id": int(frame_ids[1]),
                    "frame3_id": int(frame_ids[2]),
                    "frame4_id": int(frame_ids[3]),
                    "overall_error_m": float(overall_err[i].item()),
                    "frame4_error_m": float(frame4_err[i].item()),
                    "heading_error_deg": float(heading_err_deg[i].item()),
                    "gt_overall_x_m": float(gt_overall_m[i, 0].item()),
                    "gt_overall_y_m": float(gt_overall_m[i, 1].item()),
                    "pred_overall_x_m": float(pred_overall_m[i, 0].item()),
                    "pred_overall_y_m": float(pred_overall_m[i, 1].item()),
                    "gt_frame4_x_m": float(gt_xy4_m[i, 0].item()),
                    "gt_frame4_y_m": float(gt_xy4_m[i, 1].item()),
                    "pred_frame4_x_m": float(pred_xy4_m[i, 0].item()),
                    "pred_frame4_y_m": float(pred_xy4_m[i, 1].item()),
                })

    overall_errors = torch.cat(overall_errors)
    frame4_errors = torch.cat(frame4_errors)
    heading_errors_deg = torch.cat(heading_errors_deg)

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

    model.train()
    return metrics, rows


def save_metrics_csv(metrics, output_dir):
    path = os.path.join(output_dir, "final_meter_metrics.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, f"{float(v):.6f}"])
    return path


def save_rows_csv(rows, output_dir):
    path = os.path.join(output_dir, "test_per_sample_errors.csv")
    if not rows:
        return path
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


# -----------------------------
# Graphs
# -----------------------------
def read_loss_log(path):
    epochs, train_losses, eval_losses = [], [], []
    if not os.path.isfile(path):
        return epochs, train_losses, eval_losses
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["Epoch"]))
            train_losses.append(float(row["train_loss"]))
            eval_losses.append(float(row["val_loss"]))
    return epochs, train_losses, eval_losses


def plot_loss_curve(loss_log_path, output_dir):
    if plt is None:
        return None
    epochs, train_losses, eval_losses = read_loss_log(loss_log_path)
    if not epochs:
        return None
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    ax.plot(epochs, train_losses, marker="o", linewidth=2, label="Train loss")
    ax.plot(epochs, eval_losses, marker="o", linewidth=2, label="Validation loss")
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


def plot_meter_error_summary(metrics, output_dir):
    if plt is None:
        return None
    labels = ["Overall\nMean", "Overall\nMedian", "Overall\nRMSE", "Frame-4\nMean", "Frame-4\nMedian", "Frame-4\nRMSE"]
    values = [metrics["overall_mean_m"], metrics["overall_median_m"], metrics["overall_rmse_m"],
              metrics["frame4_mean_m"], metrics["frame4_median_m"], metrics["frame4_rmse_m"]]
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


def rows_to_series(rows, key):
    return [float(r[key]) for r in rows]


def plot_error_histogram(rows, output_dir):
    if plt is None or not rows:
        return None
    overall_err = rows_to_series(rows, "overall_error_m")
    frame4_err = rows_to_series(rows, "frame4_error_m")
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    ax.hist(overall_err, bins=40, alpha=0.55, label="Overall/window")
    ax.hist(frame4_err, bins=40, alpha=0.55, label="Frame-4")
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


def plot_trajectory(rows, output_dir):
    if plt is None or not rows:
        return None
    gt_x = rows_to_series(rows, "gt_frame4_x_m")
    gt_y = rows_to_series(rows, "gt_frame4_y_m")
    pr_x = rows_to_series(rows, "pred_frame4_x_m")
    pr_y = rows_to_series(rows, "pred_frame4_y_m")
    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    ax.plot(gt_x, gt_y, linewidth=2, label="Ground truth")
    ax.plot(pr_x, pr_y, linewidth=2, linestyle="--", label="Predicted")
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


def plot_heading_summary(metrics, output_dir):
    if plt is None:
        return None
    labels = ["Mean", "Median", "RMSE", "90th %ile"]
    values = [metrics["heading_mean_deg"], metrics["heading_median_deg"], metrics["heading_rmse_deg"], metrics["heading_p90_deg"]]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
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


def save_graphs(loss_log_path, metrics, rows, output_dir):
    graph_paths = []
    for p in [
        plot_loss_curve(loss_log_path, output_dir),
        plot_meter_error_summary(metrics, output_dir),
        plot_error_histogram(rows, output_dir),
        plot_trajectory(rows, output_dir),
        plot_heading_summary(metrics, output_dir),
    ]:
        if p is not None:
            graph_paths.append(p)
    return graph_paths


# -----------------------------
# Train util
# -----------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=r"C:\Users\admin\Desktop\spiral_avgso\maps_screenshots_spiral_28150")
    ap.add_argument("--output_dir", type=str, default="results_center_radius")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--use_infonce", type=int, default=0)
    ap.add_argument("--use_pretrained", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_start_as_center", type=int, default=1)
    args = ap.parse_args()

    cfg = CFG(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_infonce=bool(args.use_infonce),
        use_pretrained=bool(args.use_pretrained),
        num_workers=args.num_workers,
        lr=args.lr,
        dropout=args.dropout,
    )

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    output_dir = args.output_dir
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    frames = list_frames_one_folder(cfg.data_root)
    total_frames = len(frames)

    num_windows = total_frames - (cfg.T_frames - 1) * cfg.stride
    if num_windows <= 0:
        raise RuntimeError(f"Not enough frames for windows. frames={total_frames}, need >= {cfg.T_frames}")

    (tr_s, tr_e), (ev_s, ev_e), (te_s, te_e) = chronological_split(
        num_windows, cfg.train_frac, cfg.eval_frac, cfg.test_frac
    )

    # Use the first frame as local origin.
    # For your spiral design, this is the start/center.
    lat0 = float(frames[0][2])
    lon0 = float(frames[0][3])

    xy_center, radius_scale = compute_center_radius_normalization(
        frames,
        cfg,
        tr_s,
        tr_e,
        lat0,
        lon0,
        use_start_as_center=bool(args.use_start_as_center),
    )

    train_ds = GSOSlidingWindowDataset(frames, cfg, tr_s, tr_e, lat0=lat0, lon0=lon0,
                                       xy_center=xy_center, radius_scale=radius_scale)
    eval_ds = GSOSlidingWindowDataset(frames, cfg, ev_s, ev_e, lat0=lat0, lon0=lon0,
                                      xy_center=xy_center, radius_scale=radius_scale)
    test_ds = GSOSlidingWindowDataset(frames, cfg, te_s, te_e, lat0=lat0, lon0=lon0,
                                      xy_center=xy_center, radius_scale=radius_scale)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    model = SpatioTemporalTransformer(
        cfg,
        embed_dim=256,
        nhead=8,
        layers=4,
        dropout=cfg.dropout,
        backbone=BACKBONE,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best = 1e9
    best_epoch = -1
    best_path = os.path.join(checkpoint_dir, "best.pt")
    loss_log_path = os.path.join(output_dir, "loss_log.csv")

    print(f"Data root: {cfg.data_root}")
    print(f"Frames: {total_frames} | Windows: {num_windows}")
    print(f"Split windows: train={len(train_ds)} eval={len(eval_ds)} test={len(test_ds)} (chronological)")
    print(f"Use InfoNCE: {cfg.use_infonce} (w={cfg.w_nce}, T={cfg.temperature})")
    print("Normalization: center/radius")
    print(f"Local origin lat0/lon0: {lat0}, {lon0}")
    print(f"XY center meters: {xy_center.tolist()}")
    print(f"Radius scale meters: {float(radius_scale.item()):.6f}")
    print(f"Output dir: {output_dir}")
    print(f"Saving best model to: {best_path}")

    with open(loss_log_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", "train_loss", "val_loss"])

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running = 0.0
            seen = 0

            for batch in train_loader:
                batch = move_tensor_batch_to_device(batch, device)

                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    out = model(batch["images"])
                    loss, logs = total_loss(out, batch, cfg)

                if not torch.isfinite(loss):
                    print("NaN/Inf loss detected")
                    print("gt_overall_xy finite:", torch.isfinite(batch["gt_overall_xy"]).all().item())
                    print("gt_xy4 finite:", torch.isfinite(batch["gt_xy4"]).all().item())
                    print("gt_heading4 finite:", torch.isfinite(batch["gt_heading4"]).all().item())
                    raise RuntimeError("Stopping because loss became NaN/Inf")

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                running += loss.item() * batch["images"].size(0)
                seen += batch["images"].size(0)
                print(f"{running:.2f}    |    {loss.item():.4f}    |    {seen}    |    {(running / max(seen, 1)):.4f}", end="\r")

            train_loss = running / max(seen, 1)
            eval_loss = evaluate(model, eval_loader, device, cfg)
            print(f"Epoch {epoch:03d} | train={train_loss:.4f} | eval={eval_loss:.4f}")
            writer.writerow([f"{epoch:03d}", f"{train_loss:.6f}", f"{eval_loss:.6f}"])

            if eval_loss < best:
                best = eval_loss
                best_epoch = epoch
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": cfg.__dict__,
                        "normalization": "center_radius",
                        "xy_center": xy_center,
                        "radius_scale": radius_scale,
                        "lat0": lat0,
                        "lon0": lon0,
                        "use_start_as_center": bool(args.use_start_as_center),
                        "best_eval": best,
                        "best_epoch": best_epoch,
                    },
                    best_path,
                )

    # Reload best checkpoint before final test.
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    test_loss = evaluate(model, test_loader, device, cfg)
    metrics, rows = evaluate_meters(model, test_loader, device, cfg, xy_center, radius_scale, save_rows=True)

    metrics_csv = save_metrics_csv(metrics, output_dir)
    rows_csv = save_rows_csv(rows, output_dir)
    graph_paths = save_graphs(loss_log_path, metrics, rows, output_dir)

    print(f"Best eval normalized loss: {best:.4f} at epoch {best_epoch}")
    print(f"Test normalized loss: {test_loss:.4f}")
    print("Test meter errors:")
    print(f"  Overall/window mean error:   {metrics['overall_mean_m']:.2f} m")
    print(f"  Overall/window median error: {metrics['overall_median_m']:.2f} m")
    print(f"  Overall/window RMSE:         {metrics['overall_rmse_m']:.2f} m")
    print(f"  Overall/window 90th %ile:    {metrics['overall_p90_m']:.2f} m")
    print(f"  Frame-4 mean error:          {metrics['frame4_mean_m']:.2f} m")
    print(f"  Frame-4 median error:        {metrics['frame4_median_m']:.2f} m")
    print(f"  Frame-4 RMSE:                {metrics['frame4_rmse_m']:.2f} m")
    print(f"  Frame-4 90th %ile:           {metrics['frame4_p90_m']:.2f} m")
    print("Test heading errors:")
    print(f"  Heading mean error:          {metrics['heading_mean_deg']:.2f} deg")
    print(f"  Heading median error:        {metrics['heading_median_deg']:.2f} deg")
    print(f"  Heading RMSE:                {metrics['heading_rmse_deg']:.2f} deg")
    print(f"  Heading 90th %ile:           {metrics['heading_p90_deg']:.2f} deg")
    print(f"Saved metrics CSV: {metrics_csv}")
    print(f"Saved per-sample error CSV: {rows_csv}")
    if graph_paths:
        print("Saved graphs:")
        for p in graph_paths:
            print(f"  {p}")
    else:
        print("Graphs were not saved because matplotlib is unavailable.")
    print("Done.")


if __name__ == "__main__":
    main()
