# -*- coding: utf-8 -*-
r"""
External line-trajectory generalization test for the AVGSO/GSO multiframe localization model.

This script:
1. Loads a V4 center/radius checkpoint.
2. Reuses the SAME XY center/radius normalization saved in the checkpoint.
3. Tests an unseen line trajectory folder without retraining.
4. Saves meter-based metrics, CSV files, a text report, and all requested graphs.

Default layout:
  Base folder:
    C:\Users\admin\Desktop\spiral_avgso

  Checkpoint:
    C:\Users\admin\Desktop\spiral_avgso\localization_lawnmower_vgg\checkpoints\best.pt

  Training data used for normalization:
    C:\Users\admin\Desktop\spiral_avgso\maps_screenshots_Lawn_31668

  External line test folder:
    C:\Users\admin\Desktop\spiral_avgso\maps_screenshots_line

  Output folder:
    C:\Users\admin\Desktop\spiral_avgso\localization_lawnmower_vgg\line_generalization_results

Run:
  cd C:\Users\admin\Desktop\spiral_avgso
  python .\test_line_trajectory_generalization.py

Optional explicit run:
  python .\test_line_trajectory_generalization.py ^
    --checkpoint "C:\Users\admin\Desktop\spiral_avgso\localization_lawnmower_vgg\checkpoints\best.pt" ^
    --train_data_root "C:\Users\admin\Desktop\spiral_avgso\maps_screenshots_Lawn_31668" ^
    --line_data_root "C:\Users\admin\Desktop\spiral_avgso\maps_screenshots_line" ^
    --output_dir "C:\Users\admin\Desktop\spiral_avgso\localization_lawnmower_vgg\line_generalization_results"
"""

import os
import csv
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

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
# Config: keep architecture-compatible with training code
# -----------------------------
@dataclass
class CFG:
    data_root: str
    image_size: int = 224
    T_frames: int = 4
    stride: int = 1

    train_frac: float = 0.80
    eval_frac: float = 0.10
    test_frac: float = 0.10

    seed: int = 42
    batch_size: int = 4
    num_workers: int = 0
    epochs: int = 20
    lr: float = 5e-5
    weight_decay: float = 0.05

    w_overall_xy: float = 1.0
    w_xy4: float = 1.0
    w_head: float = 0.05

    use_infonce: bool = False
    w_nce: float = 0.15
    temperature: float = 0.07

    use_pretrained: bool = True


BACKBONE = "vgg"  # must match the checkpoint you trained
EARTH_R = 6378137.0
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


# -----------------------------
# Geo helpers
# -----------------------------
def ll_to_local_xy(lat_deg: torch.Tensor, lon_deg: torch.Tensor, lat0_deg: float, lon0_deg: float):
    """Local tangent approximation: x=east, y=north, in meters."""
    lat = torch.deg2rad(lat_deg)
    lon = torch.deg2rad(lon_deg)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    x = (lon - lon0) * math.cos(lat0) * EARTH_R
    y = (lat - lat0) * EARTH_R
    return x, y


def heading_to_rad(h: torch.Tensor):
    """If heading looks like degrees, convert to radians."""
    if torch.max(torch.abs(h)).item() > 6.5:
        return torch.deg2rad(h)
    return h


def wrap_heading_error_deg(pred_deg, gt_deg):
    """Smallest absolute angular difference in degrees."""
    diff = (pred_deg - gt_deg + 180.0) % 360.0 - 180.0
    return abs(diff)


# -----------------------------
# File parsing
# -----------------------------
def parse_frame_name(fname: str) -> Optional[Tuple[int, float, float, float]]:
    """
    Expected filename:
      frame_<FRAMEID>_<LAT>_<LON>_<HEADING>.<ext>
    Example:
      frame_000001_36.045944_-79.960988_90.jpg
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    if len(parts) < 5 or parts[0].lower() != "frame":
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
            f"Found only {len(items)} valid frames in {data_root}. Need at least 4. "
            "Expected filename format: frame_<id>_<lat>_<lon>_<heading>.<jpg/png/webp>"
        )
    return items


def chronological_split(num_windows: int, train_frac: float, eval_frac: float, test_frac: float):
    assert abs(train_frac + eval_frac + test_frac - 1.0) < 1e-6
    n_train = int(num_windows * train_frac)
    n_eval = int(num_windows * eval_frac)
    tr = (0, n_train)
    ev = (n_train, n_train + n_eval)
    te = (n_train + n_eval, num_windows)
    return tr, ev, te


def compute_center_radius_normalization(frames, cfg: CFG, start_idx: int, end_idx: int, lat0: float, lon0: float):
    """Fallback rebuild of V4 center/radius normalization from training windows."""
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
    xy_center = torch.zeros(2, dtype=torch.float32)
    distances = torch.norm(vals - xy_center, dim=-1)
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
    def __init__(self, frames, cfg: CFG, start_idx: int, end_idx: int, lat0: float, lon0: float, xy_center, radius_scale):
        self.frames = frames
        self.cfg = cfg
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self.xy_center = xy_center.float()
        self.radius_scale = radius_scale.float()

        self.tf = T.Compose([
            T.Resize((cfg.image_size, cfg.image_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return max(0, self.end_idx - self.start_idx)

    def _load_img(self, path):
        return self.tf(Image.open(path).convert("RGB"))

    def __getitem__(self, i):
        idx = self.start_idx + i
        chosen = [self.frames[idx + k * self.cfg.stride] for k in range(self.cfg.T_frames)]
        paths = [c[0] for c in chosen]
        images = torch.stack([self._load_img(p) for p in paths], dim=0)

        lat = torch.tensor([c[2] for c in chosen], dtype=torch.float32)
        lon = torch.tensor([c[3] for c in chosen], dtype=torch.float32)
        head = heading_to_rad(torch.tensor([c[4] for c in chosen], dtype=torch.float32))

        x, y = ll_to_local_xy(lat, lon, self.lat0, self.lon0)
        xy = torch.stack([x, y], dim=-1)

        gt_overall_xy = normalize_xy(xy.mean(dim=0), self.xy_center, self.radius_scale)
        gt_xy4 = normalize_xy(xy[3], self.xy_center, self.radius_scale)

        return {
            "images": images,
            "gt_overall_xy": gt_overall_xy,
            "gt_xy4": gt_xy4,
            "gt_heading4": head[3],
            "window_start_idx": torch.tensor(idx, dtype=torch.long),
            "frame_ids": torch.tensor([c[1] for c in chosen], dtype=torch.long),
            "paths": paths,
        }


# -----------------------------
# Model: same architecture as training code
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
            raise ValueError(f"Unsupported backbone: {backbone}")

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
            nn.Linear(512, 4),  # x, y, sin, cos
        )

    def _frame_tokens(self, images_btchw):
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

    def _add_positional(self, tok, Tctx: int):
        _, TL, _ = tok.shape
        L = TL // Tctx

        t_idx = torch.arange(Tctx, device=tok.device).repeat_interleave(L)
        s_idx = torch.arange(L, device=tok.device).repeat(Tctx)

        if Tctx == 4:
            return tok + self.time_emb_4(t_idx)[None, :, :] + self.space_emb(s_idx)[None, :, :]
        return tok + self.time_emb_3(t_idx)[None, :, :] + self.space_emb(s_idx)[None, :, :]

    def forward(self, images):
        B, Tn, C, H, W = images.shape
        assert Tn == 4, "Expected exactly 4 input frames."

        x = images.view(B * Tn, C, H, W)
        tok_bt = self._frame_tokens(x)
        L = tok_bt.size(1)

        # Pass A: all 4 frames -> overall/window coordinate
        tok_all = tok_bt.view(B, Tn * L, -1)
        tok_all = self._add_positional(tok_all, Tctx=4)
        seq_all = torch.cat([self.cls.expand(B, -1, -1), tok_all], dim=1)
        out_all = self.encoder(seq_all)
        overall_xy = self.head_overall(out_all[:, 0])

        # Pass B: first 3 frames -> frame-4 coordinate + heading
        Tctx = 3
        tok_ctx = tok_bt.view(B, Tn, L, -1)[:, :Tctx]
        tok_ctx = tok_ctx.reshape(B, Tctx * L, -1)
        tok_ctx = self._add_positional(tok_ctx, Tctx=3)
        seq_ctx = torch.cat([self.cls.expand(B, -1, -1), tok_ctx], dim=1)
        out_ctx = self.encoder(seq_ctx)

        pred = self.head_frame4(out_ctx[:, 0])
        return {
            "overall_xy": overall_xy,
            "pred_xy4": pred[:, :2],
            "pred_sincos4": F.normalize(pred[:, 2:], dim=-1),
        }


# -----------------------------
# Loss and evaluation
# -----------------------------
def heading_loss_sincos(pred_sincos, gt_heading_rad):
    gt = torch.stack([torch.sin(gt_heading_rad), torch.cos(gt_heading_rad)], dim=-1)
    gt = F.normalize(gt, dim=-1)
    cos_sim = (pred_sincos * gt).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cos_sim).mean()


def total_loss(out, batch, cfg: CFG):
    l_overall = F.smooth_l1_loss(out["overall_xy"], batch["gt_overall_xy"], beta=1.0)
    l_xy4 = F.smooth_l1_loss(out["pred_xy4"], batch["gt_xy4"], beta=1.0)
    l_head = heading_loss_sincos(out["pred_sincos4"], batch["gt_heading4"])
    return cfg.w_overall_xy * l_overall + cfg.w_xy4 * l_xy4 + cfg.w_head * l_head


def move_tensor_batch_to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def extract_paths_from_collated(paths_obj, batch_index: int, t_index: int = -1):
    """Handle PyTorch default collation for list[str] metadata."""
    try:
        # Common default collate: list length T, each item tuple/list length B
        return paths_obj[t_index][batch_index]
    except Exception:
        try:
            # Alternative shape: list length B, each item list length T
            return paths_obj[batch_index][t_index]
        except Exception:
            return ""


@torch.no_grad()
def evaluate_external_line(model, loader, device, cfg: CFG, xy_center, radius_scale):
    model.eval()
    xy_center = xy_center.to(device)
    radius_scale = radius_scale.to(device)

    rows = []
    overall_errors = []
    frame4_errors = []
    heading_errors_deg = []
    losses = []

    for batch in loader:
        batch = move_tensor_batch_to_device(batch, device)
        out = model(batch["images"])
        loss = total_loss(out, batch, cfg)
        losses.append(float(loss.item()))

        pred_overall_m = denormalize_xy(out["overall_xy"], xy_center, radius_scale)
        gt_overall_m = denormalize_xy(batch["gt_overall_xy"], xy_center, radius_scale)

        pred_xy4_m = denormalize_xy(out["pred_xy4"], xy_center, radius_scale)
        gt_xy4_m = denormalize_xy(batch["gt_xy4"], xy_center, radius_scale)

        overall_err = torch.norm(pred_overall_m - gt_overall_m, dim=-1)
        frame4_err = torch.norm(pred_xy4_m - gt_xy4_m, dim=-1)

        pred_heading_rad = torch.atan2(out["pred_sincos4"][:, 0], out["pred_sincos4"][:, 1])
        gt_heading_rad = batch["gt_heading4"]
        heading_diff = torch.atan2(
            torch.sin(pred_heading_rad - gt_heading_rad),
            torch.cos(pred_heading_rad - gt_heading_rad),
        )
        heading_err_deg = torch.rad2deg(torch.abs(heading_diff))

        pred_heading_deg = torch.rad2deg(pred_heading_rad) % 360
        gt_heading_deg = torch.rad2deg(gt_heading_rad) % 360

        overall_errors.append(overall_err.detach().cpu())
        frame4_errors.append(frame4_err.detach().cpu())
        heading_errors_deg.append(heading_err_deg.detach().cpu())

        B = batch["images"].shape[0]
        for i in range(B):
            frame_ids = batch["frame_ids"][i].detach().cpu().tolist()
            rows.append({
                "window_start_idx": int(batch["window_start_idx"][i].item()),
                "frame1_id": int(frame_ids[0]),
                "frame2_id": int(frame_ids[1]),
                "frame3_id": int(frame_ids[2]),
                "frame4_id": int(frame_ids[3]),
                "frame4_path": extract_paths_from_collated(batch.get("paths", []), i, -1),

                "normalized_loss_batch": float(loss.item()),

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

                "gt_heading_deg": float(gt_heading_deg[i].item()),
                "pred_heading_deg": float(pred_heading_deg[i].item()),
            })

    overall_errors = torch.cat(overall_errors)
    frame4_errors = torch.cat(frame4_errors)
    heading_errors_deg = torch.cat(heading_errors_deg)

    metrics = {
        "num_line_frames": None,
        "num_line_windows": int(frame4_errors.numel()),
        "mean_normalized_loss": float(sum(losses) / max(len(losses), 1)),

        "overall_mean_m": overall_errors.mean().item(),
        "overall_median_m": overall_errors.median().item(),
        "overall_rmse_m": torch.sqrt((overall_errors ** 2).mean()).item(),
        "overall_p90_m": torch.quantile(overall_errors, 0.90).item(),
        "overall_min_m": overall_errors.min().item(),
        "overall_max_m": overall_errors.max().item(),

        "frame4_mean_m": frame4_errors.mean().item(),
        "frame4_median_m": frame4_errors.median().item(),
        "frame4_rmse_m": torch.sqrt((frame4_errors ** 2).mean()).item(),
        "frame4_p90_m": torch.quantile(frame4_errors, 0.90).item(),
        "frame4_min_m": frame4_errors.min().item(),
        "frame4_max_m": frame4_errors.max().item(),

        "heading_mean_deg": heading_errors_deg.mean().item(),
        "heading_median_deg": heading_errors_deg.median().item(),
        "heading_rmse_deg": torch.sqrt((heading_errors_deg ** 2).mean()).item(),
        "heading_p90_deg": torch.quantile(heading_errors_deg, 0.90).item(),
        "heading_min_deg": heading_errors_deg.min().item(),
        "heading_max_deg": heading_errors_deg.max().item(),
    }

    model.train()
    return metrics, rows


# -----------------------------
# Saving outputs
# -----------------------------
def save_metrics_csv(metrics, output_dir):
    path = os.path.join(output_dir, "line_generalization_metrics.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v if v is None else f"{float(v):.6f}"])
    return path


def save_predictions_csv(rows, output_dir):
    path = os.path.join(output_dir, "line_per_window_predictions.csv")
    if not rows:
        return path
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_summary_report(metrics, output_dir, checkpoint, train_data_root, line_data_root, xy_center, radius_scale):
    path = os.path.join(output_dir, "line_generalization_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Line Trajectory Generalization Test\n")
        f.write("===================================\n\n")
        f.write(f"Checkpoint: {checkpoint}\n")
        f.write(f"Training data root used for normalization: {train_data_root}\n")
        f.write(f"External line trajectory root: {line_data_root}\n")
        f.write("Normalization: center/radius\n")
        f.write(f"Training XY center: {xy_center.tolist()}\n")
        f.write(f"Training radius scale: {float(radius_scale.item())}\n\n")
        f.write("Interpretation note:\n")
        f.write("This script does not retrain on the line trajectory. It tests whether the trained model generalizes to an unseen straight-line route.\n")
        f.write("Meter errors are computed after converting normalized predictions back to the original local meter coordinate system.\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    return path


def rows_to_series(rows, key):
    return [float(r[key]) for r in rows]


# -----------------------------
# Graphs: previous + requested new plots
# -----------------------------
def plot_metric_summary(metrics, output_dir):
    if plt is None:
        return None

    labels = [
        "Overall\nMean", "Overall\nMedian", "Overall\nRMSE",
        "Frame-4\nMean", "Frame-4\nMedian", "Frame-4\nRMSE",
    ]
    values = [
        metrics["overall_mean_m"], metrics["overall_median_m"], metrics["overall_rmse_m"],
        metrics["frame4_mean_m"], metrics["frame4_median_m"], metrics["frame4_rmse_m"],
    ]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    ax.bar(labels, values)
    ax.set_title("External Line Trajectory Localization Error")
    ax.set_ylabel("Error (meters)")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()

    path = os.path.join(output_dir, "line_meter_error_summary.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_trajectory(rows, output_dir):
    if plt is None:
        return None

    gt_x = rows_to_series(rows, "gt_frame4_x_m")
    gt_y = rows_to_series(rows, "gt_frame4_y_m")
    pr_x = rows_to_series(rows, "pred_frame4_x_m")
    pr_y = rows_to_series(rows, "pred_frame4_y_m")

    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    ax.plot(gt_x, gt_y, marker="o", linewidth=2, label="Ground truth line")
    ax.plot(pr_x, pr_y, marker="o", linewidth=2, linestyle="--", label="Predicted")
    ax.set_title("Line Trajectory: Ground Truth vs Predicted Frame-4 Coordinates")
    ax.set_xlabel("Local X / East (meters)")
    ax.set_ylabel("Local Y / North (meters)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, "line_trajectory_prediction_vs_ground_truth.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_error_over_time(rows, output_dir):
    if plt is None:
        return None

    ids = [int(r["frame4_id"]) for r in rows]
    frame4_err = rows_to_series(rows, "frame4_error_m")
    overall_err = rows_to_series(rows, "overall_error_m")

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    ax.plot(ids, frame4_err, marker="o", linewidth=2, label="Frame-4 error")
    ax.plot(ids, overall_err, marker="o", linewidth=2, linestyle="--", label="Overall/window error")
    ax.set_title("Line Trajectory Localization Error Over Time")
    ax.set_xlabel("Frame-4 ID")
    ax.set_ylabel("Error (meters)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, "line_error_over_time.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_heading_error_over_time(rows, output_dir):
    if plt is None:
        return None

    ids = [int(r["frame4_id"]) for r in rows]
    err = rows_to_series(rows, "heading_error_deg")

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    ax.plot(ids, err, marker="o", linewidth=2)
    ax.set_title("Line Trajectory Heading Error Over Time")
    ax.set_xlabel("Frame-4 ID")
    ax.set_ylabel("Heading error (degrees)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "line_heading_error_over_time.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_error_histogram(rows, output_dir):
    if plt is None:
        return None

    frame4_err = rows_to_series(rows, "frame4_error_m")
    overall_err = rows_to_series(rows, "overall_error_m")
    bins = min(20, max(5, len(rows) // 2))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.hist(overall_err, bins=bins, alpha=0.55, label="Overall/window")
    ax.hist(frame4_err, bins=bins, alpha=0.55, label="Frame-4")
    ax.set_title("Line Trajectory Error Distribution")
    ax.set_xlabel("Error (meters)")
    ax.set_ylabel("Number of windows")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, "line_error_histogram.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_xy_combined_scatter(rows, output_dir):
    """Previous combined X/Y scatter plot."""
    if plt is None:
        return None

    gt_x = rows_to_series(rows, "gt_frame4_x_m")
    gt_y = rows_to_series(rows, "gt_frame4_y_m")
    pr_x = rows_to_series(rows, "pred_frame4_x_m")
    pr_y = rows_to_series(rows, "pred_frame4_y_m")

    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    ax.scatter(gt_x, pr_x, label="X coordinate")
    ax.scatter(gt_y, pr_y, label="Y coordinate")
    all_vals = gt_x + gt_y + pr_x + pr_y
    mn, mx = min(all_vals), max(all_vals)
    pad = max((mx - mn) * 0.05, 1.0)
    mn -= pad
    mx += pad
    ax.plot([mn, mx], [mn, mx], linewidth=2, linestyle="--", label="Perfect prediction")
    ax.set_title("Predicted vs Ground Truth Coordinates")
    ax.set_xlabel("Ground truth coordinate (meters)")
    ax.set_ylabel("Predicted coordinate (meters)")
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, "line_predicted_vs_ground_truth_scatter.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_single_coordinate_scatter(rows, output_dir, gt_key, pred_key, coord_name, filename):
    if plt is None:
        return None

    gt = rows_to_series(rows, gt_key)
    pred = rows_to_series(rows, pred_key)
    all_vals = gt + pred
    mn, mx = min(all_vals), max(all_vals)
    pad = max((mx - mn) * 0.05, 1.0)
    mn -= pad
    mx += pad

    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    ax.scatter(gt, pred, label=f"{coord_name} prediction")
    ax.plot([mn, mx], [mn, mx], linewidth=2, linestyle="--", label="Perfect prediction")
    ax.set_title(f"Line Trajectory: {coord_name} Coordinate Prediction")
    ax.set_xlabel(f"Ground truth {coord_name} coordinate (meters)")
    ax.set_ylabel(f"Predicted {coord_name} coordinate (meters)")
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_x_coordinate_scatter(rows, output_dir):
    """New requested plot: X scatter only."""
    return _plot_single_coordinate_scatter(
        rows=rows,
        output_dir=output_dir,
        gt_key="gt_frame4_x_m",
        pred_key="pred_frame4_x_m",
        coord_name="X / East",
        filename="line_x_coordinate_scatter.png",
    )


def plot_y_coordinate_scatter(rows, output_dir):
    """New requested plot: Y scatter only."""
    return _plot_single_coordinate_scatter(
        rows=rows,
        output_dir=output_dir,
        gt_key="gt_frame4_y_m",
        pred_key="pred_frame4_y_m",
        coord_name="Y / North",
        filename="line_y_coordinate_scatter.png",
    )


def plot_heading_predicted_vs_ground_truth_scatter(rows, output_dir):
    """New requested plot: predicted heading vs ground-truth heading."""
    if plt is None:
        return None

    gt = rows_to_series(rows, "gt_heading_deg")
    pred = rows_to_series(rows, "pred_heading_deg")

    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    ax.scatter(gt, pred, label="Heading prediction")
    ax.plot([0, 360], [0, 360], linewidth=2, linestyle="--", label="Perfect prediction")
    ax.set_title("Line Trajectory: Predicted vs Ground Truth Heading")
    ax.set_xlabel("Ground truth heading (degrees)")
    ax.set_ylabel("Predicted heading (degrees)")
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, "line_heading_predicted_vs_ground_truth.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_heading_result_vs_ground_truth_over_time(rows, output_dir):
    """New requested plot: heading result and ground truth over time."""
    if plt is None:
        return None

    ids = [int(r["frame4_id"]) for r in rows]
    gt = rows_to_series(rows, "gt_heading_deg")
    pred = rows_to_series(rows, "pred_heading_deg")

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    ax.plot(ids, gt, marker="o", linewidth=2, label="Ground truth heading")
    ax.plot(ids, pred, marker="o", linewidth=2, linestyle="--", label="Predicted heading")
    ax.set_title("Line Trajectory: Heading Result vs Ground Truth")
    ax.set_xlabel("Frame-4 ID")
    ax.set_ylabel("Heading (degrees)")
    ax.set_ylim(0, 360)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, "line_heading_result_vs_ground_truth_over_time.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def save_all_graphs(metrics, rows, output_dir):
    graph_fns = [
        plot_metric_summary,
        plot_trajectory,
        plot_error_over_time,
        plot_heading_error_over_time,
        plot_error_histogram,
        plot_xy_combined_scatter,
        plot_x_coordinate_scatter,
        plot_y_coordinate_scatter,
        plot_heading_predicted_vs_ground_truth_scatter,
        plot_heading_result_vs_ground_truth_over_time,
    ]

    graph_paths = []
    for fn in graph_fns:
        if fn.__name__ == "plot_metric_summary":
            p = fn(metrics, output_dir)
        else:
            p = fn(rows, output_dir)
        if p is not None:
            graph_paths.append(p)
    return graph_paths


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=r"C:\Users\admin\Desktop\spiral_avgso\localization_spiral_vgg\v4_original_center_radius_results")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--train_data_root", type=str, default=r"C:\Users\admin\Desktop\spiral_avgso\maps_screenshots_spiral_28150")
    ap.add_argument("--line_data_root", type=str, default=r"C:\Users\admin\Desktop\spiral_avgso\maps_screenshots_Horizontal_line_36_50m")
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0, help="Use 0 on Windows if multiprocessing causes issues.")
    ap.add_argument("--use_pretrained", type=int, default=1)
    args = ap.parse_args()

    checkpoint = args.checkpoint or os.path.join(args.project_root, "checkpoints", "best.pt")
    output_dir = args.output_dir or os.path.join(args.project_root, "line_generalization_results")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    print(f"Checkpoint: {checkpoint}")
    print(f"Training data root for origin/fallback normalization: {args.train_data_root}")
    print(f"External line data root: {args.line_data_root}")
    print(f"Saving outputs to: {output_dir}")

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    ckpt = torch.load(checkpoint, map_location=device)
    ckpt_cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}

    cfg = CFG(data_root=args.line_data_root)

    # Keep key settings consistent with the checkpoint when available.
    for key in [
        "image_size", "T_frames", "stride",
        "train_frac", "eval_frac", "test_frac",
        "w_overall_xy", "w_xy4", "w_head",
        "w_nce", "temperature",
    ]:
        if key in ckpt_cfg:
            setattr(cfg, key, ckpt_cfg[key])

    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.use_infonce = False
    cfg.use_pretrained = bool(args.use_pretrained)

    train_frames = list_frames_one_folder(args.train_data_root)
    train_num_windows = len(train_frames) - (cfg.T_frames - 1) * cfg.stride
    if train_num_windows <= 0:
        raise RuntimeError("Training data root does not have enough frames to rebuild normalization.")

    (tr_s, tr_e), _, _ = chronological_split(train_num_windows, cfg.train_frac, cfg.eval_frac, cfg.test_frac)

    # Critical: use the original training origin.
    lat0 = float(train_frames[0][2])
    lon0 = float(train_frames[0][3])

    ckpt_normalization = ckpt.get("normalization") if isinstance(ckpt, dict) else None
    if ckpt_normalization == "center_radius" and "xy_center" in ckpt and "radius_scale" in ckpt:
        xy_center = ckpt["xy_center"].detach().cpu().float()
        radius_scale = ckpt["radius_scale"].detach().cpu().float()
    else:
        print("WARNING: checkpoint did not include center/radius metadata; rebuilding fallback values from training data.")
        xy_center, radius_scale = compute_center_radius_normalization(train_frames, cfg, tr_s, tr_e, lat0, lon0)

    line_frames = list_frames_one_folder(args.line_data_root)
    line_num_windows = len(line_frames) - (cfg.T_frames - 1) * cfg.stride
    if line_num_windows <= 0:
        raise RuntimeError("Line data root does not have enough frames for 4-frame windows.")

    line_ds = GSOSlidingWindowDataset(
        line_frames,
        cfg,
        start_idx=0,
        end_idx=line_num_windows,
        lat0=lat0,
        lon0=lon0,
        xy_center=xy_center,
        radius_scale=radius_scale,
    )
    line_loader = DataLoader(
        line_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = SpatioTemporalTransformer(
        cfg,
        embed_dim=256,
        nhead=8,
        layers=4,
        dropout=0.1,
        backbone=BACKBONE,
    ).to(device)

    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    print(f"Training frames: {len(train_frames)} | Training windows: {train_num_windows}")
    print(f"Line frames: {len(line_frames)} | Line windows: {line_num_windows}")
    print(f"Training origin lat0/lon0: {lat0}, {lon0}")
    print("Training XY normalization: center/radius")
    print(f"Training XY center={xy_center.tolist()} radius_scale={float(radius_scale.item())}")

    metrics, rows = evaluate_external_line(model, line_loader, device, cfg, xy_center, radius_scale)
    metrics["num_line_frames"] = len(line_frames)

    metrics_csv = save_metrics_csv(metrics, output_dir)
    pred_csv = save_predictions_csv(rows, output_dir)
    report_txt = save_summary_report(
        metrics,
        output_dir,
        checkpoint,
        args.train_data_root,
        args.line_data_root,
        xy_center,
        radius_scale,
    )
    graph_paths = save_all_graphs(metrics, rows, output_dir)

    print("\nExternal line trajectory results:")
    print(f"  Line frames:                  {len(line_frames)}")
    print(f"  Line windows:                 {line_num_windows}")
    print(f"  Mean normalized loss:         {metrics['mean_normalized_loss']:.4f}")
    print(f"  Overall/window mean error:    {metrics['overall_mean_m']:.2f} m")
    print(f"  Overall/window median error:  {metrics['overall_median_m']:.2f} m")
    print(f"  Overall/window RMSE:          {metrics['overall_rmse_m']:.2f} m")
    print(f"  Overall/window 90th %ile:     {metrics['overall_p90_m']:.2f} m")
    print(f"  Frame-4 mean error:           {metrics['frame4_mean_m']:.2f} m")
    print(f"  Frame-4 median error:         {metrics['frame4_median_m']:.2f} m")
    print(f"  Frame-4 RMSE:                 {metrics['frame4_rmse_m']:.2f} m")
    print(f"  Frame-4 90th %ile:            {metrics['frame4_p90_m']:.2f} m")
    print(f"  Heading mean error:           {metrics['heading_mean_deg']:.2f} deg")
    print(f"  Heading median error:         {metrics['heading_median_deg']:.2f} deg")
    print(f"  Heading RMSE:                 {metrics['heading_rmse_deg']:.2f} deg")
    print(f"  Heading 90th %ile:            {metrics['heading_p90_deg']:.2f} deg")

    print("\nSaved outputs:")
    print(f"  {metrics_csv}")
    print(f"  {pred_csv}")
    print(f"  {report_txt}")

    if graph_paths:
        print("\nSaved graphs:")
        for p in graph_paths:
            print(f"  {p}")
    else:
        print("\nGraphs were not saved because matplotlib is not available.")

    print("\nExpected graph files:")
    expected = [
        "line_meter_error_summary.png",
        "line_trajectory_prediction_vs_ground_truth.png",
        "line_error_over_time.png",
        "line_heading_error_over_time.png",
        "line_error_histogram.png",
        "line_predicted_vs_ground_truth_scatter.png",
        "line_x_coordinate_scatter.png",
        "line_y_coordinate_scatter.png",
        "line_heading_predicted_vs_ground_truth.png",
        "line_heading_result_vs_ground_truth_over_time.png",
    ]
    for name in expected:
        print(f"  {os.path.join(output_dir, name)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
