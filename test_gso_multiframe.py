# -*- coding: utf-8 -*-
"""
Test / inference script for GSO multiframe localization.

What it does:
- Loads frames from ONE folder (same naming convention)
- Rebuilds the chronological test split (same logic as training)
- Loads checkpoints/best.pt (or a path you specify)
- Runs the model on the test set and prints:
    - Overall location error (meters): mean/median
    - Frame4 location error (meters): mean/median
    - Heading error (degrees): mean/median
- Optionally saves per-sample results to CSV.

Run:
  pip install -r requirements.txt
  python test_gso_multiframe.py --data_root "D:\\PhD\\Code\\gso\\New_GSO\\maps_screenshots" --ckpt "checkpoints\\best.pt"
"""

import os
import math
import csv
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import torchvision.transforms as T
from torchvision.models import vgg16, VGG16_Weights


# -----------------------------
# Config (keep consistent with training defaults)
# -----------------------------
@dataclass
class CFG:
    data_root: str
    image_size: int = 224
    T_frames: int = 4
    stride: int = 1

    # chronological splits (same as train.py)
    train_frac: float = 0.80
    eval_frac: float = 0.10
    test_frac: float = 0.10
    seed: int = 42

    batch_size: int = 8
    num_workers: int = 4

    use_pretrained_vgg: bool = False


# -----------------------------
# Geo helpers
# -----------------------------
EARTH_R = 6378137.0  # meters

def ll_to_local_xy(lat_deg: torch.Tensor, lon_deg: torch.Tensor, lat0_deg: float, lon0_deg: float):
    lat = torch.deg2rad(lat_deg)
    lon = torch.deg2rad(lon_deg)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    x = (lon - lon0) * math.cos(lat0) * EARTH_R
    y = (lat - lat0) * EARTH_R
    return x, y

def heading_to_rad(h: torch.Tensor):
    if torch.max(torch.abs(h)).item() > 6.5:
        return torch.deg2rad(h)
    return h

def angle_error_deg(pred_sincos: torch.Tensor, gt_heading_rad: torch.Tensor) -> torch.Tensor:
    """
    pred_sincos: (B,2) normalized [sin, cos]
    gt_heading_rad: (B,)
    returns abs angular error in degrees (0..180)
    """
    gt = torch.stack([torch.sin(gt_heading_rad), torch.cos(gt_heading_rad)], dim=-1)
    gt = F.normalize(gt, dim=-1)
    cos_sim = (pred_sincos * gt).sum(dim=-1).clamp(-1.0, 1.0)
    err_rad = torch.acos(cos_sim)
    return torch.rad2deg(err_rad)


# -----------------------------
# File parsing (same as training)
# -----------------------------
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

def parse_frame_name(fname: str) -> Optional[Tuple[int, float, float, float]]:
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
        raise RuntimeError("Need at least 4 frames.")
    return items

def chronological_split(num_windows: int, train_frac: float, eval_frac: float, test_frac: float):
    assert abs(train_frac + eval_frac + test_frac - 1.0) < 1e-6
    n_train = int(num_windows * train_frac)
    n_eval = int(num_windows * eval_frac)
    tr = (0, n_train)
    ev = (n_train, n_train + n_eval)
    te = (n_train + n_eval, num_windows)
    return tr, ev, te


# -----------------------------
# Dataset (same windowing)
# -----------------------------
class GSOSlidingWindowDataset(Dataset):
    def __init__(
        self,
        frames: List[Tuple[str, int, float, float, float]],
        cfg: CFG,
        start_idx: int,
        end_idx: int,
        lat0: Optional[float] = None,
        lon0: Optional[float] = None,
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

        imgs = torch.stack([self._load_img(c[0]) for c in chosen], dim=0)

        lat = torch.tensor([c[2] for c in chosen], dtype=torch.float32)
        lon = torch.tensor([c[3] for c in chosen], dtype=torch.float32)
        head = heading_to_rad(torch.tensor([c[4] for c in chosen], dtype=torch.float32))

        x, y = ll_to_local_xy(lat, lon, self.lat0, self.lon0)
        xy = torch.stack([x, y], dim=-1)

        return {
            "images": imgs,
            "gt_overall_xy": xy.mean(dim=0),
            "gt_xy4": xy[3],
            "gt_heading4": head[3],
            "frame_id4": torch.tensor(chosen[3][1], dtype=torch.int64),
        }


# -----------------------------
# Model (same architecture as training file)
# -----------------------------
class SpatioTemporalTransformer(nn.Module):
    def __init__(self, cfg: CFG, embed_dim: int = 256, nhead: int = 8, layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cfg = cfg
        self.T = cfg.T_frames

        weights = VGG16_Weights.IMAGENET1K_V1 if cfg.use_pretrained_vgg else None
        base = vgg16(weights=weights)
        self.backbone = base.features
        self.in_channels = 512

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
            nn.Linear(512, 4),
        )

    def _frame_tokens(self, images_btchw: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(images_btchw)
        B_T, C, Hf, Wf = feat.shape
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
        B, T, C, H, W = images.shape
        assert T == 4

        x = images.view(B*T, C, H, W)
        tok_bt = self._frame_tokens(x)
        L = tok_bt.size(1)

        # Pass A
        tok_all = tok_bt.view(B, T*L, -1)
        tok_all = self._add_positional(tok_all, Tctx=4)
        cls = self.cls.expand(B, -1, -1)
        out_all = self.encoder(torch.cat([cls, tok_all], dim=1))
        overall_xy = self.head_overall(out_all[:, 0])

        # Pass B (3-frame context)
        Tctx = 3
        tok_ctx = tok_bt.view(B, T, L, -1)[:, :Tctx].reshape(B, Tctx*L, -1)
        tok_ctx = self._add_positional(tok_ctx, Tctx=3)
        out_ctx = self.encoder(torch.cat([cls, tok_ctx], dim=1))
        pred = self.head_frame4(out_ctx[:, 0])
        pred_xy4 = pred[:, :2]
        pred_sincos4 = F.normalize(pred[:, 2:], dim=-1)

        return {"overall_xy": overall_xy, "pred_xy4": pred_xy4, "pred_sincos4": pred_sincos4}


# -----------------------------
# Test
# -----------------------------
@torch.no_grad()
def run_test(model, loader, device, save_csv_path: Optional[str] = None):
    model.eval()
    overall_errs = []
    xy4_errs = []
    head_errs = []
    rows = []

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        gt_overall = batch["gt_overall_xy"].to(device, non_blocking=True)
        gt_xy4 = batch["gt_xy4"].to(device, non_blocking=True)
        gt_head4 = batch["gt_heading4"].to(device, non_blocking=True)
        frame_id4 = batch["frame_id4"].cpu().numpy()

        out = model(images)

        # errors
        e_overall = torch.linalg.norm(out["overall_xy"] - gt_overall, dim=-1)  # meters
        e_xy4 = torch.linalg.norm(out["pred_xy4"] - gt_xy4, dim=-1)            # meters
        e_head = angle_error_deg(out["pred_sincos4"], gt_head4)                # deg

        overall_errs.append(e_overall.cpu())
        xy4_errs.append(e_xy4.cpu())
        head_errs.append(e_head.cpu())

        if save_csv_path:
            for i in range(images.size(0)):
                rows.append({
                    "frame_id4": int(frame_id4[i]),
                    "overall_err_m": float(e_overall[i].cpu()),
                    "xy4_err_m": float(e_xy4[i].cpu()),
                    "head_err_deg": float(e_head[i].cpu()),
                })

    overall_errs = torch.cat(overall_errs)
    xy4_errs = torch.cat(xy4_errs)
    head_errs = torch.cat(head_errs)

    def stats(x):
        return float(x.mean()), float(x.median())

    o_mean, o_med = stats(overall_errs)
    x_mean, x_med = stats(xy4_errs)
    h_mean, h_med = stats(head_errs)

    print("==== TEST RESULTS ====")
    print(f"Overall location error (m): mean={o_mean:.3f} | median={o_med:.3f}")
    print(f"Frame4   location error (m): mean={x_mean:.3f} | median={x_med:.3f}")
    print(f"Heading error (deg):         mean={h_mean:.3f} | median={h_med:.3f}")

    if save_csv_path:
        with open(save_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["frame_id4", "overall_err_m", "xy4_err_m", "head_err_deg"])
            w.writeheader()
            w.writerows(rows)
        print(f"Saved per-sample CSV: {save_csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=r"D:\PhD\Code\gso\New_GSO\maps_screenshots")
    ap.add_argument("--ckpt", type=str, default=r"checkpoints\best.pt")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--save_csv", type=str, default="")
    ap.add_argument("--use_pretrained_vgg", type=int, default=0)
    args = ap.parse_args()

    cfg = CFG(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_pretrained_vgg=bool(args.use_pretrained_vgg),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    frames = list_frames_one_folder(cfg.data_root)
    total_frames = len(frames)
    num_windows = total_frames - (cfg.T_frames - 1) * cfg.stride

    (tr_s, tr_e), (ev_s, ev_e), (te_s, te_e) = chronological_split(num_windows, cfg.train_frac, cfg.eval_frac, cfg.test_frac)

    # build test dataset; origin from frame0 for consistency with training file default
    lat0 = frames[0][2]
    lon0 = frames[0][3]
    test_ds = GSOSlidingWindowDataset(frames, cfg, te_s, te_e, lat0=lat0, lon0=lon0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    model = SpatioTemporalTransformer(cfg, embed_dim=256, nhead=8, layers=4, dropout=0.1).to(device)

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    save_csv_path = args.save_csv.strip() or None
    run_test(model, test_loader, device, save_csv_path=save_csv_path)


if __name__ == "__main__":
    main()
