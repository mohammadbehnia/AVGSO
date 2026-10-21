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
2) Predict 4th-frame location + heading from ONLY the first 3 frames.

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
- Heading via sin/cos with cosine distance
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
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 0.05

    # Minimal multi-task weights
    w_overall_xy: float = 1.0
    w_xy4: float = 1.0
    w_head: float = 0.001

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
    Accepts:
      frame_<FRAMEID>_<LAT>_<LON>_<HEADING>
      1_frame_<FRAMEID>_<LAT>_<LON>_<HEADING>
      2_frame_<FRAMEID>_<LAT>_<LON>_<HEADING>
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")

    # Case: frame_000001_36.045944_-79.960988_90
    if len(parts) >= 5 and parts[0].lower() == "frame":
        offset = 0
        seq_id = 0

    # Case: 1_frame_000001_36.045944_-79.960988_90
    elif len(parts) >= 6 and parts[0].isdigit() and parts[1].lower() == "frame":
        seq_id = int(parts[0])
        offset = 1

    else:
        return None

    try:
        local_frame_id = int(parts[offset + 1])
        lat = float(parts[offset + 2])
        lon = float(parts[offset + 3])
        heading = float(parts[offset + 4])

        # Makes 1_frame_000001 and 2_frame_000001 sort separately
        frame_id = seq_id * 1_000_000_000 + local_frame_id

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
        if "_frame_" not in fn.lower() and not fn.lower().startswith("frame_"):
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

        return {
            "images": imgs,
            "gt_overall_xy": xy.mean(dim=0),
            "gt_xy4": xy[3],
            "gt_heading4": head[3],
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
            nn.Linear(512, 4),  # x4,y4,sin,cos
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

        pred = self.head_frame4(out_ctx[:, 0])      # (B,4)
        pred_xy4 = pred[:, :2]
        pred_sincos4 = F.normalize(pred[:, 2:], dim=-1)

        # embeddings for optional InfoNCE
        tok_frames = tok_bt.view(B, T, L, -1)
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
# Losses (minimal + optional InfoNCE)
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

    loss = cfg.w_overall_xy*l_overall + cfg.w_xy4*l_xy4 + cfg.w_head*l_head
    logs = {"l_overall": l_overall.detach(), "l_xy4": l_xy4.detach(), "l_head": l_head.detach()}

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
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(batch["images"])
        loss, _ = total_loss(out, batch, cfg)
        tot += loss.item() * batch["images"].size(0)
        n += batch["images"].size(0)
    model.train()
    return tot / max(n, 1)

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=r"C:\Users\admin\Desktop\spiral_avgso\Combined_dataset_59818")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--use_infonce", type=int, default=0)
    ap.add_argument("--use_pretrained", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    cfg = CFG(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_infonce=bool(args.use_infonce),
        use_pretrained=bool(args.use_pretrained),
        num_workers=args.num_workers,
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

    train_ds = GSOSlidingWindowDataset(frames, cfg, tr_s, tr_e)
    eval_ds  = GSOSlidingWindowDataset(frames, cfg, ev_s, ev_e, lat0=train_ds.lat0, lon0=train_ds.lon0)
    test_ds  = GSOSlidingWindowDataset(frames, cfg, te_s, te_e, lat0=train_ds.lat0, lon0=train_ds.lon0)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    eval_loader  = DataLoader(eval_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    model = SpatioTemporalTransformer(cfg, embed_dim=256, nhead=8, layers=4, dropout=0.1, backbone=BACKBONE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda")

    os.makedirs("checkpoints", exist_ok=True)
    best = 1e9

    print(f"Data root: {cfg.data_root}")
    print(f"Frames: {total_frames} | Windows: {num_windows}")
    print(f"Split windows: train={len(train_ds)} eval={len(eval_ds)} test={len(test_ds)} (chronological)")
    print(f"Use InfoNCE: {cfg.use_infonce} (w={cfg.w_nce}, T={cfg.temperature})")
    print("Saving best model to: checkpoints/best.pt")

    with open('loss_log.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'train_loss', 'val_loss'])

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running = 0.0
            seen = 0

            for batch in train_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    out = model(batch["images"])
                    loss, logs = total_loss(out, batch, cfg)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                running += loss.item() * batch["images"].size(0)
                seen += batch["images"].size(0)
                print(f"{running:.2f}    |    {loss.item():.2f}    |    {seen}    |    {(running / max(seen, 1)):.2f}", end='\r')

            train_loss = running / max(seen, 1)
            eval_loss = evaluate(model, eval_loader, device, cfg)
            print(f"Epoch {epoch:03d} | train={train_loss:.4f} | eval={eval_loss:.4f}")
            writer.writerow([f"{epoch:03d}" , f"{train_loss:.4f}" , f"{eval_loss:.4f}"])

            if eval_loss < best:
                best = eval_loss
                torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, "checkpoints/best.pt")

    test_loss = evaluate(model, test_loader, device, cfg)
    print(f"Best eval: {best:.4f} | Test: {test_loss:.4f}")

    print("Done.")

if __name__ == "__main__":
    main()
