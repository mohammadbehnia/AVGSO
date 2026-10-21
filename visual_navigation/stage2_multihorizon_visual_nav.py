# -*- coding: utf-8 -*-
"""
Stage 2A multi-horizon visual navigation for AVGSO spiral data.

Assumption:
  The model receives a current multi-frame visual sequence and a target image
  sampled from a fixed future frame offset. It does not receive start, goal,
  current, or target coordinates as inputs. Coordinates are used only to build
  labels and evaluation metrics.

Inputs:
  - current multi-frame visual sequence
  - target image from current + horizon
  - horizon token for offsets +1, +4, +10, +40

Outputs:
  - command per horizon: STOP, LEFT, RIGHT, FORWARD, TURN_AROUND
  - target-relative state: dx, dy, remaining distance, progress, heading error
  - reached probability

This is a trajectory-following visual navigation formulation. It tests whether
the model can learn horizon-specific navigation behavior from future target
views without explicit coordinate conditioning.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

TORCH_AVAILABLE = True
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as T
    from torchvision.models import VGG16_BN_Weights, VGG16_Weights, vgg16, vgg16_bn
except ModuleNotFoundError:
    TORCH_AVAILABLE = False

    class _NoGrad:
        def __call__(self, fn=None):
            if fn is None:
                return self
            return fn

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _TorchStub:
        Tensor = object
        no_grad = _NoGrad()

    class _NNStub:
        Module = object

    torch = _TorchStub()
    nn = _NNStub()
    F = None
    DataLoader = object
    Dataset = object
    T = None
    VGG16_BN_Weights = None
    VGG16_Weights = None
    vgg16 = None
    vgg16_bn = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


EARTH_R = 6378137.0
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
COMMANDS = ["STOP", "LEFT", "RIGHT", "FORWARD", "TURN_AROUND"]
COMMAND_TO_ID = {name: i for i, name in enumerate(COMMANDS)}
HORIZONS = ["plus1", "plus4", "plus10", "plus40"]
HORIZON_OFFSETS = [1, 4, 10, 40]
HORIZON_TO_ID = {name: i for i, name in enumerate(HORIZONS)}


@dataclass
class FrameRecord:
    path: str
    frame_id: int
    lat: float
    lon: float
    heading_rad: float
    x: float
    y: float


@dataclass
class MultiHorizonSample:
    current_idx: int
    target_idx: int
    horizon_id: int
    offset: int
    command_id: int
    reached: float
    dx_to_target: float
    dy_to_target: float
    distance_m: float
    progress: float
    heading_error_rad: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def heading_to_rad(value: float) -> float:
    return math.radians(value) if abs(value) > 6.5 else value


def wrap_angle_rad(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def angular_error_deg_np(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    diff = np.arctan2(np.sin(pred - gt), np.cos(pred - gt))
    return np.degrees(np.abs(diff))


def ll_to_local_xy(lat_deg: float, lon_deg: float, lat0_deg: float, lon0_deg: float) -> Tuple[float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    x = (lon - lon0) * math.cos(lat0) * EARTH_R
    y = (lat - lat0) * EARTH_R
    return x, y


def parse_frame_name(path: Path) -> Optional[Tuple[int, float, float, float]]:
    parts = path.stem.split("_")
    try:
        frame_pos = parts.index("frame")
    except ValueError:
        return None
    if len(parts) < frame_pos + 5:
        return None
    try:
        frame_id = int(parts[frame_pos + 1])
        lat = float(parts[frame_pos + 2])
        lon = float(parts[frame_pos + 3])
        heading = float(parts[frame_pos + 4])
    except ValueError:
        return None
    return frame_id, lat, lon, heading


def list_frames_one_folder(data_root: str) -> List[FrameRecord]:
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"data_root not found: {root}")

    parsed: List[Tuple[Path, int, float, float, float]] = []
    for path in root.iterdir():
        if path.suffix.lower() not in IMG_EXT:
            continue
        item = parse_frame_name(path)
        if item is None:
            continue
        frame_id, lat, lon, heading = item
        parsed.append((path, frame_id, lat, lon, heading))

    parsed.sort(key=lambda row: (row[1], str(row[0])))
    if len(parsed) < 64:
        raise RuntimeError(f"Found only {len(parsed)} valid frames in {root}.")

    lat0 = parsed[0][2]
    lon0 = parsed[0][3]
    records: List[FrameRecord] = []
    for path, frame_id, lat, lon, heading in parsed:
        x, y = ll_to_local_xy(lat, lon, lat0, lon0)
        records.append(
            FrameRecord(
                path=str(path),
                frame_id=frame_id,
                lat=lat,
                lon=lon,
                heading_rad=heading_to_rad(heading),
                x=x,
                y=y,
            )
        )
    return records


def split_ranges(num_frames: int, train_frac: float, val_frac: float, split_gap: int) -> Dict[str, Tuple[int, int]]:
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("Expected 0 < train_frac, 0 < val_frac, and train_frac + val_frac < 1.")
    train_end = int(num_frames * train_frac)
    val_end = int(num_frames * (train_frac + val_frac))
    ranges = {
        "train": (0, max(0, train_end)),
        "val": (min(num_frames, train_end + split_gap), max(train_end + split_gap, val_end)),
        "test": (min(num_frames, val_end + split_gap), num_frames),
    }
    for name, (start, end) in ranges.items():
        if end <= start:
            raise RuntimeError(f"Empty {name} split after split_gap={split_gap}: start={start}, end={end}")
    return ranges


def command_from_target_geometry(
    dx_to_target: float,
    dy_to_target: float,
    current_heading_rad: float,
    stop_threshold_m: float,
    forward_angle_rad: float,
    turn_around_angle_rad: float,
) -> Tuple[int, float, float]:
    distance = math.hypot(dx_to_target, dy_to_target)
    if distance <= stop_threshold_m:
        return COMMAND_TO_ID["STOP"], 0.0, distance

    target_direction = math.atan2(dx_to_target, dy_to_target)
    heading_error = wrap_angle_rad(target_direction - current_heading_rad)
    abs_error = abs(heading_error)
    if abs_error <= forward_angle_rad:
        return COMMAND_TO_ID["FORWARD"], heading_error, distance
    if abs_error >= turn_around_angle_rad:
        return COMMAND_TO_ID["TURN_AROUND"], heading_error, distance
    if heading_error < 0:
        return COMMAND_TO_ID["LEFT"], heading_error, distance
    return COMMAND_TO_ID["RIGHT"], heading_error, distance


def build_samples_for_split(
    records: Sequence[FrameRecord],
    split_range: Tuple[int, int],
    seq_len: int,
    offsets: Sequence[int],
    stop_threshold_m: float,
    forward_angle_rad: float,
    turn_around_angle_rad: float,
    stride: int,
    approx_m_per_frame: float,
    max_offset_distance_factor: float,
    max_offset_distance_extra_m: float,
) -> List[MultiHorizonSample]:
    start, end = split_range
    max_offset = max(offsets)
    first_current = start + seq_len - 1
    last_current = end - max_offset - 1
    if last_current <= first_current:
        raise RuntimeError(f"Split too small for offsets={offsets}: start={start}, end={end}")

    samples: List[MultiHorizonSample] = []
    for current_idx in range(first_current, last_current + 1, max(stride, 1)):
        cur = records[current_idx]
        for horizon_id, offset in enumerate(offsets):
            target_idx = current_idx + offset
            target = records[target_idx]
            dx = target.x - cur.x
            dy = target.y - cur.y
            distance = math.hypot(dx, dy)
            expected_distance = approx_m_per_frame * offset
            max_allowed_distance = max(
                expected_distance * max_offset_distance_factor,
                expected_distance + max_offset_distance_extra_m,
            )
            if distance > max_allowed_distance:
                continue
            command_id, heading_error, distance = command_from_target_geometry(
                dx, dy, cur.heading_rad, stop_threshold_m, forward_angle_rad, turn_around_angle_rad
            )
            progress = min(1.0, offset / max(float(max_offset), 1.0))
            samples.append(
                MultiHorizonSample(
                    current_idx=current_idx,
                    target_idx=target_idx,
                    horizon_id=horizon_id,
                    offset=offset,
                    command_id=command_id,
                    reached=1.0 if distance <= stop_threshold_m else 0.0,
                    dx_to_target=dx,
                    dy_to_target=dy,
                    distance_m=distance,
                    progress=progress,
                    heading_error_rad=heading_error,
                )
            )
    if not samples:
        raise RuntimeError("No multi-horizon samples were built.")
    return samples


def maybe_limit(samples: List[MultiHorizonSample], limit: int, seed: int) -> List[MultiHorizonSample]:
    if limit <= 0 or len(samples) <= limit:
        return samples
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    keep = sorted(indices[:limit])
    return [samples[i] for i in keep]


def command_counts(samples: Sequence[MultiHorizonSample]) -> Dict[str, int]:
    counts = {name: 0 for name in COMMANDS}
    for sample in samples:
        counts[COMMANDS[sample.command_id]] += 1
    return counts


def horizon_counts(samples: Sequence[MultiHorizonSample]) -> Dict[str, int]:
    counts = {name: 0 for name in HORIZONS}
    for sample in samples:
        counts[HORIZONS[sample.horizon_id]] += 1
    return counts


def numeric_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def sample_stats(samples: Sequence[MultiHorizonSample]) -> Dict[str, Dict[str, float]]:
    return {
        "target_distance_m": numeric_stats([s.distance_m for s in samples]),
        "heading_error_deg": numeric_stats([math.degrees(s.heading_error_rad) for s in samples]),
        "abs_heading_error_deg": numeric_stats([abs(math.degrees(s.heading_error_rad)) for s in samples]),
    }


def horizon_sample_stats(samples: Sequence[MultiHorizonSample]) -> Dict[str, Dict[str, Dict[str, float] | int]]:
    stats: Dict[str, Dict[str, Dict[str, float] | int]] = {}
    for horizon_id, horizon_name in enumerate(HORIZONS):
        subset = [s for s in samples if s.horizon_id == horizon_id]
        stats[horizon_name] = {"support": len(subset), **sample_stats(subset)}
    return stats


def class_weights_from_counts(counts: Dict[str, int]) -> torch.Tensor:
    values = np.array([counts[name] for name in COMMANDS], dtype=np.float32)
    values = np.maximum(values, 1.0)
    weights = values.sum() / (len(values) * values)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def pos_weight_from_samples(samples: Sequence[MultiHorizonSample]) -> float:
    positives = sum(1 for s in samples if s.reached > 0.5)
    negatives = max(len(samples) - positives, 1)
    return float(negatives / max(positives, 1))


def compute_normalization(samples: Sequence[MultiHorizonSample], max_distance_m: float) -> Dict[str, float]:
    return {"state_distance_scale": float(max(max_distance_m, 1.0))}


class MultiHorizonDataset(Dataset):
    def __init__(
        self,
        records: Sequence[FrameRecord],
        samples: Sequence[MultiHorizonSample],
        normalization: Dict[str, float],
        seq_len: int,
        image_size: int,
    ):
        self.records = records
        self.samples = samples
        self.seq_len = seq_len
        self.state_scale = float(normalization["state_distance_scale"])
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_norm(self, path: str) -> torch.Tensor:
        with Image.open(path) as img:
            return self.transform(img.convert("RGB"))

    def _state_tensor(self, sample: MultiHorizonSample) -> torch.Tensor:
        return torch.tensor(
            [
                sample.dx_to_target / self.state_scale,
                sample.dy_to_target / self.state_scale,
                sample.distance_m / self.state_scale,
                sample.progress,
                math.sin(sample.heading_error_rad),
                math.cos(sample.heading_error_rad),
            ],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        seq_indices = range(sample.current_idx - self.seq_len + 1, sample.current_idx + 1)
        current_images = torch.stack([self._load_norm(self.records[i].path) for i in seq_indices], dim=0)
        target_image = self._load_norm(self.records[sample.target_idx].path)
        return {
            "current_images": current_images,
            "target_image": target_image,
            "horizon_id": torch.tensor(sample.horizon_id, dtype=torch.long),
            "command": torch.tensor(sample.command_id, dtype=torch.long),
            "reached": torch.tensor(sample.reached, dtype=torch.float32),
            "state": self._state_tensor(sample),
            "offset": torch.tensor(sample.offset, dtype=torch.long),
            "target_distance_m": torch.tensor(sample.distance_m, dtype=torch.float32),
            "heading_error": torch.tensor(sample.heading_error_rad, dtype=torch.float32),
            "current_frame_id": torch.tensor(self.records[sample.current_idx].frame_id, dtype=torch.long),
            "target_frame_id": torch.tensor(self.records[sample.target_idx].frame_id, dtype=torch.long),
            "current_path": Path(self.records[sample.current_idx].path).name,
            "target_path": Path(self.records[sample.target_idx].path).name,
        }


class VGGTokenStem(nn.Module):
    def __init__(self, backbone: str, d_model: int, use_pretrained: bool, freeze_backbone: bool = False):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        weights = None
        if use_pretrained:
            try:
                weights = VGG16_BN_Weights.IMAGENET1K_V1 if backbone == "vgg16_bn" else VGG16_Weights.IMAGENET1K_V1
            except Exception as exc:
                print(f"Warning: could not prepare pretrained weights ({exc}); using random initialization.")
                weights = None
        try:
            base = vgg16_bn(weights=weights) if backbone == "vgg16_bn" else vgg16(weights=weights)
        except Exception as exc:
            if use_pretrained:
                print(f"Warning: pretrained backbone load failed ({exc}); retrying without pretrained weights.")
                base = vgg16_bn(weights=None) if backbone == "vgg16_bn" else vgg16(weights=None)
            else:
                raise
        self.features = base.features
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        self.proj = nn.Linear(512, d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                feat = self.features(images)
        else:
            feat = self.features(images)
        tokens = feat.flatten(2).transpose(1, 2)
        return self.proj(tokens)


class MultiHorizonNavModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        nhead: int,
        layers: int,
        dropout: float,
        backbone: str,
        use_pretrained: bool,
        freeze_backbone: bool,
        max_tokens: int = 1024,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.max_tokens = max_tokens
        self.stem = VGGTokenStem(backbone, d_model, use_pretrained, freeze_backbone)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.time_emb = nn.Embedding(seq_len, d_model)
        self.space_emb = nn.Embedding(max_tokens, d_model)
        self.current_segment = nn.Parameter(torch.zeros(1, 1, d_model))
        self.target_segment = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.current_segment, std=0.02)
        nn.init.trunc_normal_(self.target_segment, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.target_mlp = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout))
        self.horizon_emb = nn.Embedding(len(HORIZONS), d_model)
        fusion_dim = d_model * 6
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.command_head = self._head(d_model, len(COMMANDS), dropout)
        self.reached_head = self._head(d_model, 1, dropout)
        self.state_head = self._head(d_model, 6, dropout)

    @staticmethod
    def _head(d_model: int, out_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

    def _add_current_pos(self, tokens: torch.Tensor, t: int, l_tokens: int) -> torch.Tensor:
        time_ids = torch.arange(t, device=tokens.device).repeat_interleave(l_tokens)
        space_ids = torch.arange(l_tokens, device=tokens.device).repeat(t)
        pos = self.time_emb(time_ids)[None, :, :] + self.space_emb(space_ids)[None, :, :]
        return tokens + pos + self.current_segment

    def _add_target_pos(self, tokens: torch.Tensor) -> torch.Tensor:
        space_ids = torch.arange(tokens.size(1), device=tokens.device)
        return tokens + self.space_emb(space_ids)[None, :, :] + self.target_segment

    def forward(self, current_images: torch.Tensor, target_image: torch.Tensor, horizon_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, t, c, h, w = current_images.shape
        if t != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {t}")
        cur_tokens_bt = self.stem(current_images.reshape(b * t, c, h, w))
        l_tokens = cur_tokens_bt.size(1)
        if l_tokens > self.max_tokens:
            raise RuntimeError(f"Feature tokens {l_tokens} exceed max_tokens={self.max_tokens}")
        cur_tokens = cur_tokens_bt.reshape(b, t * l_tokens, self.d_model)
        cur_tokens = self._add_current_pos(cur_tokens, t, l_tokens)
        cls = self.cls.expand(b, -1, -1)
        cur_encoded = self.encoder(torch.cat([cls, cur_tokens], dim=1))
        current_repr = cur_encoded[:, 0]

        target_tokens = self.stem(target_image)
        target_tokens = self._add_target_pos(target_tokens)
        target_repr = self.target_mlp(target_tokens.mean(dim=1))
        horizon_repr = self.horizon_emb(horizon_id)
        fusion_input = torch.cat(
            [
                current_repr,
                target_repr,
                current_repr - target_repr,
                torch.abs(current_repr - target_repr),
                current_repr * target_repr,
                horizon_repr,
            ],
            dim=-1,
        )
        fused = self.fusion(fusion_input)
        state = self.state_head(fused)
        state = torch.cat([state[:, :4], F.normalize(state[:, 4:], dim=-1, eps=1e-6)], dim=-1)
        return {
            "command_logits": self.command_head(fused),
            "reached_logit": self.reached_head(fused).squeeze(-1),
            "state": state,
        }


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved: Dict[str, object] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def circular_sincos_loss(pred_sincos: torch.Tensor, gt_sincos: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred_sincos, dim=-1, eps=1e-6)
    gt = F.normalize(gt_sincos, dim=-1, eps=1e-6)
    cos_sim = (pred * gt).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cos_sim).mean()


def state_loss(pred: torch.Tensor, gt: torch.Tensor, args: argparse.Namespace) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    loss_rel = F.smooth_l1_loss(pred[:, :2], gt[:, :2], beta=args.huber_beta)
    loss_distance = F.smooth_l1_loss(pred[:, 2], gt[:, 2], beta=args.huber_beta)
    loss_progress = F.smooth_l1_loss(pred[:, 3], gt[:, 3], beta=args.huber_beta)
    loss_heading = circular_sincos_loss(pred[:, 4:], gt[:, 4:])
    total = (
        args.w_state_rel * loss_rel
        + args.w_state_distance * loss_distance
        + args.w_state_progress * loss_progress
        + args.w_state_heading * loss_heading
    )
    return total, {
        "state_rel": loss_rel,
        "state_distance": loss_distance,
        "state_progress": loss_progress,
        "state_heading": loss_heading,
    }


def total_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    command_weights: torch.Tensor,
    reached_pos_weight: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    command_loss = F.cross_entropy(out["command_logits"], batch["command"], weight=command_weights)
    state_total, parts = state_loss(out["state"], batch["state"], args)
    reached_loss = F.binary_cross_entropy_with_logits(out["reached_logit"], batch["reached"], pos_weight=reached_pos_weight)
    loss = args.w_command * command_loss + args.w_state * state_total + args.w_reached * reached_loss
    logs = {
        "loss": float(loss.detach().cpu()),
        "command": float(command_loss.detach().cpu()),
        "state": float(state_total.detach().cpu()),
        "reached": float(reached_loss.detach().cpu()),
        "state_rel": float(parts["state_rel"].detach().cpu()),
        "state_distance": float(parts["state_distance"].detach().cpu()),
        "state_progress": float(parts["state_progress"].detach().cpu()),
        "state_heading": float(parts["state_heading"].detach().cpu()),
    }
    return loss, logs


def average_logs(total: Dict[str, float], logs: Dict[str, float], batch_size: int) -> None:
    for key, value in logs.items():
        total[key] = total.get(key, 0.0) + value * batch_size


def finalize_logs(total: Dict[str, float], count: int) -> Dict[str, float]:
    return {key: value / max(count, 1) for key, value in total.items()}


def write_live_progress(path: Optional[Path], message: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(message + "\n")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    command_weights: torch.Tensor,
    reached_pos_weight: torch.Tensor,
    args: argparse.Namespace,
    epoch: int,
    total_epochs: int,
    live_progress_path: Optional[Path],
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    count = 0
    start_time = time.time()
    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(bool(args.amp) and device.type == "cuda")):
            out = model(batch["current_images"], batch["target_image"], batch["horizon_id"])
            loss, logs = total_loss(out, batch, command_weights, reached_pos_weight, args)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        bs = batch["current_images"].size(0)
        average_logs(totals, logs, bs)
        count += bs
        if args.progress_every > 0 and (batch_idx == 1 or batch_idx == total_batches or batch_idx % args.progress_every == 0):
            running = finalize_logs(totals, count)
            elapsed = max(time.time() - start_time, 1e-6)
            batches_per_sec = batch_idx / elapsed
            remaining = (total_batches - batch_idx) / max(batches_per_sec, 1e-6)
            message = (
                f"Epoch {epoch:03d}/{total_epochs:03d} "
                f"batch {batch_idx:05d}/{total_batches:05d} samples {count:06d} "
                f"loss={running['loss']:.4f} cmd={running['command']:.4f} "
                f"state={running['state']:.4f} elapsed={elapsed/60:.1f}m eta={remaining/60:.1f}m"
            )
            print(message, flush=True)
            write_live_progress(live_progress_path, message)
    return finalize_logs(totals, count)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    command_weights: torch.Tensor,
    reached_pos_weight: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch["current_images"], batch["target_image"], batch["horizon_id"])
        _, logs = total_loss(out, batch, command_weights, reached_pos_weight, args)
        bs = batch["current_images"].size(0)
        average_logs(totals, logs, bs)
        count += bs
    return finalize_logs(totals, count)


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def decode_state(state: np.ndarray, scale: float) -> Dict[str, np.ndarray]:
    dx = state[:, 0] * scale
    dy = state[:, 1] * scale
    distance = state[:, 2] * scale
    progress = state[:, 3]
    heading = np.arctan2(state[:, 4], state[:, 5])
    return {"dx": dx, "dy": dy, "distance": distance, "progress": progress, "heading_error": heading}


@torch.no_grad()
def evaluate_full(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    state_scale: float,
    reached_threshold: float,
) -> Tuple[Dict[str, float], List[Dict[str, object]], Dict[str, np.ndarray]]:
    model.eval()
    rows: List[Dict[str, object]] = []
    arrays: Dict[str, List[np.ndarray]] = {
        "gt_command": [],
        "pred_command": [],
        "gt_reached": [],
        "pred_reached_prob": [],
        "gt_state": [],
        "pred_state": [],
        "horizon_id": [],
        "offset": [],
        "current_frame_id": [],
        "target_frame_id": [],
    }

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch["current_images"], batch["target_image"], batch["horizon_id"])
        pred_command = torch.argmax(out["command_logits"], dim=-1)
        pred_reached_prob = torch.sigmoid(out["reached_logit"])
        batch_np = {
            "gt_command": tensor_to_numpy(batch["command"]),
            "pred_command": tensor_to_numpy(pred_command),
            "gt_reached": tensor_to_numpy(batch["reached"]),
            "pred_reached_prob": tensor_to_numpy(pred_reached_prob),
            "gt_state": tensor_to_numpy(batch["state"]),
            "pred_state": tensor_to_numpy(out["state"]),
            "horizon_id": tensor_to_numpy(batch["horizon_id"]),
            "offset": tensor_to_numpy(batch["offset"]),
            "current_frame_id": tensor_to_numpy(batch["current_frame_id"]),
            "target_frame_id": tensor_to_numpy(batch["target_frame_id"]),
        }
        for key, value in batch_np.items():
            arrays[key].append(np.asarray(value))

        current_paths = batch["current_path"]
        target_paths = batch["target_path"]
        gt_state = decode_state(batch_np["gt_state"], state_scale)
        pred_state = decode_state(batch_np["pred_state"], state_scale)
        for i in range(batch_np["gt_command"].shape[0]):
            rows.append(
                {
                    "current_frame_id": int(batch_np["current_frame_id"][i]),
                    "target_frame_id": int(batch_np["target_frame_id"][i]),
                    "current_path": current_paths[i],
                    "target_path": target_paths[i],
                    "horizon": HORIZONS[int(batch_np["horizon_id"][i])],
                    "offset": int(batch_np["offset"][i]),
                    "gt_command": COMMANDS[int(batch_np["gt_command"][i])],
                    "pred_command": COMMANDS[int(batch_np["pred_command"][i])],
                    "gt_reached": float(batch_np["gt_reached"][i]),
                    "pred_reached_prob": float(batch_np["pred_reached_prob"][i]),
                    "gt_dx_to_target_m": float(gt_state["dx"][i]),
                    "gt_dy_to_target_m": float(gt_state["dy"][i]),
                    "pred_dx_to_target_m": float(pred_state["dx"][i]),
                    "pred_dy_to_target_m": float(pred_state["dy"][i]),
                    "gt_distance_m": float(gt_state["distance"][i]),
                    "pred_distance_m": float(pred_state["distance"][i]),
                    "gt_progress": float(gt_state["progress"][i]),
                    "pred_progress": float(pred_state["progress"][i]),
                    "gt_heading_error_deg": float(math.degrees(gt_state["heading_error"][i])),
                    "pred_heading_error_deg": float(math.degrees(pred_state["heading_error"][i])),
                }
            )

    final_arrays = {key: np.concatenate(value, axis=0) if value else np.array([]) for key, value in arrays.items()}
    metrics = compute_metrics(final_arrays, state_scale, reached_threshold)
    return metrics, rows, final_arrays


def safe_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def safe_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def safe_p90(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, 0.90)) if finite.size else float("nan")


def confusion_matrix_np(gt: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    mat = np.zeros((n_classes, n_classes), dtype=np.int64)
    for g, p in zip(gt, pred):
        mat[int(g), int(p)] += 1
    return mat


def per_class_metrics(confusion: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(COMMANDS):
        tp = float(confusion[idx, idx])
        fp = float(confusion[:, idx].sum() - confusion[idx, idx])
        fn = float(confusion[idx, :].sum() - confusion[idx, idx])
        support = float(confusion[idx, :].sum())
        precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
        recall = tp / (tp + fn) if tp + fn > 0 else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else float("nan")
        out[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return out


def state_metrics(prefix: str, gt_state: np.ndarray, pred_state: np.ndarray, scale: float) -> Dict[str, float]:
    gt = decode_state(gt_state, scale)
    pred = decode_state(pred_state, scale)
    rel_error = np.sqrt((pred["dx"] - gt["dx"]) ** 2 + (pred["dy"] - gt["dy"]) ** 2)
    distance_abs = np.abs(pred["distance"] - gt["distance"])
    progress_abs = np.abs(pred["progress"] - gt["progress"])
    heading_abs = angular_error_deg_np(pred["heading_error"], gt["heading_error"])
    return {
        f"{prefix}_dxdy_mean_m": safe_mean(rel_error),
        f"{prefix}_dxdy_median_m": safe_median(rel_error),
        f"{prefix}_dxdy_p90_m": safe_p90(rel_error),
        f"{prefix}_distance_mae_m": safe_mean(distance_abs),
        f"{prefix}_distance_median_abs_m": safe_median(distance_abs),
        f"{prefix}_progress_mae": safe_mean(progress_abs),
        f"{prefix}_heading_error_mean_deg": safe_mean(heading_abs),
        f"{prefix}_heading_error_median_deg": safe_median(heading_abs),
    }


def compute_metrics(arrays: Dict[str, np.ndarray], state_scale: float, reached_threshold: float) -> Dict[str, float]:
    gt_cmd = arrays["gt_command"].astype(np.int64)
    pred_cmd = arrays["pred_command"].astype(np.int64)
    confusion = confusion_matrix_np(gt_cmd, pred_cmd, len(COMMANDS))
    per_class = per_class_metrics(confusion)
    gt_reached = arrays["gt_reached"] > 0.5
    pred_reached = arrays["pred_reached_prob"] >= reached_threshold
    tp = float(np.logical_and(gt_reached, pred_reached).sum())
    fp = float(np.logical_and(~gt_reached, pred_reached).sum())
    fn = float(np.logical_and(gt_reached, ~pred_reached).sum())
    reached_precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
    reached_recall = tp / (tp + fn) if tp + fn > 0 else float("nan")
    reached_f1 = (
        2 * reached_precision * reached_recall / (reached_precision + reached_recall)
        if reached_precision + reached_recall > 0
        else float("nan")
    )
    metrics: Dict[str, float] = {
        "num_samples": float(gt_cmd.size),
        "command_accuracy": float(np.mean(gt_cmd == pred_cmd)) if gt_cmd.size else float("nan"),
        "macro_f1": float(np.nanmean([per_class[name]["f1"] for name in COMMANDS])),
        "balanced_accuracy": float(np.nanmean([per_class[name]["recall"] for name in COMMANDS])),
        "reached_accuracy": float(np.mean(gt_reached == pred_reached)) if gt_reached.size else float("nan"),
        "reached_precision": reached_precision,
        "reached_recall": reached_recall,
        "reached_f1": reached_f1,
    }
    for name in COMMANDS:
        metrics[f"{name.lower()}_precision"] = per_class[name]["precision"]
        metrics[f"{name.lower()}_recall"] = per_class[name]["recall"]
        metrics[f"{name.lower()}_f1"] = per_class[name]["f1"]
        metrics[f"{name.lower()}_support"] = per_class[name]["support"]
    metrics.update(state_metrics("target", arrays["gt_state"], arrays["pred_state"], state_scale))

    for horizon_id, horizon_name in enumerate(HORIZONS):
        mask = arrays["horizon_id"].astype(np.int64) == horizon_id
        if np.any(mask):
            metrics[f"{horizon_name}_samples"] = float(mask.sum())
            metrics[f"{horizon_name}_command_accuracy"] = float(np.mean(gt_cmd[mask] == pred_cmd[mask]))
            metrics[f"{horizon_name}_distance_mae_m"] = state_metrics(
                horizon_name, arrays["gt_state"][mask], arrays["pred_state"][mask], state_scale
            )[f"{horizon_name}_distance_mae_m"]
        else:
            metrics[f"{horizon_name}_samples"] = 0.0
            metrics[f"{horizon_name}_command_accuracy"] = float("nan")
            metrics[f"{horizon_name}_distance_mae_m"] = float("nan")
    return metrics


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_metrics_csv(path: Path, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, f"{value:.8f}" if isinstance(value, float) else value])


def save_predictions_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_loss_log(path: Path, history: Sequence[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_loss_curves(history: Sequence[Dict[str, float]], path: Path) -> Optional[str]:
    if plt is None or not history:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.plot(epochs, [row["train_loss"] for row in history], label="train")
    ax.plot(epochs, [row["val_loss"] for row in history], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss")
    ax.set_title("Multi-Horizon Visual Navigation Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_confusion(arrays: Dict[str, np.ndarray], path: Path) -> Optional[str]:
    if plt is None or arrays["gt_command"].size == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    mat = confusion_matrix_np(arrays["gt_command"], arrays["pred_command"], len(COMMANDS))
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(COMMANDS)), COMMANDS, rotation=35, ha="right")
    ax.set_yticks(range(len(COMMANDS)), COMMANDS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("Command Confusion Matrix")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_state_scatter(arrays: Dict[str, np.ndarray], state_scale: float, path: Path) -> Optional[str]:
    if plt is None or arrays["gt_state"].size == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    gt = decode_state(arrays["gt_state"], state_scale)
    pred = decode_state(arrays["pred_state"], state_scale)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    axes[0].scatter(gt["distance"], pred["distance"], s=8, alpha=0.35)
    lim = max(float(np.nanmax(gt["distance"])), float(np.nanmax(pred["distance"])), 1.0)
    axes[0].plot([0, lim], [0, lim], "r--", linewidth=1)
    axes[0].set_xlabel("GT distance (m)")
    axes[0].set_ylabel("Pred distance (m)")
    axes[0].set_title("Target Distance")
    axes[1].scatter(np.degrees(gt["heading_error"]), np.degrees(pred["heading_error"]), s=8, alpha=0.35)
    axes[1].plot([-180, 180], [-180, 180], "r--", linewidth=1)
    axes[1].set_xlabel("GT heading error (deg)")
    axes[1].set_ylabel("Pred heading error (deg)")
    axes[1].set_title("Heading Error")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_horizon_accuracy(metrics: Dict[str, float], path: Path) -> Optional[str]:
    if plt is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [metrics.get(f"{name}_command_accuracy", float("nan")) for name in HORIZONS]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    ax.bar(HORIZONS, values, color="#4c78a8")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Command accuracy")
    ax.set_title("Command Accuracy by Horizon")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def save_plots(run_dir: Path, history: Sequence[Dict[str, float]], arrays: Dict[str, np.ndarray], state_scale: float, metrics: Dict[str, float]) -> None:
    plot_dir = run_dir / "plots"
    plot_loss_curves(history, plot_dir / "loss_curve.png")
    plot_confusion(arrays, plot_dir / "command_confusion_matrix.png")
    plot_state_scatter(arrays, state_scale, plot_dir / "state_prediction_scatter.png")
    plot_horizon_accuracy(metrics, plot_dir / "horizon_command_accuracy.png")


def make_run_dir(output_root: str, run_name: str) -> Path:
    root = Path(output_root)
    if not run_name:
        run_name = "stage2a_multihorizon_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / run_name
    for sub in ["checkpoints", "metrics", "plots", "predictions"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def parse_offsets(value: str) -> List[int]:
    offsets = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not offsets:
        raise ValueError("--offsets must include at least one integer offset.")
    if any(offset <= 0 for offset in offsets):
        raise ValueError("--offsets must be positive.")
    return offsets


def update_horizon_names(offsets: Sequence[int]) -> None:
    global HORIZONS, HORIZON_OFFSETS, HORIZON_TO_ID
    HORIZON_OFFSETS = list(offsets)
    HORIZONS = [f"plus{offset}" for offset in offsets]
    HORIZON_TO_ID = {name: i for i, name in enumerate(HORIZONS)}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2A multi-horizon visual navigation.")
    parser.add_argument("--data_root", type=str, default=r"C:\Users\admin\Desktop\spiral_avgso\maps_screenshots_spiral_28150")
    parser.add_argument("--output_root", type=str, default=r"C:\Users\admin\Desktop\spiral_avgso\visual_navigation\results\stage2_multihorizon")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--offsets", type=str, default="1,4,10,40")
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--approx_m_per_frame", type=float, default=27.8)
    parser.add_argument("--max_offset_distance_factor", type=float, default=1.75)
    parser.add_argument("--max_offset_distance_extra_m", type=float, default=150.0)
    parser.add_argument("--stop_threshold_m", type=float, default=8.0)
    parser.add_argument("--forward_angle_deg", type=float, default=15.0)
    parser.add_argument("--turn_around_angle_deg", type=float, default=135.0)
    parser.add_argument("--train_frac", type=float, default=0.80)
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--split_gap", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--backbone", type=str, default="vgg16", choices=["vgg16", "vgg16_bn"])
    parser.add_argument("--use_pretrained", type=int, default=1)
    parser.add_argument("--freeze_backbone", type=int, default=0)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--huber_beta", type=float, default=0.1)
    parser.add_argument("--progress_every", type=int, default=0)
    parser.add_argument("--reached_threshold", type=float, default=0.5)

    parser.add_argument("--w_command", type=float, default=1.0)
    parser.add_argument("--w_state", type=float, default=0.45)
    parser.add_argument("--w_reached", type=float, default=0.30)
    parser.add_argument("--w_state_rel", type=float, default=1.0)
    parser.add_argument("--w_state_distance", type=float, default=0.75)
    parser.add_argument("--w_state_progress", type=float, default=0.25)
    parser.add_argument("--w_state_heading", type=float, default=0.50)

    parser.add_argument("--dry_run", type=int, default=0)
    parser.add_argument("--limit_train_samples", type=int, default=0)
    parser.add_argument("--limit_eval_samples", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    offsets = parse_offsets(args.offsets)
    update_horizon_names(offsets)
    set_seed(args.seed)
    run_dir = make_run_dir(args.output_root, args.run_name)
    save_json(run_dir / "config.json", vars(args))

    print(f"Data root: {args.data_root}")
    print(f"Run directory: {run_dir}")
    print(f"Offsets: {offsets}")
    records = list_frames_one_folder(args.data_root)
    ranges = split_ranges(len(records), args.train_frac, args.val_frac, args.split_gap)
    forward_angle_rad = math.radians(args.forward_angle_deg)
    turn_around_angle_rad = math.radians(args.turn_around_angle_deg)

    build_kwargs = dict(
        seq_len=args.seq_len,
        offsets=offsets,
        stop_threshold_m=args.stop_threshold_m,
        forward_angle_rad=forward_angle_rad,
        turn_around_angle_rad=turn_around_angle_rad,
        stride=args.sample_stride,
        approx_m_per_frame=args.approx_m_per_frame,
        max_offset_distance_factor=args.max_offset_distance_factor,
        max_offset_distance_extra_m=args.max_offset_distance_extra_m,
    )
    train_samples = build_samples_for_split(records, ranges["train"], **build_kwargs)
    val_samples = build_samples_for_split(records, ranges["val"], **build_kwargs)
    test_samples = build_samples_for_split(records, ranges["test"], **build_kwargs)
    train_samples = maybe_limit(train_samples, args.limit_train_samples, args.seed)
    val_samples = maybe_limit(val_samples, args.limit_eval_samples, args.seed + 1)
    test_samples = maybe_limit(test_samples, args.limit_eval_samples, args.seed + 2)

    max_distance_m = max(s.distance_m for s in train_samples)
    normalization = compute_normalization(train_samples, max_distance_m)
    train_counts = command_counts(train_samples)
    val_counts = command_counts(val_samples)
    test_counts = command_counts(test_samples)
    summary = {
        "num_frames": len(records),
        "split_ranges": {k: list(v) for k, v in ranges.items()},
        "offsets": offsets,
        "approx_distance_per_offset_m": {f"+{offset}": float(offset * args.approx_m_per_frame) for offset in offsets},
        "max_allowed_distance_per_offset_m": {
            f"+{offset}": float(
                max(
                    offset * args.approx_m_per_frame * args.max_offset_distance_factor,
                    offset * args.approx_m_per_frame + args.max_offset_distance_extra_m,
                )
            )
            for offset in offsets
        },
        "samples": {"train": len(train_samples), "val": len(val_samples), "test": len(test_samples)},
        "command_counts": {"train": train_counts, "val": val_counts, "test": test_counts},
        "horizon_counts": {
            "train": horizon_counts(train_samples),
            "val": horizon_counts(val_samples),
            "test": horizon_counts(test_samples),
        },
        "sample_stats": {
            "train": sample_stats(train_samples),
            "val": sample_stats(val_samples),
            "test": sample_stats(test_samples),
        },
        "horizon_stats": {
            "train": horizon_sample_stats(train_samples),
            "val": horizon_sample_stats(val_samples),
            "test": horizon_sample_stats(test_samples),
        },
        "normalization": normalization,
        "command_names": COMMANDS,
        "horizon_names": HORIZONS,
        "assumption": "Target image is current+horizon. Coordinates are labels only, not model inputs.",
    }
    save_json(run_dir / "metrics" / "dataset_summary.json", summary)
    save_json(run_dir / "metrics" / "normalization_stats.json", normalization)
    print(f"Frames: {len(records)}")
    print(f"Split ranges: {ranges}")
    print(f"Samples: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")
    print(f"Train command counts: {train_counts}")
    print(f"Train horizon counts: {horizon_counts(train_samples)}")
    print(f"Val command counts:   {val_counts}")
    print(f"Test command counts:  {test_counts}")

    if args.dry_run:
        print("Dry run complete. No model was trained.")
        return
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available in this Python environment. Use --dry_run 1 for metadata checks.")

    train_ds = MultiHorizonDataset(records, train_samples, normalization, args.seq_len, args.image_size)
    val_ds = MultiHorizonDataset(records, val_samples, normalization, args.seq_len, args.image_size)
    test_ds = MultiHorizonDataset(records, test_samples, normalization, args.seq_len, args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHorizonNavModel(
        seq_len=args.seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        layers=args.layers,
        dropout=args.dropout,
        backbone=args.backbone,
        use_pretrained=bool(args.use_pretrained),
        freeze_backbone=bool(args.freeze_backbone),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(bool(args.amp) and device.type == "cuda"))
    command_weights = class_weights_from_counts(train_counts).to(device)
    reached_pos_weight = torch.tensor(pos_weight_from_samples(train_samples), dtype=torch.float32, device=device)

    print(f"Device: {device}")
    print(f"Command loss weights: {tensor_to_numpy(command_weights)}")
    print(f"Reached positive weight: {float(reached_pos_weight.detach().cpu()):.4f}")
    print("No coordinates are model inputs; coordinates are labels only.")

    history: List[Dict[str, float]] = []
    best_val = float("inf")
    best_path = run_dir / "checkpoints" / "best.pt"
    live_progress_path = run_dir / "metrics" / "live_progress.txt"

    for epoch in range(1, args.epochs + 1):
        train_logs = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            command_weights,
            reached_pos_weight,
            args,
            epoch,
            args.epochs,
            live_progress_path,
        )
        val_logs = evaluate_loss(model, val_loader, device, command_weights, reached_pos_weight, args)
        scheduler.step()
        row: Dict[str, float] = {"epoch": float(epoch)}
        row.update({f"train_{k}": v for k, v in train_logs.items()})
        row.update({f"val_{k}": v for k, v in val_logs.items()})
        row["train_loss"] = train_logs["loss"]
        row["val_loss"] = val_logs["loss"]
        history.append(row)
        save_loss_log(run_dir / "metrics" / "loss_log.csv", history)
        print(
            f"Epoch {epoch:03d} | train={train_logs['loss']:.4f} | val={val_logs['loss']:.4f} | "
            f"cmd={val_logs['command']:.4f} state={val_logs['state']:.4f}",
            flush=True,
        )
        write_live_progress(
            live_progress_path,
            f"Finished epoch {epoch:03d}/{args.epochs:03d}: train={train_logs['loss']:.4f}, val={val_logs['loss']:.4f}",
        )
        if val_logs["loss"] < best_val:
            best_val = val_logs["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": vars(args),
                    "normalization": normalization,
                    "best_val_loss": best_val,
                },
                best_path,
            )

    print(f"Loading best checkpoint from {best_path}")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    metrics, rows, arrays = evaluate_full(
        model,
        test_loader,
        device,
        float(normalization["state_distance_scale"]),
        args.reached_threshold,
    )
    test_logs = evaluate_loss(model, test_loader, device, command_weights, reached_pos_weight, args)
    metrics["best_val_loss"] = float(best_val)
    metrics["test_loss"] = test_logs["loss"]
    save_metrics_csv(run_dir / "metrics" / "final_navigation_metrics.csv", metrics)
    save_predictions_csv(run_dir / "predictions" / "test_per_sample_predictions.csv", rows)
    save_plots(run_dir, history, arrays, float(normalization["state_distance_scale"]), metrics)
    write_live_progress(live_progress_path, "Finished final evaluation.")
    print(f"Saved metrics to {run_dir / 'metrics' / 'final_navigation_metrics.csv'}")
    print(f"Saved plots to {run_dir / 'plots'}")
    print(f"Best val loss: {best_val:.4f} | Test loss: {metrics['test_loss']:.4f}")
    print(
        f"Command acc: {metrics['command_accuracy']:.4f} | "
        f"Balanced acc: {metrics['balanced_accuracy']:.4f} | "
        f"Target distance MAE: {metrics['target_distance_mae_m']:.2f} m"
    )


if __name__ == "__main__":
    main()
