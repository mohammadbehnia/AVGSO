# -*- coding: utf-8 -*-
"""
Stage 1 goal-conditioned visual navigation for AVGSO spiral data.

Task:
  Given a short current visual sequence and a target-view image, predict direct
  target-relative navigation for the immediate distance horizon.

Default horizon:
  Immediate/local target distance: 15-80 m. This includes the one-frame
  spiral spacing of approximately 27.8 m/frame and the nearby arm-to-arm
  neighborhood of approximately 66.8 m.

Inputs:
  current sequence: I[t-seq_len+1], ..., I[t]
  target image:     I[j]
  horizon token:    immediate

Main outputs:
  - target direction and target distance, converted to dx, dy in local meters
  - command logits for STOP, LEFT, RIGHT, FORWARD, TURN_AROUND

Geometry-derived analysis:
  - target direction from predicted dx, dy
  - heading error relative to predicted/ground-truth current heading

Auxiliary outputs:
  - current pose estimate x, y, heading
  - optional next-frame prediction I[t+1] for visual motion consistency

The spiral route is used as a set of geo-referenced visual places, not as a
trajectory to imitate. Target frames are selected by direct Euclidean distance
from the current frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

        @staticmethod
        def no_grad():
            return _NoGrad()

    class _NNStub:
        Module = object

    torch = _TorchStub()
    nn = _NNStub()
    F = None
    DataLoader = None
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
HORIZON_TO_ID = {"immediate": 0, "short": 1, "medium": 2, "long": 3}


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
class NavSample:
    anchor_idx: int
    target_idx: int
    horizon_id: int
    command_id: int
    dx: float
    dy: float
    distance_m: float
    desired_heading_rad: float
    heading_error_rad: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def heading_to_rad(value: float) -> float:
    if abs(value) > 6.5:
        return math.radians(value)
    return value


def wrap_angle_rad(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def wrap_angle_tensor(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


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
    """
    Parse current AVGSO names:
      frame_<id>_<lat>_<lon>_<heading>.<ext>

    Also accepts prefixed names such as:
      2_frame_<id>_<lat>_<lon>_<heading>.<ext>
    """
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
    if len(parsed) < 8:
        raise RuntimeError(
            f"Found only {len(parsed)} valid frames in {root}. "
            "Expected names like frame_<id>_<lat>_<lon>_<heading>.<ext>."
        )

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
            raise RuntimeError(
                f"Empty {name} split after applying split_gap={split_gap}: start={start}, end={end}."
            )
    return ranges


def cell_key(x: float, y: float, cell_size: float) -> Tuple[int, int]:
    return int(math.floor(x / cell_size)), int(math.floor(y / cell_size))


def build_spatial_index(records: Sequence[FrameRecord], start: int, end: int, cell_size: float) -> Dict[Tuple[int, int], List[int]]:
    grid: Dict[Tuple[int, int], List[int]] = {}
    for idx in range(start, end):
        rec = records[idx]
        key = cell_key(rec.x, rec.y, cell_size)
        grid.setdefault(key, []).append(idx)
    return grid


def candidate_indices_near(
    records: Sequence[FrameRecord],
    grid: Dict[Tuple[int, int], List[int]],
    anchor_idx: int,
    cell_size: float,
    max_distance_m: float,
) -> Iterable[int]:
    anchor = records[anchor_idx]
    cx, cy = cell_key(anchor.x, anchor.y, cell_size)
    radius_cells = int(math.ceil(max_distance_m / cell_size)) + 1
    for gx in range(cx - radius_cells, cx + radius_cells + 1):
        for gy in range(cy - radius_cells, cy + radius_cells + 1):
            for idx in grid.get((gx, gy), []):
                yield idx


def command_from_geometry(
    dx: float,
    dy: float,
    current_heading_rad: float,
    stop_threshold_m: float,
    forward_angle_rad: float,
    turn_around_angle_rad: float,
) -> Tuple[int, float, float, float]:
    distance = math.hypot(dx, dy)
    if distance <= stop_threshold_m:
        desired = current_heading_rad
        heading_error = 0.0
        return COMMAND_TO_ID["STOP"], desired, heading_error, distance

    # Compass-style bearing: 0 rad is north, positive is clockwise toward east.
    desired = math.atan2(dx, dy)
    heading_error = wrap_angle_rad(desired - current_heading_rad)
    abs_error = abs(heading_error)
    if abs_error <= forward_angle_rad:
        command = COMMAND_TO_ID["FORWARD"]
    elif abs_error >= turn_around_angle_rad:
        command = COMMAND_TO_ID["TURN_AROUND"]
    elif heading_error < 0:
        command = COMMAND_TO_ID["LEFT"]
    else:
        command = COMMAND_TO_ID["RIGHT"]
    return command, desired, heading_error, distance


def build_samples_for_split(
    records: Sequence[FrameRecord],
    split_range: Tuple[int, int],
    seq_len: int,
    horizon_id: int,
    distance_min_m: float,
    distance_max_m: float,
    targets_per_anchor: int,
    include_stop: bool,
    stop_sample_ratio: float,
    stop_threshold_m: float,
    forward_angle_rad: float,
    turn_around_angle_rad: float,
    seed: int,
) -> List[NavSample]:
    start, end = split_range
    if end - start <= seq_len + 2:
        raise RuntimeError(f"Split too small for seq_len={seq_len}: start={start}, end={end}")

    rng = random.Random(seed)
    cell_size = max(distance_max_m, stop_threshold_m, 1.0)
    grid = build_spatial_index(records, start, end, cell_size)
    samples: List[NavSample] = []

    anchor_start = start + seq_len - 1
    anchor_end = end - 1  # anchor needs t+1 for next-frame auxiliary target

    for anchor_idx in range(anchor_start, anchor_end):
        anchor = records[anchor_idx]
        context_start = anchor_idx - seq_len + 1
        context_indices = set(range(context_start, anchor_idx + 1))

        if include_stop and rng.random() < stop_sample_ratio:
            samples.append(
                NavSample(
                    anchor_idx=anchor_idx,
                    target_idx=anchor_idx,
                    horizon_id=horizon_id,
                    command_id=COMMAND_TO_ID["STOP"],
                    dx=0.0,
                    dy=0.0,
                    distance_m=0.0,
                    desired_heading_rad=anchor.heading_rad,
                    heading_error_rad=0.0,
                )
            )

        candidates: List[NavSample] = []
        for target_idx in candidate_indices_near(records, grid, anchor_idx, cell_size, distance_max_m):
            if target_idx == anchor_idx:
                continue
            if target_idx in context_indices:
                # Avoid giving the model a target image that is already part of
                # the current observation history.
                continue
            target = records[target_idx]
            dx = target.x - anchor.x
            dy = target.y - anchor.y
            command, desired, heading_error, distance = command_from_geometry(
                dx, dy, anchor.heading_rad, stop_threshold_m, forward_angle_rad, turn_around_angle_rad
            )
            if distance_min_m <= distance <= distance_max_m:
                candidates.append(
                    NavSample(
                        anchor_idx=anchor_idx,
                        target_idx=target_idx,
                        horizon_id=horizon_id,
                        command_id=command,
                        dx=dx,
                        dy=dy,
                        distance_m=distance,
                        desired_heading_rad=desired,
                        heading_error_rad=heading_error,
                    )
                )

        if candidates:
            rng.shuffle(candidates)
            samples.extend(candidates[:targets_per_anchor])

    if not samples:
        raise RuntimeError(
            "No navigation samples were built. Check distance_min_m, distance_max_m, "
            "split ranges, and filename parsing."
        )
    return samples


def compute_normalization(records: Sequence[FrameRecord], samples: Sequence[NavSample]) -> Dict[str, List[float]]:
    rel = np.array([[s.dx, s.dy] for s in samples], dtype=np.float32)
    xy = np.array([[records[s.anchor_idx].x, records[s.anchor_idx].y] for s in samples], dtype=np.float32)

    rel_mean = rel.mean(axis=0)
    rel_std = np.maximum(rel.std(axis=0), 1.0)
    xy_mean = xy.mean(axis=0)
    xy_std = np.maximum(xy.std(axis=0), 1.0)

    return {
        "rel_mean": rel_mean.tolist(),
        "rel_std": rel_std.tolist(),
        "xy_mean": xy_mean.tolist(),
        "xy_std": xy_std.tolist(),
    }


def command_counts(samples: Sequence[NavSample]) -> Dict[str, int]:
    counts = {name: 0 for name in COMMANDS}
    for sample in samples:
        counts[COMMANDS[sample.command_id]] += 1
    return counts


def class_weights_from_counts(counts: Dict[str, int]) -> torch.Tensor:
    values = np.array([counts[name] for name in COMMANDS], dtype=np.float32)
    values = np.maximum(values, 1.0)
    weights = values.sum() / (len(values) * values)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class GoalNavigationDataset(Dataset):
    def __init__(
        self,
        records: Sequence[FrameRecord],
        samples: Sequence[NavSample],
        normalization: Dict[str, List[float]],
        seq_len: int,
        image_size: int,
        use_next_frame_aux: bool = False,
    ):
        self.records = records
        self.samples = list(samples)
        self.seq_len = seq_len
        self.use_next_frame_aux = use_next_frame_aux
        self.rel_mean = torch.tensor(normalization["rel_mean"], dtype=torch.float32)
        self.rel_std = torch.tensor(normalization["rel_std"], dtype=torch.float32)
        self.xy_mean = torch.tensor(normalization["xy_mean"], dtype=torch.float32)
        self.xy_std = torch.tensor(normalization["xy_std"], dtype=torch.float32)

        self.raw_tf = None
        if self.use_next_frame_aux:
            self.raw_tf = T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.ToTensor(),
                ]
            )
        self.norm_tf = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_raw(self, path: str) -> torch.Tensor:
        if self.raw_tf is None:
            raise RuntimeError("Raw image loading requested while next-frame auxiliary task is disabled.")
        with Image.open(path) as img:
            return self.raw_tf(img.convert("RGB"))

    def _load_norm(self, path: str) -> torch.Tensor:
        with Image.open(path) as img:
            return self.norm_tf(img.convert("RGB"))

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        anchor = self.records[sample.anchor_idx]
        target = self.records[sample.target_idx]

        seq_start = sample.anchor_idx - self.seq_len + 1
        seq_indices = list(range(seq_start, sample.anchor_idx + 1))
        current_images = torch.stack([self._load_norm(self.records[i].path) for i in seq_indices], dim=0)

        target_image = self._load_norm(target.path)

        rel = torch.tensor([sample.dx, sample.dy], dtype=torch.float32)
        rel_norm = (rel - self.rel_mean) / self.rel_std

        current_xy = torch.tensor([anchor.x, anchor.y], dtype=torch.float32)
        current_xy_norm = (current_xy - self.xy_mean) / self.xy_std
        current_heading = torch.tensor(anchor.heading_rad, dtype=torch.float32)
        current_heading_sincos = torch.stack([torch.sin(current_heading), torch.cos(current_heading)])

        item = {
            "current_images": current_images,
            "target_image": target_image,
            "gt_rel": rel,
            "gt_rel_norm": rel_norm,
            "gt_current_xy": current_xy,
            "gt_current_xy_norm": current_xy_norm,
            "gt_current_heading": current_heading,
            "gt_current_heading_sincos": current_heading_sincos,
            "gt_bearing": torch.tensor(sample.desired_heading_rad, dtype=torch.float32),
            "gt_heading_error": torch.tensor(sample.heading_error_rad, dtype=torch.float32),
            "gt_distance": torch.tensor(sample.distance_m, dtype=torch.float32),
            "gt_command": torch.tensor(sample.command_id, dtype=torch.long),
            "horizon_id": torch.tensor(sample.horizon_id, dtype=torch.long),
            "anchor_idx": torch.tensor(sample.anchor_idx, dtype=torch.long),
            "target_idx": torch.tensor(sample.target_idx, dtype=torch.long),
            "anchor_frame_id": torch.tensor(anchor.frame_id, dtype=torch.long),
            "target_frame_id": torch.tensor(target.frame_id, dtype=torch.long),
            "anchor_path": Path(anchor.path).name,
            "target_path": Path(target.path).name,
        }
        if self.use_next_frame_aux:
            next_rec = self.records[sample.anchor_idx + 1]
            item.update(
                {
                    "next_frame": self._load_raw(next_rec.path),
                    "last_frame": self._load_raw(anchor.path),
                    "target_frame": self._load_raw(target.path),
                }
            )
        return item


class VGGTokenStem(nn.Module):
    def __init__(self, backbone: str, d_model: int, use_pretrained: bool):
        super().__init__()
        backbone = backbone.lower()
        weights = None
        if use_pretrained:
            try:
                if backbone == "vgg16_bn":
                    weights = VGG16_BN_Weights.IMAGENET1K_V1
                elif backbone == "vgg16":
                    weights = VGG16_Weights.IMAGENET1K_V1
            except Exception as exc:
                print(f"Warning: could not prepare pretrained weights ({exc}); using random initialization.")
                weights = None

        try:
            if backbone == "vgg16_bn":
                base = vgg16_bn(weights=weights)
            elif backbone == "vgg16":
                base = vgg16(weights=weights)
            else:
                raise ValueError("backbone must be 'vgg16' or 'vgg16_bn'")
        except Exception as exc:
            if use_pretrained:
                print(f"Warning: pretrained backbone load failed ({exc}); retrying without pretrained weights.")
                if backbone == "vgg16_bn":
                    base = vgg16_bn(weights=None)
                else:
                    base = vgg16(weights=None)
            else:
                raise

        self.features = base.features
        self.proj = nn.Linear(512, d_model)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        feat = self.features(images)
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.proj(tokens)
        return tokens, h, w


class GoalConditionedTransNav(nn.Module):
    def __init__(
        self,
        seq_len: int,
        image_size: int,
        d_model: int,
        nhead: int,
        layers: int,
        dropout: float,
        backbone: str,
        use_pretrained: bool,
        use_next_frame_aux: bool,
        rel_output_mode: str,
        distance_scale_m: float,
        use_pairwise_fusion: bool,
        max_tokens: int = 1024,
        num_horizons: int = 4,
        num_commands: int = len(COMMANDS),
    ):
        super().__init__()
        self.seq_len = seq_len
        self.image_size = image_size
        self.d_model = d_model
        self.max_tokens = max_tokens
        self.use_next_frame_aux = use_next_frame_aux
        self.rel_output_mode = rel_output_mode
        self.distance_scale_m = max(float(distance_scale_m), 1.0)
        self.use_pairwise_fusion = use_pairwise_fusion

        self.stem = VGGTokenStem(backbone=backbone, d_model=d_model, use_pretrained=use_pretrained)

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

        self.target_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.horizon_emb = nn.Embedding(num_horizons, d_model)
        fusion_dim = d_model * (6 if self.use_pairwise_fusion else 3)
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.rel_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )
        self.direction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )
        self.distance_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        self.command_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_commands),
        )
        self.pose_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 4),  # current x,y normalized + heading sin,cos
        )

        self.token_decoder = None
        self.reconstruction_head = None
        if self.use_next_frame_aux:
            self.token_decoder = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.reconstruction_head = nn.Sequential(
                nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(d_model // 2, 3, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )

    def _add_current_pos(self, tokens: torch.Tensor, t: int, l_tokens: int) -> torch.Tensor:
        b, _, d = tokens.shape
        time_ids = torch.arange(t, device=tokens.device).repeat_interleave(l_tokens)
        space_ids = torch.arange(l_tokens, device=tokens.device).repeat(t)
        pos = self.time_emb(time_ids)[None, :, :] + self.space_emb(space_ids)[None, :, :]
        return tokens + pos + self.current_segment

    def _add_target_pos(self, tokens: torch.Tensor, l_tokens: int) -> torch.Tensor:
        space_ids = torch.arange(l_tokens, device=tokens.device)
        pos = self.space_emb(space_ids)[None, :, :]
        return tokens + pos + self.target_segment

    def forward(self, current_images: torch.Tensor, target_image: torch.Tensor, horizon_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, t, c, h_img, w_img = current_images.shape
        if t != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {t}")

        flat = current_images.reshape(b * t, c, h_img, w_img)
        cur_tokens_bt, h_feat, w_feat = self.stem(flat)
        l_tokens = cur_tokens_bt.size(1)
        if l_tokens > self.max_tokens:
            raise RuntimeError(f"Feature tokens {l_tokens} exceed max_tokens={self.max_tokens}")

        cur_tokens = cur_tokens_bt.reshape(b, t * l_tokens, self.d_model)
        cur_tokens = self._add_current_pos(cur_tokens, t, l_tokens)
        cls = self.cls.expand(b, -1, -1)
        cur_encoded = self.encoder(torch.cat([cls, cur_tokens], dim=1))
        current_repr = cur_encoded[:, 0]

        target_tokens, _, _ = self.stem(target_image)
        target_tokens = self._add_target_pos(target_tokens, target_tokens.size(1))
        target_repr = self.target_mlp(target_tokens.mean(dim=1))

        h_repr = self.horizon_emb(horizon_id)
        if self.use_pairwise_fusion:
            fusion_input = torch.cat(
                [
                    current_repr,
                    target_repr,
                    current_repr - target_repr,
                    torch.abs(current_repr - target_repr),
                    current_repr * target_repr,
                    h_repr,
                ],
                dim=-1,
            )
        else:
            fusion_input = torch.cat([current_repr, target_repr, h_repr], dim=-1)
        fused = self.fusion(fusion_input)

        result: Dict[str, torch.Tensor]
        if self.rel_output_mode == "polar":
            direction_unit = F.normalize(self.direction_head(fused), dim=-1, eps=1e-6)
            distance_unit = torch.sigmoid(self.distance_head(fused).squeeze(-1))
            pred_distance_m = distance_unit * self.distance_scale_m
            rel_m = direction_unit * pred_distance_m.unsqueeze(-1)
            result = {
                "rel_m": rel_m,
                "pred_direction_unit": direction_unit,
                "pred_distance_m": pred_distance_m,
            }
        else:
            rel_norm = self.rel_head(fused)
            result = {"rel_norm": rel_norm}

        command_logits = self.command_head(fused)
        pose_raw = self.pose_head(current_repr)
        current_xy_norm = pose_raw[:, :2]
        current_heading_sincos = F.normalize(pose_raw[:, 2:], dim=-1)

        result.update(
            {
                "command_logits": command_logits,
                "current_xy_norm": current_xy_norm,
                "current_heading_sincos": current_heading_sincos,
            }
        )
        if self.use_next_frame_aux:
            # Tokens corresponding to the last frame in the current sequence.
            last_start = 1 + (t - 1) * l_tokens
            last_end = 1 + t * l_tokens
            last_tokens = cur_encoded[:, last_start:last_end, :]
            decoded = self.token_decoder(last_tokens)
            feat = decoded.transpose(1, 2).reshape(b, self.d_model, h_feat, w_feat)
            feat_up = F.interpolate(feat, size=(h_img, w_img), mode="bilinear", align_corners=False)
            result["next_frame"] = self.reconstruction_head(feat_up)
        return result


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved: Dict[str, object] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def circular_sincos_loss(pred_sincos: torch.Tensor, gt_sincos: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred_sincos, dim=-1)
    gt = F.normalize(gt_sincos, dim=-1)
    cos_sim = (pred * gt).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cos_sim).mean()


def predicted_relative_m(out: Dict[str, torch.Tensor], rel_mean: torch.Tensor, rel_std: torch.Tensor) -> torch.Tensor:
    if "rel_m" in out:
        return out["rel_m"]
    return out["rel_norm"] * rel_std + rel_mean


def target_direction_loss_from_relative(
    pred_rel: torch.Tensor,
    gt_bearing: torch.Tensor,
    gt_distance: torch.Tensor,
    stop_threshold_m: float,
) -> torch.Tensor:
    pred_bearing = torch.atan2(pred_rel[:, 0], pred_rel[:, 1])
    mask = gt_distance > stop_threshold_m
    if not torch.any(mask):
        return pred_rel.new_tensor(0.0)
    diff = wrap_angle_tensor(pred_bearing[mask] - gt_bearing[mask])
    return (1.0 - torch.cos(diff)).mean()


def total_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    rel_mean: torch.Tensor,
    rel_std: torch.Tensor,
    command_weights: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred_rel = predicted_relative_m(out, rel_mean, rel_std)
    pred_rel_norm = (pred_rel - rel_mean) / rel_std
    pred_distance = torch.linalg.norm(pred_rel, dim=-1)
    distance_scale = max(float(args.distance_max_m), 1.0)

    loss_rel = F.smooth_l1_loss(pred_rel_norm, batch["gt_rel_norm"], beta=args.huber_beta)
    loss_bearing = target_direction_loss_from_relative(
        pred_rel,
        batch["gt_bearing"],
        batch["gt_distance"],
        args.stop_threshold_m,
    )
    loss_distance = F.smooth_l1_loss(
        pred_distance / distance_scale,
        batch["gt_distance"] / distance_scale,
        beta=args.distance_huber_beta,
    )
    loss_cmd = F.cross_entropy(out["command_logits"], batch["gt_command"], weight=command_weights)
    loss_pose_xy = F.smooth_l1_loss(out["current_xy_norm"], batch["gt_current_xy_norm"], beta=args.huber_beta)
    loss_pose_head = circular_sincos_loss(out["current_heading_sincos"], batch["gt_current_heading_sincos"])
    if args.use_next_frame_aux:
        loss_img = F.l1_loss(out["next_frame"], batch["next_frame"])
    else:
        loss_img = loss_rel.new_tensor(0.0)

    loss = (
        args.w_rel * loss_rel
        + args.w_bearing * loss_bearing
        + args.w_distance * loss_distance
        + args.w_command * loss_cmd
        + args.w_pose_xy * loss_pose_xy
        + args.w_pose_heading * loss_pose_head
        + (args.w_next_frame if args.use_next_frame_aux else 0.0) * loss_img
    )

    logs = {
        "loss": float(loss.detach().cpu()),
        "rel": float(loss_rel.detach().cpu()),
        "bearing": float(loss_bearing.detach().cpu()),
        "distance": float(loss_distance.detach().cpu()),
        "command": float(loss_cmd.detach().cpu()),
        "pose_xy": float(loss_pose_xy.detach().cpu()),
        "pose_heading": float(loss_pose_head.detach().cpu()),
        "next_frame": float(loss_img.detach().cpu()),
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
    rel_mean: torch.Tensor,
    rel_std: torch.Tensor,
    command_weights: torch.Tensor,
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
        with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
            out = model(batch["current_images"], batch["target_image"], batch["horizon_id"])
            loss, logs = total_loss(out, batch, rel_mean, rel_std, command_weights, args)

        if not torch.isfinite(loss):
            print("Non-finite loss detected.")
            print("anchor_frame_id:", batch["anchor_frame_id"])
            print("target_frame_id:", batch["target_frame_id"])
            raise RuntimeError("Stopping because training loss became NaN/Inf.")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        bs = batch["current_images"].size(0)
        average_logs(totals, logs, bs)
        count += bs

        should_print = (
            args.progress_every > 0
            and (batch_idx == 1 or batch_idx == total_batches or batch_idx % args.progress_every == 0)
        )
        if should_print:
            running = finalize_logs(totals, count)
            elapsed = max(time.time() - start_time, 1e-6)
            batches_per_sec = batch_idx / elapsed
            remaining = (total_batches - batch_idx) / max(batches_per_sec, 1e-6)
            message = (
                f"Epoch {epoch:03d}/{total_epochs:03d} "
                f"batch {batch_idx:05d}/{total_batches:05d} "
                f"samples {count:06d} "
                f"loss={running['loss']:.4f} rel={running['rel']:.4f} "
                f"dist={running['distance']:.4f} cmd={running['command']:.4f} img={running['next_frame']:.4f} "
                f"elapsed={elapsed/60:.1f}m eta={remaining/60:.1f}m"
            )
            print(message, flush=True)
            write_live_progress(live_progress_path, message)

    return finalize_logs(totals, count)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    rel_mean: torch.Tensor,
    rel_std: torch.Tensor,
    command_weights: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch["current_images"], batch["target_image"], batch["horizon_id"])
        _, logs = total_loss(out, batch, rel_mean, rel_std, command_weights, args)
        bs = batch["current_images"].size(0)
        average_logs(totals, logs, bs)
        count += bs
    return finalize_logs(totals, count)


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def commands_from_geometry_np(
    dx: np.ndarray,
    dy: np.ndarray,
    current_heading: np.ndarray,
    stop_threshold_m: float,
    forward_angle_rad: float,
    turn_around_angle_rad: float,
) -> np.ndarray:
    distance = np.sqrt(dx * dx + dy * dy)
    desired = np.arctan2(dx, dy)
    heading_error = np.arctan2(np.sin(desired - current_heading), np.cos(desired - current_heading))
    out = np.full(distance.shape, COMMAND_TO_ID["FORWARD"], dtype=np.int64)
    out[distance <= stop_threshold_m] = COMMAND_TO_ID["STOP"]
    non_stop = distance > stop_threshold_m
    turn_around = np.logical_and(non_stop, np.abs(heading_error) >= turn_around_angle_rad)
    out[turn_around] = COMMAND_TO_ID["TURN_AROUND"]
    out[np.logical_and.reduce((non_stop, ~turn_around, heading_error < -forward_angle_rad))] = COMMAND_TO_ID["LEFT"]
    out[np.logical_and.reduce((non_stop, ~turn_around, heading_error > forward_angle_rad))] = COMMAND_TO_ID["RIGHT"]
    return out


def global_ssim_np(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    # Lightweight global SSIM per image, used only for summary diagnostics.
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    pred_flat = pred.reshape(pred.shape[0], -1)
    gt_flat = gt.reshape(gt.shape[0], -1)
    mu_x = pred_flat.mean(axis=1)
    mu_y = gt_flat.mean(axis=1)
    var_x = pred_flat.var(axis=1)
    var_y = gt_flat.var(axis=1)
    cov = ((pred_flat - mu_x[:, None]) * (gt_flat - mu_y[:, None])).mean(axis=1)
    return ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))


@torch.no_grad()
def evaluate_full(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    normalization: Dict[str, List[float]],
    args: argparse.Namespace,
    example_count: int = 8,
) -> Tuple[Dict[str, float], List[Dict[str, object]], Dict[str, np.ndarray]]:
    model.eval()
    rel_mean = torch.tensor(normalization["rel_mean"], dtype=torch.float32, device=device)
    rel_std = torch.tensor(normalization["rel_std"], dtype=torch.float32, device=device)
    xy_mean = torch.tensor(normalization["xy_mean"], dtype=torch.float32, device=device)
    xy_std = torch.tensor(normalization["xy_std"], dtype=torch.float32, device=device)

    rows: List[Dict[str, object]] = []
    arrays: Dict[str, List[np.ndarray]] = {
        "gt_rel": [],
        "pred_rel": [],
        "rel_error": [],
        "gt_bearing": [],
        "pred_bearing": [],
        "bearing_error_deg": [],
        "gt_command": [],
        "pred_command": [],
        "geom_command": [],
        "gt_current_xy": [],
        "pred_current_xy": [],
        "current_pose_error": [],
        "gt_current_heading": [],
        "pred_current_heading": [],
        "current_heading_error_deg": [],
        "pred_distance": [],
        "gt_distance": [],
        "next_l1": [],
        "next_psnr": [],
        "next_ssim": [],
        "anchor_frame_id": [],
        "target_frame_id": [],
    }
    examples: Dict[str, List[np.ndarray]] = {"last": [], "target": [], "gt_next": [], "pred_next": []}

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch["current_images"], batch["target_image"], batch["horizon_id"])

        pred_rel = predicted_relative_m(out, rel_mean, rel_std)
        gt_rel = batch["gt_rel"]
        rel_error = torch.linalg.norm(pred_rel - gt_rel, dim=-1)
        pred_distance = torch.linalg.norm(pred_rel, dim=-1)

        pred_bearing = torch.atan2(pred_rel[:, 0], pred_rel[:, 1])
        gt_bearing = batch["gt_bearing"]
        bearing_error = torch.rad2deg(torch.abs(wrap_angle_tensor(pred_bearing - gt_bearing)))
        bearing_error = torch.where(
            batch["gt_distance"] > args.stop_threshold_m,
            bearing_error,
            torch.zeros_like(bearing_error),
        )

        pred_cmd = torch.argmax(out["command_logits"], dim=-1)
        gt_cmd = batch["gt_command"]

        pred_xy = out["current_xy_norm"] * xy_std + xy_mean
        gt_xy = batch["gt_current_xy"]
        pose_error = torch.linalg.norm(pred_xy - gt_xy, dim=-1)

        pred_heading = torch.atan2(out["current_heading_sincos"][:, 0], out["current_heading_sincos"][:, 1])
        gt_heading = batch["gt_current_heading"]
        heading_error_deg = torch.rad2deg(torch.abs(wrap_angle_tensor(pred_heading - gt_heading)))

        bs = batch["current_images"].size(0)
        if args.use_next_frame_aux:
            next_abs = torch.abs(out["next_frame"] - batch["next_frame"])
            next_l1 = next_abs.flatten(1).mean(dim=1)
            next_mse = ((out["next_frame"] - batch["next_frame"]) ** 2).flatten(1).mean(dim=1).clamp_min(1e-10)
            next_psnr = -10.0 * torch.log10(next_mse)

            pred_np_img = tensor_to_numpy(out["next_frame"].clamp(0, 1))
            gt_np_img = tensor_to_numpy(batch["next_frame"].clamp(0, 1))
            next_ssim = global_ssim_np(pred_np_img, gt_np_img)
        else:
            pred_np_img = np.array([])
            next_l1 = torch.full((bs,), float("nan"), device=device)
            next_psnr = torch.full((bs,), float("nan"), device=device)
            next_ssim = np.full((bs,), np.nan, dtype=np.float32)

        pred_rel_np = tensor_to_numpy(pred_rel)
        pred_heading_np = tensor_to_numpy(pred_heading)
        geom_cmd_np = commands_from_geometry_np(
            pred_rel_np[:, 0],
            pred_rel_np[:, 1],
            pred_heading_np,
            args.stop_threshold_m,
            math.radians(args.forward_angle_deg),
            math.radians(args.turn_around_angle_deg),
        )

        batch_np = {
            "gt_rel": tensor_to_numpy(gt_rel),
            "pred_rel": pred_rel_np,
            "rel_error": tensor_to_numpy(rel_error),
            "gt_bearing": tensor_to_numpy(gt_bearing),
            "pred_bearing": tensor_to_numpy(pred_bearing),
            "bearing_error_deg": tensor_to_numpy(bearing_error),
            "gt_command": tensor_to_numpy(gt_cmd),
            "pred_command": tensor_to_numpy(pred_cmd),
            "geom_command": geom_cmd_np,
            "gt_current_xy": tensor_to_numpy(gt_xy),
            "pred_current_xy": tensor_to_numpy(pred_xy),
            "current_pose_error": tensor_to_numpy(pose_error),
            "gt_current_heading": tensor_to_numpy(gt_heading),
            "pred_current_heading": pred_heading_np,
            "current_heading_error_deg": tensor_to_numpy(heading_error_deg),
            "pred_distance": tensor_to_numpy(pred_distance),
            "gt_distance": tensor_to_numpy(batch["gt_distance"]),
            "next_l1": tensor_to_numpy(next_l1),
            "next_psnr": tensor_to_numpy(next_psnr),
            "next_ssim": next_ssim,
            "anchor_frame_id": tensor_to_numpy(batch["anchor_frame_id"]),
            "target_frame_id": tensor_to_numpy(batch["target_frame_id"]),
        }

        for key, value in batch_np.items():
            arrays[key].append(np.asarray(value))

        anchor_paths = batch["anchor_path"]
        target_paths = batch["target_path"]
        for i in range(bs):
            rows.append(
                {
                    "anchor_frame_id": int(batch_np["anchor_frame_id"][i]),
                    "target_frame_id": int(batch_np["target_frame_id"][i]),
                    "anchor_path": anchor_paths[i],
                    "target_path": target_paths[i],
                    "gt_command": COMMANDS[int(batch_np["gt_command"][i])],
                    "pred_command": COMMANDS[int(batch_np["pred_command"][i])],
                    "geometry_command_from_prediction": COMMANDS[int(batch_np["geom_command"][i])],
                    "gt_dx_m": float(batch_np["gt_rel"][i, 0]),
                    "gt_dy_m": float(batch_np["gt_rel"][i, 1]),
                    "pred_dx_m": float(batch_np["pred_rel"][i, 0]),
                    "pred_dy_m": float(batch_np["pred_rel"][i, 1]),
                    "relative_error_m": float(batch_np["rel_error"][i]),
                    "target_distance_m": float(batch_np["gt_distance"][i]),
                    "pred_target_distance_m": float(batch_np["pred_distance"][i]),
                    "gt_bearing_deg": float(math.degrees(batch_np["gt_bearing"][i])),
                    "pred_bearing_deg": float(math.degrees(batch_np["pred_bearing"][i])),
                    "bearing_error_deg": float(batch_np["bearing_error_deg"][i]),
                    "gt_current_x_m": float(batch_np["gt_current_xy"][i, 0]),
                    "gt_current_y_m": float(batch_np["gt_current_xy"][i, 1]),
                    "pred_current_x_m": float(batch_np["pred_current_xy"][i, 0]),
                    "pred_current_y_m": float(batch_np["pred_current_xy"][i, 1]),
                    "current_pose_error_m": float(batch_np["current_pose_error"][i]),
                    "gt_current_heading_deg": float(math.degrees(batch_np["gt_current_heading"][i])),
                    "pred_current_heading_deg": float(math.degrees(batch_np["pred_current_heading"][i])),
                    "current_heading_error_deg": float(batch_np["current_heading_error_deg"][i]),
                    "next_frame_l1": float(batch_np["next_l1"][i]),
                    "next_frame_psnr": float(batch_np["next_psnr"][i]),
                    "next_frame_ssim": float(batch_np["next_ssim"][i]),
                }
            )

        if args.use_next_frame_aux and len(examples["last"]) < example_count:
            take = min(example_count - len(examples["last"]), bs)
            examples["last"].extend(list(tensor_to_numpy(batch["last_frame"][:take])))
            examples["target"].extend(list(tensor_to_numpy(batch["target_frame"][:take])))
            examples["gt_next"].extend(list(tensor_to_numpy(batch["next_frame"][:take])))
            examples["pred_next"].extend(list(pred_np_img[:take]))

    final_arrays = {key: np.concatenate(value, axis=0) if value else np.array([]) for key, value in arrays.items()}
    metrics = compute_metrics(final_arrays)
    example_arrays = {key: np.stack(value, axis=0) if value else np.array([]) for key, value in examples.items()}
    return metrics, rows, {**final_arrays, **{f"example_{k}": v for k, v in example_arrays.items()}}


def safe_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def safe_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def safe_rmse(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(finite**2))) if finite.size else float("nan")


def safe_p90(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, 0.90)) if finite.size else float("nan")


def compute_metrics(arrays: Dict[str, np.ndarray]) -> Dict[str, float]:
    gt_cmd = arrays["gt_command"].astype(np.int64)
    pred_cmd = arrays["pred_command"].astype(np.int64)
    geom_cmd = arrays["geom_command"].astype(np.int64)
    confusion = confusion_matrix_np(gt_cmd, pred_cmd, len(COMMANDS))
    per_class = per_class_metrics(confusion)
    non_stop = gt_cmd != COMMAND_TO_ID["STOP"]

    metrics: Dict[str, float] = {
        "num_samples": float(gt_cmd.size),
        "relative_mean_m": safe_mean(arrays["rel_error"]),
        "relative_median_m": safe_median(arrays["rel_error"]),
        "relative_rmse_m": safe_rmse(arrays["rel_error"]),
        "relative_p90_m": safe_p90(arrays["rel_error"]),
        "dx_mae_m": safe_mean(np.abs(arrays["pred_rel"][:, 0] - arrays["gt_rel"][:, 0])),
        "dy_mae_m": safe_mean(np.abs(arrays["pred_rel"][:, 1] - arrays["gt_rel"][:, 1])),
        "target_distance_mean_m": safe_mean(arrays["gt_distance"]),
        "pred_target_distance_mean_m": safe_mean(arrays["pred_distance"]),
        "pred_target_distance_median_m": safe_median(arrays["pred_distance"]),
        "target_distance_mae_m": safe_mean(np.abs(arrays["pred_distance"] - arrays["gt_distance"])),
        "bearing_mean_deg_nonstop": safe_mean(arrays["bearing_error_deg"][non_stop]),
        "bearing_median_deg_nonstop": safe_median(arrays["bearing_error_deg"][non_stop]),
        "bearing_rmse_deg_nonstop": safe_rmse(arrays["bearing_error_deg"][non_stop]),
        "bearing_p90_deg_nonstop": safe_p90(arrays["bearing_error_deg"][non_stop]),
        "command_accuracy": float(np.mean(gt_cmd == pred_cmd)) if gt_cmd.size else float("nan"),
        "geometry_command_consistency": float(np.mean(pred_cmd == geom_cmd)) if gt_cmd.size else float("nan"),
        "geometry_command_accuracy": float(np.mean(gt_cmd == geom_cmd)) if gt_cmd.size else float("nan"),
        "current_pose_mean_m": safe_mean(arrays["current_pose_error"]),
        "current_pose_median_m": safe_median(arrays["current_pose_error"]),
        "current_pose_rmse_m": safe_rmse(arrays["current_pose_error"]),
        "current_pose_p90_m": safe_p90(arrays["current_pose_error"]),
        "current_heading_mean_deg": safe_mean(arrays["current_heading_error_deg"]),
        "current_heading_median_deg": safe_median(arrays["current_heading_error_deg"]),
        "current_heading_rmse_deg": safe_rmse(arrays["current_heading_error_deg"]),
        "current_heading_p90_deg": safe_p90(arrays["current_heading_error_deg"]),
        "next_frame_l1": safe_mean(arrays["next_l1"]),
        "next_frame_psnr": safe_mean(arrays["next_psnr"]),
        "next_frame_ssim": safe_mean(arrays["next_ssim"]),
        "macro_f1": float(np.nanmean([per_class[name]["f1"] for name in COMMANDS])),
        "balanced_accuracy": float(np.nanmean([per_class[name]["recall"] for name in COMMANDS])),
    }
    for name in COMMANDS:
        metrics[f"{name.lower()}_precision"] = per_class[name]["precision"]
        metrics[f"{name.lower()}_recall"] = per_class[name]["recall"]
        metrics[f"{name.lower()}_f1"] = per_class[name]["f1"]
        metrics[f"{name.lower()}_support"] = float(per_class[name]["support"])
    return metrics


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
    ax.set_title("Stage 1 Navigation Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_hist(values: np.ndarray, title: str, xlabel: str, path: Path, bins: int = 50) -> Optional[str]:
    if plt is None or values.size == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    ax.hist(values, bins=bins, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_displacement_scatter(arrays: Dict[str, np.ndarray], path: Path) -> Optional[str]:
    if plt is None or arrays["gt_rel"].size == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    gt = arrays["gt_rel"]
    pred = arrays["pred_rel"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=160)
    for dim, label, ax in [(0, "dx east (m)", axes[0]), (1, "dy north (m)", axes[1])]:
        ax.scatter(gt[:, dim], pred[:, dim], s=8, alpha=0.45)
        lo = min(gt[:, dim].min(), pred[:, dim].min())
        hi = max(gt[:, dim].max(), pred[:, dim].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel(f"Ground truth {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Target-Relative Displacement Prediction")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_vector_comparison(arrays: Dict[str, np.ndarray], path: Path, max_points: int = 400) -> Optional[str]:
    if plt is None or arrays["gt_rel"].size == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    gt = arrays["gt_rel"]
    pred = arrays["pred_rel"]
    n = gt.shape[0]
    idx = np.linspace(0, n - 1, min(max_points, n)).astype(int)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    ax.quiver(np.zeros_like(idx), np.zeros_like(idx), gt[idx, 0], gt[idx, 1], angles="xy", scale_units="xy", scale=1, alpha=0.35, color="#1f77b4", label="GT")
    ax.quiver(np.zeros_like(idx), np.zeros_like(idx), pred[idx, 0], pred[idx, 1], angles="xy", scale_units="xy", scale=1, alpha=0.35, color="#ff7f0e", label="Pred")
    all_xy = np.vstack([gt[idx], pred[idx]])
    lim = max(10.0, float(np.max(np.abs(all_xy))) * 1.1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("dx east (m)")
    ax.set_ylabel("dy north (m)")
    ax.set_title("Predicted vs Ground-Truth Target Vectors")
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
    mat = confusion_matrix_np(arrays["gt_command"].astype(np.int64), arrays["pred_command"].astype(np.int64), len(COMMANDS))
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=160)
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(COMMANDS)), COMMANDS, rotation=35, ha="right")
    ax.set_yticks(range(len(COMMANDS)), COMMANDS)
    ax.set_xlabel("Predicted command")
    ax.set_ylabel("Ground-truth command")
    ax.set_title("Command Confusion Matrix")
    for i in range(len(COMMANDS)):
        for j in range(len(COMMANDS)):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_command_distribution(arrays: Dict[str, np.ndarray], path: Path) -> Optional[str]:
    if plt is None or arrays["gt_command"].size == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    gt_counts = np.bincount(arrays["gt_command"].astype(np.int64), minlength=len(COMMANDS))
    pred_counts = np.bincount(arrays["pred_command"].astype(np.int64), minlength=len(COMMANDS))
    x = np.arange(len(COMMANDS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    ax.bar(x - width / 2, gt_counts, width, label="GT")
    ax.bar(x + width / 2, pred_counts, width, label="Pred")
    ax.set_xticks(x, COMMANDS)
    ax.set_ylabel("Count")
    ax.set_title("Command Distribution")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_current_pose(arrays: Dict[str, np.ndarray], path: Path, max_points: int = 1500) -> Optional[str]:
    if plt is None or arrays["gt_current_xy"].size == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(arrays["anchor_frame_id"])
    order = order[:max_points]
    gt = arrays["gt_current_xy"][order]
    pred = arrays["pred_current_xy"][order]
    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    ax.plot(gt[:, 0], gt[:, 1], color="black", lw=1.5, label="GT current pose")
    ax.scatter(pred[:, 0], pred[:, 1], s=8, alpha=0.45, label="Pred current pose")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East x (m)")
    ax.set_ylabel("North y (m)")
    ax.set_title("Auxiliary Current Pose Prediction")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_target_pair_map(arrays: Dict[str, np.ndarray], path: Path, max_pairs: int = 700) -> Optional[str]:
    if plt is None or arrays["gt_current_xy"].size == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    current = arrays["gt_current_xy"]
    target = current + arrays["gt_rel"]
    n = current.shape[0]
    idx = np.linspace(0, n - 1, min(max_pairs, n)).astype(int)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    for i in idx:
        ax.plot([current[i, 0], target[i, 0]], [current[i, 1], target[i, 1]], color="#1f77b4", alpha=0.12, lw=0.8)
    ax.scatter(current[idx, 0], current[idx, 1], s=7, color="black", alpha=0.5, label="Current")
    ax.scatter(target[idx, 0], target[idx, 1], s=7, color="#d62728", alpha=0.5, label="Target")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East x (m)")
    ax.set_ylabel("North y (m)")
    ax.set_title("Direct Current-to-Target Training Pairs")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def chw_to_hwc(img: np.ndarray) -> np.ndarray:
    return np.clip(np.transpose(img, (1, 2, 0)), 0.0, 1.0)


def plot_next_frame_examples(arrays: Dict[str, np.ndarray], path: Path) -> Optional[str]:
    if plt is None or arrays.get("example_last", np.array([])).size == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    last = arrays["example_last"]
    target = arrays["example_target"]
    gt_next = arrays["example_gt_next"]
    pred_next = arrays["example_pred_next"]
    n = min(last.shape[0], 8)
    fig, axes = plt.subplots(n, 4, figsize=(10, 2.4 * n), dpi=160)
    if n == 1:
        axes = axes[None, :]
    titles = ["Last current", "Target", "GT next", "Pred next"]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title)
    for i in range(n):
        for col, img in enumerate([last[i], target[i], gt_next[i], pred_next[i]]):
            axes[i, col].imshow(chw_to_hwc(img))
            axes[i, col].axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def save_all_plots(run_dir: Path, history: Sequence[Dict[str, float]], arrays: Dict[str, np.ndarray]) -> List[str]:
    plot_dir = run_dir / "plots"
    paths = [
        plot_loss_curves(history, plot_dir / "loss_curve.png"),
        plot_hist(arrays["rel_error"], "Target-Relative Error", "Error (m)", plot_dir / "relative_error_histogram.png"),
        plot_hist(arrays["bearing_error_deg"], "Bearing Error", "Error (deg)", plot_dir / "bearing_error_histogram.png"),
        plot_hist(arrays["current_pose_error"], "Auxiliary Current Pose Error", "Error (m)", plot_dir / "current_pose_error_histogram.png"),
        plot_displacement_scatter(arrays, plot_dir / "displacement_predicted_vs_gt.png"),
        plot_vector_comparison(arrays, plot_dir / "target_vector_comparison.png"),
        plot_confusion(arrays, plot_dir / "command_confusion_matrix.png"),
        plot_command_distribution(arrays, plot_dir / "command_distribution.png"),
        plot_current_pose(arrays, plot_dir / "current_pose_prediction_map.png"),
        plot_target_pair_map(arrays, plot_dir / "target_pair_map.png"),
        plot_next_frame_examples(arrays, run_dir / "examples" / "next_frame_examples.png"),
    ]
    return [p for p in paths if p is not None]


def make_run_dir(output_root: str, run_name: str, horizon_name: str) -> Path:
    root = Path(output_root)
    if not run_name:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{horizon_name}_{stamp}"
    run_dir = root / run_name
    for sub in ["checkpoints", "metrics", "plots", "predictions", "examples"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 1 goal-conditioned AVGSO visual navigation.")
    parser.add_argument("--data_root", type=str, default=r"C:\Users\admin\Desktop\spiral_avgso\maps_screenshots_spiral_28150")
    parser.add_argument("--output_root", type=str, default=r"C:\Users\admin\Desktop\spiral_avgso\visual_navigation\results\stage1_immediate")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--horizon_name", type=str, default="immediate", choices=list(HORIZON_TO_ID.keys()))
    parser.add_argument("--distance_min_m", type=float, default=15.0)
    parser.add_argument("--distance_max_m", type=float, default=80.0)
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--targets_per_anchor", type=int, default=2)
    parser.add_argument("--eval_targets_per_anchor", type=int, default=1)
    parser.add_argument("--include_stop", type=int, default=1)
    parser.add_argument("--stop_sample_ratio", type=float, default=0.25)
    parser.add_argument("--stop_threshold_m", type=float, default=8.0)
    parser.add_argument("--forward_angle_deg", type=float, default=15.0)
    parser.add_argument("--turn_around_angle_deg", type=float, default=135.0)
    parser.add_argument("--train_frac", type=float, default=0.80)
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--split_gap", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--backbone", type=str, default="vgg16", choices=["vgg16", "vgg16_bn"])
    parser.add_argument("--use_pretrained", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--rel_output_mode", type=str, default="polar", choices=["polar", "vector"], help="Use polar target direction+distance output, or legacy normalized dx/dy vector output.")
    parser.add_argument("--use_pairwise_fusion", type=int, default=1, help="Fuse current/target features with difference, absolute difference, and elementwise product terms.")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--huber_beta", type=float, default=1.0)
    parser.add_argument("--distance_huber_beta", type=float, default=0.10)
    parser.add_argument("--progress_every", type=int, default=0, help="Print live training progress every N batches. Use 1 for every batch, 0 for V3-style epoch-only output.")

    parser.add_argument("--w_rel", type=float, default=1.0)
    parser.add_argument("--w_bearing", type=float, default=0.20)
    parser.add_argument("--w_distance", type=float, default=0.50)
    parser.add_argument("--w_command", type=float, default=0.50)
    parser.add_argument("--w_pose_xy", type=float, default=0.30)
    parser.add_argument("--w_pose_heading", type=float, default=0.05)
    parser.add_argument("--use_next_frame_aux", type=int, default=0, help="Enable next-frame auxiliary decoder/loss/metrics. Default 0 for faster Stage 1 navigation training.")
    parser.add_argument("--w_next_frame", type=float, default=0.10)

    parser.add_argument("--dry_run", type=int, default=0, help="Build samples and output metadata without training.")
    parser.add_argument("--limit_train_samples", type=int, default=0, help="Optional cap for quick debugging.")
    parser.add_argument("--limit_eval_samples", type=int, default=0, help="Optional cap for quick debugging.")
    parser.add_argument("--save_examples", type=int, default=8)
    return parser


def maybe_limit(samples: List[NavSample], limit: int, seed: int) -> List[NavSample]:
    if limit <= 0 or len(samples) <= limit:
        return samples
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    keep = sorted(indices[:limit])
    return [samples[i] for i in keep]


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)

    run_dir = make_run_dir(args.output_root, args.run_name, args.horizon_name)
    save_json(run_dir / "config.json", vars(args))

    print(f"Data root: {args.data_root}")
    print(f"Run directory: {run_dir}")
    print(f"Horizon: {args.horizon_name} | distance bin: {args.distance_min_m:.1f}-{args.distance_max_m:.1f} m")

    records = list_frames_one_folder(args.data_root)
    ranges = split_ranges(len(records), args.train_frac, args.val_frac, args.split_gap)
    horizon_id = HORIZON_TO_ID[args.horizon_name]
    forward_angle_rad = math.radians(args.forward_angle_deg)
    turn_around_angle_rad = math.radians(args.turn_around_angle_deg)

    train_samples = build_samples_for_split(
        records,
        ranges["train"],
        args.seq_len,
        horizon_id,
        args.distance_min_m,
        args.distance_max_m,
        args.targets_per_anchor,
        bool(args.include_stop),
        args.stop_sample_ratio,
        args.stop_threshold_m,
        forward_angle_rad,
        turn_around_angle_rad,
        args.seed,
    )
    val_samples = build_samples_for_split(
        records,
        ranges["val"],
        args.seq_len,
        horizon_id,
        args.distance_min_m,
        args.distance_max_m,
        args.eval_targets_per_anchor,
        bool(args.include_stop),
        args.stop_sample_ratio,
        args.stop_threshold_m,
        forward_angle_rad,
        turn_around_angle_rad,
        args.seed + 1,
    )
    test_samples = build_samples_for_split(
        records,
        ranges["test"],
        args.seq_len,
        horizon_id,
        args.distance_min_m,
        args.distance_max_m,
        args.eval_targets_per_anchor,
        bool(args.include_stop),
        args.stop_sample_ratio,
        args.stop_threshold_m,
        forward_angle_rad,
        turn_around_angle_rad,
        args.seed + 2,
    )

    train_samples = maybe_limit(train_samples, args.limit_train_samples, args.seed)
    val_samples = maybe_limit(val_samples, args.limit_eval_samples, args.seed + 1)
    test_samples = maybe_limit(test_samples, args.limit_eval_samples, args.seed + 2)

    normalization = compute_normalization(records, train_samples)
    train_counts = command_counts(train_samples)
    val_counts = command_counts(val_samples)
    test_counts = command_counts(test_samples)

    summary = {
        "num_frames": len(records),
        "split_ranges": ranges,
        "samples": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "command_counts": {
            "train": train_counts,
            "val": val_counts,
            "test": test_counts,
        },
        "normalization": normalization,
        "command_names": COMMANDS,
        "horizon_to_id": HORIZON_TO_ID,
    }
    save_json(run_dir / "metrics" / "dataset_summary.json", summary)
    save_json(run_dir / "metrics" / "normalization_stats.json", normalization)

    print(f"Frames: {len(records)}")
    print(f"Split ranges: {ranges}")
    print(f"Samples: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")
    print(f"Train command counts: {train_counts}")
    print(f"Val command counts:   {val_counts}")
    print(f"Test command counts:  {test_counts}")

    if args.dry_run:
        print("Dry run complete. No model was trained.")
        return

    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch/torchvision are required for training. Run this script from the "
            "same Anaconda environment used for the localization models."
        )

    use_next_frame_aux = bool(args.use_next_frame_aux)
    train_ds = GoalNavigationDataset(records, train_samples, normalization, args.seq_len, args.image_size, use_next_frame_aux)
    val_ds = GoalNavigationDataset(records, val_samples, normalization, args.seq_len, args.image_size, use_next_frame_aux)
    test_ds = GoalNavigationDataset(records, test_samples, normalization, args.seq_len, args.image_size, use_next_frame_aux)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = GoalConditionedTransNav(
        seq_len=args.seq_len,
        image_size=args.image_size,
        d_model=args.d_model,
        nhead=args.nhead,
        layers=args.layers,
        dropout=args.dropout,
        backbone=args.backbone,
        use_pretrained=bool(args.use_pretrained),
        use_next_frame_aux=use_next_frame_aux,
        rel_output_mode=args.rel_output_mode,
        distance_scale_m=args.distance_max_m,
        use_pairwise_fusion=bool(args.use_pairwise_fusion),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    rel_mean = torch.tensor(normalization["rel_mean"], dtype=torch.float32, device=device)
    rel_std = torch.tensor(normalization["rel_std"], dtype=torch.float32, device=device)
    command_weights = class_weights_from_counts(train_counts).to(device)

    print(f"Device: {device}")
    print(f"Next-frame auxiliary task: {use_next_frame_aux}")
    print(f"Relative output mode: {args.rel_output_mode} | distance scale: {args.distance_max_m:.1f} m")
    print(f"Pairwise current-target fusion: {bool(args.use_pairwise_fusion)}")
    print(f"Command loss weights: {tensor_to_numpy(command_weights)}")
    print(f"Relative normalization mean={normalization['rel_mean']} std={normalization['rel_std']}")

    history: List[Dict[str, float]] = []
    best_val = float("inf")
    best_path = run_dir / "checkpoints" / "best.pt"
    live_progress_path = run_dir / "metrics" / "live_progress.txt"

    for epoch in range(1, args.epochs + 1):
        if args.progress_every > 0:
            print(f"Starting epoch {epoch:03d}/{args.epochs:03d}", flush=True)
            write_live_progress(live_progress_path, f"Starting epoch {epoch:03d}/{args.epochs:03d}")
        train_logs = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            rel_mean,
            rel_std,
            command_weights,
            args,
            epoch,
            args.epochs,
            live_progress_path,
        )
        if args.progress_every > 0:
            print(f"Evaluating epoch {epoch:03d}/{args.epochs:03d}...", flush=True)
            write_live_progress(live_progress_path, f"Evaluating epoch {epoch:03d}/{args.epochs:03d}...")
        val_logs = evaluate_loss(
            model,
            val_loader,
            device,
            rel_mean,
            rel_std,
            command_weights,
            args,
        )
        scheduler.step()

        row: Dict[str, float] = {"epoch": float(epoch)}
        row.update({f"train_{k}": v for k, v in train_logs.items()})
        row.update({f"val_{k}": v for k, v in val_logs.items()})
        row["train_loss"] = train_logs["loss"]
        row["val_loss"] = val_logs["loss"]
        history.append(row)
        save_loss_log(run_dir / "metrics" / "loss_log.csv", history)

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_logs['loss']:.4f} | val={val_logs['loss']:.4f} | "
            f"rel={val_logs['rel']:.4f} dist={val_logs['distance']:.4f} "
            f"cmd={val_logs['command']:.4f} img={val_logs['next_frame']:.4f}",
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
                    "normalization": normalization,
                    "args": vars(args),
                    "command_names": COMMANDS,
                    "horizon_to_id": HORIZON_TO_ID,
                },
                best_path,
            )

    print("Loading best checkpoint and running final test evaluation...", flush=True)
    write_live_progress(live_progress_path, "Loading best checkpoint and running final test evaluation...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss_logs = evaluate_loss(model, test_loader, device, rel_mean, rel_std, command_weights, args)
    print("Computing final metrics and collecting prediction examples...", flush=True)
    write_live_progress(live_progress_path, "Computing final metrics and collecting prediction examples...")
    metrics, prediction_rows, arrays = evaluate_full(
        model,
        test_loader,
        device,
        normalization,
        args,
        example_count=args.save_examples,
    )
    metrics["best_val_loss"] = float(best_val)
    metrics["test_loss"] = float(test_loss_logs["loss"])

    save_metrics_csv(run_dir / "metrics" / "final_navigation_metrics.csv", metrics)
    save_predictions_csv(run_dir / "predictions" / "test_per_sample_predictions.csv", prediction_rows)
    print("Saving analysis plots...", flush=True)
    write_live_progress(live_progress_path, "Saving analysis plots...")
    plot_paths = save_all_plots(run_dir, history, arrays)

    print("Final test metrics:")
    for key in [
        "relative_mean_m",
        "relative_p90_m",
        "bearing_mean_deg_nonstop",
        "command_accuracy",
        "balanced_accuracy",
        "macro_f1",
        "geometry_command_consistency",
        "current_pose_mean_m",
        "current_heading_mean_deg",
        "next_frame_l1",
        "next_frame_psnr",
    ]:
        print(f"  {key}: {metrics.get(key, float('nan')):.4f}")
    print(f"Saved best checkpoint: {best_path}")
    print(f"Saved metrics: {run_dir / 'metrics' / 'final_navigation_metrics.csv'}")
    print(f"Saved predictions: {run_dir / 'predictions' / 'test_per_sample_predictions.csv'}")
    if plot_paths:
        print("Saved plots:")
        for path in plot_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
