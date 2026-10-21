# -*- coding: utf-8 -*-
"""
Stage 1 route-conditioned visual navigation for AVGSO spiral data.

Assumption:
  The UAV is given only the start and goal coordinates, plus a target image.
  Intermediate/current coordinates are not provided to the model. Hidden frame
  coordinates are used only to create training labels and evaluation metrics.

Inputs:
  - current multi-frame visual sequence
  - target image at the goal location
  - start/goal route prior from coordinates:
      start x/y, goal x/y, start-goal dx/dy, total distance, initial direction

Main output:
  - navigation command: STOP, LEFT, RIGHT, FORWARD, TURN_AROUND

Auxiliary outputs:
  - current hidden navigation state relative to the goal
  - next hidden navigation state relative to the goal
  - next command
  - reached/arrival probability

Distance horizons:
  short:  < 100 m
  medium: 100-500 m
  long:   500-1500 m
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
HORIZONS = ["short", "medium", "long"]
HORIZON_TO_ID = {name: i for i, name in enumerate(HORIZONS)}
HEADING_CLASSES = ["LEFT", "RIGHT", "FORWARD", "TURN_AROUND"]
HEADING_TO_ID = {name: i for i, name in enumerate(HEADING_CLASSES)}
STEERING_SIGNS = ["LEFT", "CENTER", "RIGHT", "BACK"]
STEERING_TO_ID = {name: i for i, name in enumerate(STEERING_SIGNS)}
HEADING_TO_STEERING = {
    HEADING_TO_ID["LEFT"]: STEERING_TO_ID["LEFT"],
    HEADING_TO_ID["RIGHT"]: STEERING_TO_ID["RIGHT"],
    HEADING_TO_ID["FORWARD"]: STEERING_TO_ID["CENTER"],
    HEADING_TO_ID["TURN_AROUND"]: STEERING_TO_ID["BACK"],
}


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
class RouteSample:
    start_idx: int
    current_idx: int
    goal_idx: int
    next_idx: int
    horizon_id: int
    command_id: int
    next_command_id: int
    heading_class_id: int
    next_heading_class_id: int
    reached: float
    next_reached: float
    route_dx: float
    route_dy: float
    route_distance_m: float
    route_direction_rad: float
    current_dx_to_goal: float
    current_dy_to_goal: float
    current_remaining_m: float
    current_progress: float
    current_heading_error_rad: float
    next_dx_to_goal: float
    next_dy_to_goal: float
    next_remaining_m: float
    next_progress: float
    next_heading_error_rad: float


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
            "Expected frame_<id>_<lat>_<lon>_<heading> image names."
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
            raise RuntimeError(f"Empty {name} split after split_gap={split_gap}: start={start}, end={end}")
    return ranges


def horizon_from_distance(distance_m: float, short_max_m: float, medium_max_m: float, long_max_m: float) -> Optional[int]:
    if 0.0 < distance_m < short_max_m:
        return HORIZON_TO_ID["short"]
    if short_max_m <= distance_m < medium_max_m:
        return HORIZON_TO_ID["medium"]
    if medium_max_m <= distance_m <= long_max_m:
        return HORIZON_TO_ID["long"]
    return None


def command_from_goal_geometry(
    dx_to_goal: float,
    dy_to_goal: float,
    current_heading_rad: float,
    stop_threshold_m: float,
    forward_angle_rad: float,
    turn_around_angle_rad: float,
) -> Tuple[int, int, float, float]:
    distance = math.hypot(dx_to_goal, dy_to_goal)
    if distance <= stop_threshold_m:
        return COMMAND_TO_ID["STOP"], HEADING_TO_ID["FORWARD"], 0.0, distance

    target_direction = math.atan2(dx_to_goal, dy_to_goal)
    heading_error = wrap_angle_rad(target_direction - current_heading_rad)
    abs_error = abs(heading_error)
    if abs_error <= forward_angle_rad:
        return COMMAND_TO_ID["FORWARD"], HEADING_TO_ID["FORWARD"], heading_error, distance
    if abs_error >= turn_around_angle_rad:
        return COMMAND_TO_ID["TURN_AROUND"], HEADING_TO_ID["TURN_AROUND"], heading_error, distance
    if heading_error < 0:
        return COMMAND_TO_ID["LEFT"], HEADING_TO_ID["LEFT"], heading_error, distance
    return COMMAND_TO_ID["RIGHT"], HEADING_TO_ID["RIGHT"], heading_error, distance


def route_progress(start_rec: FrameRecord, goal_rec: FrameRecord, current_rec: FrameRecord, route_distance_m: float) -> float:
    if route_distance_m <= 1e-6:
        return 1.0
    vx = goal_rec.x - start_rec.x
    vy = goal_rec.y - start_rec.y
    wx = current_rec.x - start_rec.x
    wy = current_rec.y - start_rec.y
    projection = (wx * vx + wy * vy) / (route_distance_m * route_distance_m)
    return float(max(0.0, min(1.0, projection)))


def build_one_sample(
    records: Sequence[FrameRecord],
    start_idx: int,
    current_idx: int,
    goal_idx: int,
    next_idx: int,
    horizon_id: int,
    stop_threshold_m: float,
    forward_angle_rad: float,
    turn_around_angle_rad: float,
) -> RouteSample:
    start_rec = records[start_idx]
    current_rec = records[current_idx]
    goal_rec = records[goal_idx]
    next_rec = records[next_idx]

    route_dx = goal_rec.x - start_rec.x
    route_dy = goal_rec.y - start_rec.y
    route_distance = math.hypot(route_dx, route_dy)
    route_direction = math.atan2(route_dx, route_dy)

    current_dx = goal_rec.x - current_rec.x
    current_dy = goal_rec.y - current_rec.y
    command_id, heading_class_id, heading_error, remaining = command_from_goal_geometry(
        current_dx, current_dy, current_rec.heading_rad, stop_threshold_m, forward_angle_rad, turn_around_angle_rad
    )
    next_dx = goal_rec.x - next_rec.x
    next_dy = goal_rec.y - next_rec.y
    next_command_id, next_heading_class_id, next_heading_error, next_remaining = command_from_goal_geometry(
        next_dx, next_dy, next_rec.heading_rad, stop_threshold_m, forward_angle_rad, turn_around_angle_rad
    )

    progress = route_progress(start_rec, goal_rec, current_rec, route_distance)
    next_progress = route_progress(start_rec, goal_rec, next_rec, route_distance)
    reached = 1.0 if remaining <= stop_threshold_m else 0.0
    next_reached = 1.0 if next_remaining <= stop_threshold_m else 0.0

    return RouteSample(
        start_idx=start_idx,
        current_idx=current_idx,
        goal_idx=goal_idx,
        next_idx=next_idx,
        horizon_id=horizon_id,
        command_id=command_id,
        next_command_id=next_command_id,
        heading_class_id=heading_class_id,
        next_heading_class_id=next_heading_class_id,
        reached=reached,
        next_reached=next_reached,
        route_dx=route_dx,
        route_dy=route_dy,
        route_distance_m=route_distance,
        route_direction_rad=route_direction,
        current_dx_to_goal=current_dx,
        current_dy_to_goal=current_dy,
        current_remaining_m=remaining,
        current_progress=progress,
        current_heading_error_rad=heading_error,
        next_dx_to_goal=next_dx,
        next_dy_to_goal=next_dy,
        next_remaining_m=next_remaining,
        next_progress=next_progress,
        next_heading_error_rad=next_heading_error,
    )


def candidate_goal_indices(
    start_idx: int,
    end: int,
    rng: random.Random,
    candidate_trials: int,
    max_offset_frames: int,
) -> Iterable[int]:
    deterministic_offsets = [
        1,
        2,
        3,
        4,
        6,
        8,
        10,
        12,
        16,
        20,
        28,
        40,
        55,
        75,
        100,
        140,
        190,
        260,
        360,
        500,
        700,
    ]
    seen = set()
    for offset in deterministic_offsets:
        goal_idx = start_idx + offset
        if goal_idx < end:
            seen.add(goal_idx)
            yield goal_idx

    max_goal = min(end - 1, start_idx + max_offset_frames)
    if max_goal <= start_idx:
        return
    for _ in range(candidate_trials):
        goal_idx = rng.randint(start_idx + 1, max_goal)
        if goal_idx in seen:
            continue
        seen.add(goal_idx)
        yield goal_idx


def sampled_route_state_indices(start_idx: int, goal_idx: int, route_states: int) -> List[int]:
    if route_states <= 1 or goal_idx <= start_idx:
        return [start_idx, goal_idx]
    values = np.linspace(start_idx, goal_idx, route_states)
    out = sorted({int(round(v)) for v in values})
    if out[-1] != goal_idx:
        out.append(goal_idx)
    return out


def build_route_samples_for_split(
    records: Sequence[FrameRecord],
    split_range: Tuple[int, int],
    seq_len: int,
    route_start_stride: int,
    route_states: int,
    targets_per_horizon: int,
    candidate_trials: int,
    max_offset_frames: int,
    short_max_m: float,
    medium_max_m: float,
    long_max_m: float,
    stop_threshold_m: float,
    forward_angle_rad: float,
    turn_around_angle_rad: float,
    max_state_remaining_factor: float,
    max_state_remaining_extra_m: float,
    max_allowed_state_distance_m: float,
    active_horizon_ids: Sequence[int],
    seed: int,
) -> List[RouteSample]:
    start, end = split_range
    if end - start <= seq_len + 2:
        raise RuntimeError(f"Split too small for seq_len={seq_len}: start={start}, end={end}")

    rng = random.Random(seed)
    samples: List[RouteSample] = []
    first_start = start + seq_len - 1
    last_start = end - 2

    for start_idx in range(first_start, last_start, max(route_start_stride, 1)):
        start_rec = records[start_idx]
        by_horizon: Dict[int, List[int]] = {i: [] for i in range(len(HORIZONS))}
        for goal_idx in candidate_goal_indices(start_idx, end, rng, candidate_trials, max_offset_frames):
            goal_rec = records[goal_idx]
            distance = math.hypot(goal_rec.x - start_rec.x, goal_rec.y - start_rec.y)
            horizon_id = horizon_from_distance(distance, short_max_m, medium_max_m, long_max_m)
            if horizon_id is None:
                continue
            if horizon_id not in active_horizon_ids:
                continue
            by_horizon[horizon_id].append(goal_idx)

        for horizon_id, goals in by_horizon.items():
            if not goals:
                continue
            rng.shuffle(goals)
            for goal_idx in goals[:targets_per_horizon]:
                for current_idx in sampled_route_state_indices(start_idx, goal_idx, route_states):
                    if current_idx < first_start or current_idx >= end:
                        continue
                    next_idx = min(current_idx + 1, goal_idx)
                    sample = build_one_sample(
                        records,
                        start_idx,
                        current_idx,
                        goal_idx,
                        next_idx,
                        horizon_id,
                        stop_threshold_m,
                        forward_angle_rad,
                        turn_around_angle_rad,
                    )
                    max_allowed_remaining = min(
                        max_allowed_state_distance_m,
                        sample.route_distance_m * max_state_remaining_factor + max_state_remaining_extra_m,
                    )
                    if (
                        sample.current_remaining_m <= max_allowed_remaining
                        and sample.next_remaining_m <= max_allowed_remaining
                    ):
                        samples.append(sample)

    if not samples:
        raise RuntimeError("No route samples were built. Check distance bins, split ranges, and frame parsing.")
    return samples


def parse_horizons(value: str) -> List[str]:
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("--horizons must include at least one of: short, medium, long")
    invalid = [name for name in names if name not in HORIZON_TO_ID]
    if invalid:
        raise ValueError(f"Invalid --horizons value(s): {invalid}. Use comma-separated short,medium,long.")
    deduped: List[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def max_distance_for_horizons(
    horizon_names: Sequence[str],
    short_max_m: float,
    medium_max_m: float,
    long_max_m: float,
) -> float:
    upper_by_horizon = {
        "short": short_max_m,
        "medium": medium_max_m,
        "long": long_max_m,
    }
    return max(upper_by_horizon[name] for name in horizon_names)


def compute_normalization(
    records: Sequence[FrameRecord],
    train_range: Tuple[int, int],
    train_samples: Sequence[RouteSample],
    max_distance_m: float,
) -> Dict[str, List[float] | float]:
    start, end = train_range
    xy = np.array([[records[i].x, records[i].y] for i in range(start, end)], dtype=np.float32)
    xy_mean = xy.mean(axis=0)
    xy_std = np.maximum(xy.std(axis=0), 1.0)

    route_prior = np.array(
        [route_prior_features(records, sample, xy_mean, xy_std, max_distance_m) for sample in train_samples],
        dtype=np.float32,
    )
    route_mean = route_prior.mean(axis=0)
    route_std = np.maximum(route_prior.std(axis=0), 1e-3)
    return {
        "xy_mean": xy_mean.tolist(),
        "xy_std": xy_std.tolist(),
        "route_prior_mean": route_mean.tolist(),
        "route_prior_std": route_std.tolist(),
        "state_distance_scale": float(max(max_distance_m, 1.0)),
    }


def route_prior_features(
    records: Sequence[FrameRecord],
    sample: RouteSample,
    xy_mean: np.ndarray,
    xy_std: np.ndarray,
    max_distance_m: float,
) -> np.ndarray:
    start_rec = records[sample.start_idx]
    goal_rec = records[sample.goal_idx]
    start_xy = (np.array([start_rec.x, start_rec.y], dtype=np.float32) - xy_mean) / xy_std
    goal_xy = (np.array([goal_rec.x, goal_rec.y], dtype=np.float32) - xy_mean) / xy_std
    scale = max(max_distance_m, 1.0)
    return np.array(
        [
            start_xy[0],
            start_xy[1],
            goal_xy[0],
            goal_xy[1],
            sample.route_dx / scale,
            sample.route_dy / scale,
            sample.route_distance_m / scale,
            math.sin(sample.route_direction_rad),
            math.cos(sample.route_direction_rad),
        ],
        dtype=np.float32,
    )


def command_counts(samples: Sequence[RouteSample], attr: str = "command_id") -> Dict[str, int]:
    counts = {name: 0 for name in COMMANDS}
    for sample in samples:
        counts[COMMANDS[getattr(sample, attr)]] += 1
    return counts


def horizon_counts(samples: Sequence[RouteSample]) -> Dict[str, int]:
    counts = {name: 0 for name in HORIZONS}
    for sample in samples:
        counts[HORIZONS[sample.horizon_id]] += 1
    return counts


def numeric_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float32)
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def route_state_stats(samples: Sequence[RouteSample]) -> Dict[str, Dict[str, float]]:
    return {
        "route_distance_m": numeric_stats([s.route_distance_m for s in samples]),
        "current_remaining_m": numeric_stats([s.current_remaining_m for s in samples]),
        "next_remaining_m": numeric_stats([s.next_remaining_m for s in samples]),
        "current_progress": numeric_stats([s.current_progress for s in samples]),
        "next_progress": numeric_stats([s.next_progress for s in samples]),
    }


def command_geometry_stats(samples: Sequence[RouteSample]) -> Dict[str, Dict[str, Dict[str, float] | int]]:
    stats: Dict[str, Dict[str, Dict[str, float] | int]] = {}
    for command_id, command_name in enumerate(COMMANDS):
        subset = [sample for sample in samples if sample.command_id == command_id]
        heading_deg = [math.degrees(sample.current_heading_error_rad) for sample in subset]
        stats[command_name] = {
            "support": len(subset),
            "remaining_m": numeric_stats([sample.current_remaining_m for sample in subset]),
            "heading_error_deg": numeric_stats(heading_deg),
            "abs_heading_error_deg": numeric_stats([abs(value) for value in heading_deg]),
        }
    return stats


def class_weights_from_counts(counts: Dict[str, int]) -> torch.Tensor:
    values = np.array([counts[name] for name in COMMANDS], dtype=np.float32)
    values = np.maximum(values, 1.0)
    weights = values.sum() / (len(values) * values)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def steering_counts(samples: Sequence[RouteSample], next_label: bool = False) -> Dict[str, int]:
    counts = {name: 0 for name in STEERING_SIGNS}
    for sample in samples:
        heading_id = sample.next_heading_class_id if next_label else sample.heading_class_id
        counts[STEERING_SIGNS[HEADING_TO_STEERING[heading_id]]] += 1
    return counts


def class_weights_from_named_counts(counts: Dict[str, int], names: Sequence[str]) -> torch.Tensor:
    values = np.array([counts[name] for name in names], dtype=np.float32)
    values = np.maximum(values, 1.0)
    weights = values.sum() / (len(values) * values)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def pos_weight_from_samples(samples: Sequence[RouteSample]) -> float:
    positives = sum(1 for s in samples if s.reached > 0.5)
    negatives = max(len(samples) - positives, 1)
    return float(negatives / max(positives, 1))


class RouteConditionedDataset(Dataset):
    def __init__(
        self,
        records: Sequence[FrameRecord],
        samples: Sequence[RouteSample],
        normalization: Dict[str, List[float] | float],
        seq_len: int,
        image_size: int,
    ):
        self.records = records
        self.samples = samples
        self.seq_len = seq_len
        self.image_size = image_size
        self.xy_mean = np.array(normalization["xy_mean"], dtype=np.float32)
        self.xy_std = np.array(normalization["xy_std"], dtype=np.float32)
        self.route_prior_mean = np.array(normalization["route_prior_mean"], dtype=np.float32)
        self.route_prior_std = np.array(normalization["route_prior_std"], dtype=np.float32)
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

    def _state_tensor(self, sample: RouteSample, prefix: str) -> torch.Tensor:
        if prefix == "current":
            dx = sample.current_dx_to_goal
            dy = sample.current_dy_to_goal
            remaining = sample.current_remaining_m
            progress = sample.current_progress
            heading_error = sample.current_heading_error_rad
        elif prefix == "next":
            dx = sample.next_dx_to_goal
            dy = sample.next_dy_to_goal
            remaining = sample.next_remaining_m
            progress = sample.next_progress
            heading_error = sample.next_heading_error_rad
        else:
            raise ValueError(prefix)
        scale = self.state_scale
        return torch.tensor(
            [
                dx / scale,
                dy / scale,
                remaining / scale,
                progress,
                math.sin(heading_error),
                math.cos(heading_error),
            ],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        current_indices = range(sample.current_idx - self.seq_len + 1, sample.current_idx + 1)
        current_images = torch.stack([self._load_norm(self.records[i].path) for i in current_indices], dim=0)
        target_image = self._load_norm(self.records[sample.goal_idx].path)

        route_prior = route_prior_features(
            self.records,
            sample,
            self.xy_mean,
            self.xy_std,
            self.state_scale,
        )
        route_prior_norm = (route_prior - self.route_prior_mean) / self.route_prior_std

        return {
            "current_images": current_images,
            "target_image": target_image,
            "route_prior": torch.tensor(route_prior_norm, dtype=torch.float32),
            "horizon_id": torch.tensor(sample.horizon_id, dtype=torch.long),
            "command": torch.tensor(sample.command_id, dtype=torch.long),
            "next_command": torch.tensor(sample.next_command_id, dtype=torch.long),
            "heading_class": torch.tensor(sample.heading_class_id, dtype=torch.long),
            "next_heading_class": torch.tensor(sample.next_heading_class_id, dtype=torch.long),
            "steering_sign": torch.tensor(HEADING_TO_STEERING[sample.heading_class_id], dtype=torch.long),
            "next_steering_sign": torch.tensor(HEADING_TO_STEERING[sample.next_heading_class_id], dtype=torch.long),
            "reached": torch.tensor(sample.reached, dtype=torch.float32),
            "next_reached": torch.tensor(sample.next_reached, dtype=torch.float32),
            "current_state": self._state_tensor(sample, "current"),
            "next_state": self._state_tensor(sample, "next"),
            "route_distance_m": torch.tensor(sample.route_distance_m, dtype=torch.float32),
            "current_remaining_m": torch.tensor(sample.current_remaining_m, dtype=torch.float32),
            "next_remaining_m": torch.tensor(sample.next_remaining_m, dtype=torch.float32),
            "current_progress": torch.tensor(sample.current_progress, dtype=torch.float32),
            "next_progress": torch.tensor(sample.next_progress, dtype=torch.float32),
            "current_heading_error": torch.tensor(sample.current_heading_error_rad, dtype=torch.float32),
            "next_heading_error": torch.tensor(sample.next_heading_error_rad, dtype=torch.float32),
            "start_frame_id": torch.tensor(self.records[sample.start_idx].frame_id, dtype=torch.long),
            "current_frame_id": torch.tensor(self.records[sample.current_idx].frame_id, dtype=torch.long),
            "goal_frame_id": torch.tensor(self.records[sample.goal_idx].frame_id, dtype=torch.long),
            "next_frame_id": torch.tensor(self.records[sample.next_idx].frame_id, dtype=torch.long),
            "start_path": Path(self.records[sample.start_idx].path).name,
            "current_path": Path(self.records[sample.current_idx].path).name,
            "goal_path": Path(self.records[sample.goal_idx].path).name,
            "next_path": Path(self.records[sample.next_idx].path).name,
        }


class VGGTokenStem(nn.Module):
    def __init__(self, backbone: str, d_model: int, use_pretrained: bool, freeze_backbone: bool = False):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        weights = None
        if use_pretrained:
            try:
                if backbone == "vgg16_bn":
                    weights = VGG16_BN_Weights.IMAGENET1K_V1
                else:
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
                base = vgg16_bn(weights=None) if backbone == "vgg16_bn" else vgg16(weights=None)
            else:
                raise
        self.features = base.features
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        self.proj = nn.Linear(512, d_model)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        if self.freeze_backbone:
            with torch.no_grad():
                feat = self.features(images)
        else:
            feat = self.features(images)
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        return self.proj(tokens), h, w


class RouteConditionedNavModel(nn.Module):
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
        route_prior_dim: int = 9,
        max_tokens: int = 1024,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.max_tokens = max_tokens
        self.stem = VGGTokenStem(
            backbone=backbone,
            d_model=d_model,
            use_pretrained=use_pretrained,
            freeze_backbone=freeze_backbone,
        )

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
        self.route_mlp = nn.Sequential(
            nn.LayerNorm(route_prior_dim),
            nn.Linear(route_prior_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.horizon_emb = nn.Embedding(len(HORIZONS), d_model)
        fusion_dim = d_model * 7
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.command_head = self._head(d_model, len(COMMANDS), dropout)
        self.next_command_head = self._head(d_model, len(COMMANDS), dropout)
        self.steering_sign_head = self._head(d_model, len(STEERING_SIGNS), dropout)
        self.next_steering_sign_head = self._head(d_model, len(STEERING_SIGNS), dropout)
        self.heading_class_head = self._head(d_model, len(HEADING_CLASSES), dropout)
        self.next_heading_class_head = self._head(d_model, len(HEADING_CLASSES), dropout)
        self.reached_head = self._head(d_model, 1, dropout)
        self.next_reached_head = self._head(d_model, 1, dropout)
        self.current_state_head = self._head(d_model, 6, dropout)
        self.next_state_head = self._head(d_model, 6, dropout)

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

    def _add_target_pos(self, tokens: torch.Tensor, l_tokens: int) -> torch.Tensor:
        space_ids = torch.arange(l_tokens, device=tokens.device)
        pos = self.space_emb(space_ids)[None, :, :]
        return tokens + pos + self.target_segment

    def forward(self, current_images: torch.Tensor, target_image: torch.Tensor, route_prior: torch.Tensor, horizon_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, t, c, h, w = current_images.shape
        if t != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {t}")

        flat = current_images.reshape(b * t, c, h, w)
        cur_tokens_bt, _, _ = self.stem(flat)
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

        route_repr = self.route_mlp(route_prior)
        horizon_repr = self.horizon_emb(horizon_id)
        fusion_input = torch.cat(
            [
                current_repr,
                target_repr,
                current_repr - target_repr,
                torch.abs(current_repr - target_repr),
                current_repr * target_repr,
                route_repr,
                horizon_repr,
            ],
            dim=-1,
        )
        fused = self.fusion(fusion_input)
        current_state = self.current_state_head(fused)
        next_state = self.next_state_head(fused)
        current_state = torch.cat([current_state[:, :4], F.normalize(current_state[:, 4:], dim=-1, eps=1e-6)], dim=-1)
        next_state = torch.cat([next_state[:, :4], F.normalize(next_state[:, 4:], dim=-1, eps=1e-6)], dim=-1)
        return {
            "command_logits": self.command_head(fused),
            "next_command_logits": self.next_command_head(fused),
            "steering_sign_logits": self.steering_sign_head(fused),
            "next_steering_sign_logits": self.next_steering_sign_head(fused),
            "heading_class_logits": self.heading_class_head(fused),
            "next_heading_class_logits": self.next_heading_class_head(fused),
            "reached_logit": self.reached_head(fused).squeeze(-1),
            "next_reached_logit": self.next_reached_head(fused).squeeze(-1),
            "current_state": current_state,
            "next_state": next_state,
        }


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved: Dict[str, object] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def circular_sincos_loss(pred_sincos: torch.Tensor, gt_sincos: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred_sincos, dim=-1, eps=1e-6)
    gt = F.normalize(gt_sincos, dim=-1, eps=1e-6)
    cos_sim = (pred * gt).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cos_sim).mean()


def state_loss(pred: torch.Tensor, gt: torch.Tensor, args: argparse.Namespace) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    loss_rel = F.smooth_l1_loss(pred[:, :2], gt[:, :2], beta=args.huber_beta)
    loss_remaining = F.smooth_l1_loss(pred[:, 2], gt[:, 2], beta=args.huber_beta)
    loss_progress = F.smooth_l1_loss(pred[:, 3], gt[:, 3], beta=args.huber_beta)
    loss_heading = circular_sincos_loss(pred[:, 4:], gt[:, 4:])
    total = (
        args.w_state_rel * loss_rel
        + args.w_state_remaining * loss_remaining
        + args.w_state_progress * loss_progress
        + args.w_state_heading * loss_heading
    )
    return total, {
        "state_rel": loss_rel,
        "state_remaining": loss_remaining,
        "state_progress": loss_progress,
        "state_heading": loss_heading,
    }


def total_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    command_weights: torch.Tensor,
    steering_weights: torch.Tensor,
    reached_pos_weight: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    current_state_loss, current_parts = state_loss(out["current_state"], batch["current_state"], args)
    next_state_loss, next_parts = state_loss(out["next_state"], batch["next_state"], args)
    command_loss = F.cross_entropy(out["command_logits"], batch["command"], weight=command_weights)
    next_command_loss = F.cross_entropy(out["next_command_logits"], batch["next_command"], weight=command_weights)
    steering_sign_loss = F.cross_entropy(out["steering_sign_logits"], batch["steering_sign"], weight=steering_weights)
    next_steering_sign_loss = F.cross_entropy(
        out["next_steering_sign_logits"], batch["next_steering_sign"], weight=steering_weights
    )
    heading_class_loss = F.cross_entropy(out["heading_class_logits"], batch["heading_class"])
    next_heading_class_loss = F.cross_entropy(out["next_heading_class_logits"], batch["next_heading_class"])
    reached_loss = F.binary_cross_entropy_with_logits(
        out["reached_logit"], batch["reached"], pos_weight=reached_pos_weight
    )
    next_reached_loss = F.binary_cross_entropy_with_logits(
        out["next_reached_logit"], batch["next_reached"], pos_weight=reached_pos_weight
    )

    loss = (
        args.w_command * command_loss
        + args.w_state * current_state_loss
        + args.w_reached * reached_loss
        + args.w_heading_class * heading_class_loss
        + args.w_steering_sign * steering_sign_loss
        + args.w_next_command * next_command_loss
        + args.w_next_state * next_state_loss
        + args.w_next_reached * next_reached_loss
        + args.w_next_heading_class * next_heading_class_loss
        + args.w_next_steering_sign * next_steering_sign_loss
    )
    logs = {
        "loss": float(loss.detach().cpu()),
        "command": float(command_loss.detach().cpu()),
        "state": float(current_state_loss.detach().cpu()),
        "reached": float(reached_loss.detach().cpu()),
        "heading_class": float(heading_class_loss.detach().cpu()),
        "steering_sign": float(steering_sign_loss.detach().cpu()),
        "next_command": float(next_command_loss.detach().cpu()),
        "next_state": float(next_state_loss.detach().cpu()),
        "next_reached": float(next_reached_loss.detach().cpu()),
        "next_heading_class": float(next_heading_class_loss.detach().cpu()),
        "next_steering_sign": float(next_steering_sign_loss.detach().cpu()),
        "state_rel": float(current_parts["state_rel"].detach().cpu()),
        "state_remaining": float(current_parts["state_remaining"].detach().cpu()),
        "state_progress": float(current_parts["state_progress"].detach().cpu()),
        "state_heading": float(current_parts["state_heading"].detach().cpu()),
        "next_state_rel": float(next_parts["state_rel"].detach().cpu()),
        "next_state_remaining": float(next_parts["state_remaining"].detach().cpu()),
        "next_state_progress": float(next_parts["state_progress"].detach().cpu()),
        "next_state_heading": float(next_parts["state_heading"].detach().cpu()),
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
    steering_weights: torch.Tensor,
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
            out = model(batch["current_images"], batch["target_image"], batch["route_prior"], batch["horizon_id"])
            loss, logs = total_loss(out, batch, command_weights, steering_weights, reached_pos_weight, args)
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
                f"batch {batch_idx:05d}/{total_batches:05d} samples {count:06d} "
                f"loss={running['loss']:.4f} cmd={running['command']:.4f} "
                f"state={running['state']:.4f} next={running['next_state']:.4f} "
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
    command_weights: torch.Tensor,
    steering_weights: torch.Tensor,
    reached_pos_weight: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch["current_images"], batch["target_image"], batch["route_prior"], batch["horizon_id"])
        _, logs = total_loss(out, batch, command_weights, steering_weights, reached_pos_weight, args)
        bs = batch["current_images"].size(0)
        average_logs(totals, logs, bs)
        count += bs
    return finalize_logs(totals, count)


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def decode_state(state: np.ndarray, scale: float) -> Dict[str, np.ndarray]:
    dx = state[:, 0] * scale
    dy = state[:, 1] * scale
    remaining = state[:, 2] * scale
    progress = state[:, 3]
    heading = np.arctan2(state[:, 4], state[:, 5])
    return {"dx": dx, "dy": dy, "remaining": remaining, "progress": progress, "heading_error": heading}


@torch.no_grad()
def evaluate_full(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    state_scale: float,
    args: argparse.Namespace,
) -> Tuple[Dict[str, float], List[Dict[str, object]], Dict[str, np.ndarray]]:
    model.eval()
    rows: List[Dict[str, object]] = []
    arrays: Dict[str, List[np.ndarray]] = {
        "gt_command": [],
        "pred_command": [],
        "gt_next_command": [],
        "pred_next_command": [],
        "gt_steering_sign": [],
        "pred_steering_sign": [],
        "gt_next_steering_sign": [],
        "pred_next_steering_sign": [],
        "gt_reached": [],
        "pred_reached_prob": [],
        "gt_next_reached": [],
        "pred_next_reached_prob": [],
        "gt_state": [],
        "pred_state": [],
        "gt_next_state": [],
        "pred_next_state": [],
        "horizon_id": [],
        "route_distance": [],
        "start_frame_id": [],
        "current_frame_id": [],
        "goal_frame_id": [],
        "next_frame_id": [],
    }

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch["current_images"], batch["target_image"], batch["route_prior"], batch["horizon_id"])
        pred_command = torch.argmax(out["command_logits"], dim=-1)
        pred_next_command = torch.argmax(out["next_command_logits"], dim=-1)
        pred_steering_sign = torch.argmax(out["steering_sign_logits"], dim=-1)
        pred_next_steering_sign = torch.argmax(out["next_steering_sign_logits"], dim=-1)
        pred_reached_prob = torch.sigmoid(out["reached_logit"])
        pred_next_reached_prob = torch.sigmoid(out["next_reached_logit"])

        batch_np = {
            "gt_command": tensor_to_numpy(batch["command"]),
            "pred_command": tensor_to_numpy(pred_command),
            "gt_next_command": tensor_to_numpy(batch["next_command"]),
            "pred_next_command": tensor_to_numpy(pred_next_command),
            "gt_steering_sign": tensor_to_numpy(batch["steering_sign"]),
            "pred_steering_sign": tensor_to_numpy(pred_steering_sign),
            "gt_next_steering_sign": tensor_to_numpy(batch["next_steering_sign"]),
            "pred_next_steering_sign": tensor_to_numpy(pred_next_steering_sign),
            "gt_reached": tensor_to_numpy(batch["reached"]),
            "pred_reached_prob": tensor_to_numpy(pred_reached_prob),
            "gt_next_reached": tensor_to_numpy(batch["next_reached"]),
            "pred_next_reached_prob": tensor_to_numpy(pred_next_reached_prob),
            "gt_state": tensor_to_numpy(batch["current_state"]),
            "pred_state": tensor_to_numpy(out["current_state"]),
            "gt_next_state": tensor_to_numpy(batch["next_state"]),
            "pred_next_state": tensor_to_numpy(out["next_state"]),
            "horizon_id": tensor_to_numpy(batch["horizon_id"]),
            "route_distance": tensor_to_numpy(batch["route_distance_m"]),
            "start_frame_id": tensor_to_numpy(batch["start_frame_id"]),
            "current_frame_id": tensor_to_numpy(batch["current_frame_id"]),
            "goal_frame_id": tensor_to_numpy(batch["goal_frame_id"]),
            "next_frame_id": tensor_to_numpy(batch["next_frame_id"]),
        }
        for key, value in batch_np.items():
            arrays[key].append(np.asarray(value))

        start_paths = batch["start_path"]
        current_paths = batch["current_path"]
        goal_paths = batch["goal_path"]
        next_paths = batch["next_path"]
        gt_state = decode_state(batch_np["gt_state"], state_scale)
        pred_state = decode_state(batch_np["pred_state"], state_scale)
        gt_next_state = decode_state(batch_np["gt_next_state"], state_scale)
        pred_next_state = decode_state(batch_np["pred_next_state"], state_scale)
        for i in range(batch_np["gt_command"].shape[0]):
            rows.append(
                {
                    "start_frame_id": int(batch_np["start_frame_id"][i]),
                    "current_frame_id": int(batch_np["current_frame_id"][i]),
                    "goal_frame_id": int(batch_np["goal_frame_id"][i]),
                    "next_frame_id": int(batch_np["next_frame_id"][i]),
                    "start_path": start_paths[i],
                    "current_path": current_paths[i],
                    "goal_path": goal_paths[i],
                    "next_path": next_paths[i],
                    "horizon": HORIZONS[int(batch_np["horizon_id"][i])],
                    "route_distance_m": float(batch_np["route_distance"][i]),
                    "gt_command": COMMANDS[int(batch_np["gt_command"][i])],
                    "pred_command": COMMANDS[int(batch_np["pred_command"][i])],
                    "gt_next_command": COMMANDS[int(batch_np["gt_next_command"][i])],
                    "pred_next_command": COMMANDS[int(batch_np["pred_next_command"][i])],
                    "gt_steering_sign": STEERING_SIGNS[int(batch_np["gt_steering_sign"][i])],
                    "pred_steering_sign": STEERING_SIGNS[int(batch_np["pred_steering_sign"][i])],
                    "gt_next_steering_sign": STEERING_SIGNS[int(batch_np["gt_next_steering_sign"][i])],
                    "pred_next_steering_sign": STEERING_SIGNS[int(batch_np["pred_next_steering_sign"][i])],
                    "gt_reached": float(batch_np["gt_reached"][i]),
                    "pred_reached_prob": float(batch_np["pred_reached_prob"][i]),
                    "gt_dx_to_goal_m": float(gt_state["dx"][i]),
                    "gt_dy_to_goal_m": float(gt_state["dy"][i]),
                    "pred_dx_to_goal_m": float(pred_state["dx"][i]),
                    "pred_dy_to_goal_m": float(pred_state["dy"][i]),
                    "gt_remaining_m": float(gt_state["remaining"][i]),
                    "pred_remaining_m": float(pred_state["remaining"][i]),
                    "gt_progress": float(gt_state["progress"][i]),
                    "pred_progress": float(pred_state["progress"][i]),
                    "gt_heading_error_deg": float(math.degrees(gt_state["heading_error"][i])),
                    "pred_heading_error_deg": float(math.degrees(pred_state["heading_error"][i])),
                    "gt_next_remaining_m": float(gt_next_state["remaining"][i]),
                    "pred_next_remaining_m": float(pred_next_state["remaining"][i]),
                    "gt_next_progress": float(gt_next_state["progress"][i]),
                    "pred_next_progress": float(pred_next_state["progress"][i]),
                }
            )

    final_arrays = {key: np.concatenate(value, axis=0) if value else np.array([]) for key, value in arrays.items()}
    metrics = compute_metrics(final_arrays, state_scale, reached_threshold=args.reached_threshold)
    return metrics, rows, final_arrays


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


def confusion_matrix_np(gt: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    mat = np.zeros((n_classes, n_classes), dtype=np.int64)
    for g, p in zip(gt, pred):
        mat[int(g), int(p)] += 1
    return mat


def per_class_metrics(confusion: np.ndarray, names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(names):
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
    remaining_abs = np.abs(pred["remaining"] - gt["remaining"])
    progress_abs = np.abs(pred["progress"] - gt["progress"])
    heading_abs = angular_error_deg_np(pred["heading_error"], gt["heading_error"])
    return {
        f"{prefix}_dxdy_mean_m": safe_mean(rel_error),
        f"{prefix}_dxdy_median_m": safe_median(rel_error),
        f"{prefix}_dxdy_p90_m": safe_p90(rel_error),
        f"{prefix}_remaining_mae_m": safe_mean(remaining_abs),
        f"{prefix}_remaining_median_abs_m": safe_median(remaining_abs),
        f"{prefix}_progress_mae": safe_mean(progress_abs),
        f"{prefix}_heading_error_mean_deg": safe_mean(heading_abs),
        f"{prefix}_heading_error_median_deg": safe_median(heading_abs),
    }


def compute_metrics(arrays: Dict[str, np.ndarray], state_scale: float, reached_threshold: float) -> Dict[str, float]:
    gt_cmd = arrays["gt_command"].astype(np.int64)
    pred_cmd = arrays["pred_command"].astype(np.int64)
    confusion = confusion_matrix_np(gt_cmd, pred_cmd, len(COMMANDS))
    per_class = per_class_metrics(confusion, COMMANDS)
    gt_next_cmd = arrays["gt_next_command"].astype(np.int64)
    pred_next_cmd = arrays["pred_next_command"].astype(np.int64)
    gt_steering = arrays["gt_steering_sign"].astype(np.int64)
    pred_steering = arrays["pred_steering_sign"].astype(np.int64)
    steering_confusion = confusion_matrix_np(gt_steering, pred_steering, len(STEERING_SIGNS))
    steering_per_class = per_class_metrics(steering_confusion, STEERING_SIGNS)
    gt_next_steering = arrays["gt_next_steering_sign"].astype(np.int64)
    pred_next_steering = arrays["pred_next_steering_sign"].astype(np.int64)

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
        "next_command_accuracy": float(np.mean(gt_next_cmd == pred_next_cmd)) if gt_next_cmd.size else float("nan"),
        "macro_f1": float(np.nanmean([per_class[name]["f1"] for name in COMMANDS])),
        "balanced_accuracy": float(np.nanmean([per_class[name]["recall"] for name in COMMANDS])),
        "steering_sign_accuracy": float(np.mean(gt_steering == pred_steering)) if gt_steering.size else float("nan"),
        "steering_sign_balanced_accuracy": float(
            np.nanmean([steering_per_class[name]["recall"] for name in STEERING_SIGNS])
        ),
        "next_steering_sign_accuracy": float(np.mean(gt_next_steering == pred_next_steering))
        if gt_next_steering.size
        else float("nan"),
        "reached_accuracy": float(np.mean(gt_reached == pred_reached)) if gt_reached.size else float("nan"),
        "reached_precision": reached_precision,
        "reached_recall": reached_recall,
        "reached_f1": reached_f1,
    }
    for name in COMMANDS:
        metrics[f"{name.lower()}_precision"] = per_class[name]["precision"]
        metrics[f"{name.lower()}_recall"] = per_class[name]["recall"]
        metrics[f"{name.lower()}_f1"] = per_class[name]["f1"]
        metrics[f"{name.lower()}_support"] = float(per_class[name]["support"])
    for name in STEERING_SIGNS:
        key = name.lower()
        metrics[f"steering_{key}_precision"] = steering_per_class[name]["precision"]
        metrics[f"steering_{key}_recall"] = steering_per_class[name]["recall"]
        metrics[f"steering_{key}_f1"] = steering_per_class[name]["f1"]
        metrics[f"steering_{key}_support"] = float(steering_per_class[name]["support"])
    metrics.update(state_metrics("current", arrays["gt_state"], arrays["pred_state"], state_scale))
    metrics.update(state_metrics("next", arrays["gt_next_state"], arrays["pred_next_state"], state_scale))

    for horizon_id, horizon_name in enumerate(HORIZONS):
        mask = arrays["horizon_id"].astype(np.int64) == horizon_id
        if np.any(mask):
            metrics[f"{horizon_name}_samples"] = float(mask.sum())
            metrics[f"{horizon_name}_command_accuracy"] = float(np.mean(gt_cmd[mask] == pred_cmd[mask]))
            metrics[f"{horizon_name}_remaining_mae_m"] = state_metrics(
                horizon_name, arrays["gt_state"][mask], arrays["pred_state"][mask], state_scale
            )[f"{horizon_name}_remaining_mae_m"]
        else:
            metrics[f"{horizon_name}_samples"] = 0.0
            metrics[f"{horizon_name}_command_accuracy"] = float("nan")
            metrics[f"{horizon_name}_remaining_mae_m"] = float("nan")
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
    ax.set_title("Route-Conditioned Navigation Loss")
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
    ax.hist(values[np.isfinite(values)], bins=bins, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)
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


def plot_state_scatter(arrays: Dict[str, np.ndarray], scale: float, path: Path) -> Optional[str]:
    if plt is None or arrays["gt_state"].size == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    gt = decode_state(arrays["gt_state"], scale)
    pred = decode_state(arrays["pred_state"], scale)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=160)
    axes[0].scatter(gt["remaining"], pred["remaining"], s=8, alpha=0.45)
    lo = min(float(gt["remaining"].min()), float(pred["remaining"].min()))
    hi = max(float(gt["remaining"].max()), float(pred["remaining"].max()))
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=1)
    axes[0].set_xlabel("Ground truth remaining distance (m)")
    axes[0].set_ylabel("Predicted remaining distance (m)")
    axes[0].grid(True, alpha=0.3)
    axes[1].scatter(gt["progress"], pred["progress"], s=8, alpha=0.45)
    axes[1].plot([0, 1], [0, 1], "r--", lw=1)
    axes[1].set_xlabel("Ground truth route progress")
    axes[1].set_ylabel("Predicted route progress")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Route State Prediction")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_dxdy_scatter(arrays: Dict[str, np.ndarray], scale: float, path: Path) -> Optional[str]:
    if plt is None or arrays["gt_state"].size == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    gt = decode_state(arrays["gt_state"], scale)
    pred = decode_state(arrays["pred_state"], scale)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=160)
    for key, label, ax in [("dx", "dx to goal east (m)", axes[0]), ("dy", "dy to goal north (m)", axes[1])]:
        ax.scatter(gt[key], pred[key], s=8, alpha=0.45)
        lo = min(float(gt[key].min()), float(pred[key].min()))
        hi = max(float(gt[key].max()), float(pred[key].max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel(f"Ground truth {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Auxiliary dx/dy-to-Goal Prediction")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_horizon_accuracy(metrics: Dict[str, float], path: Path) -> Optional[str]:
    if plt is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [metrics.get(f"{name}_command_accuracy", float("nan")) for name in HORIZONS]
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=160)
    ax.bar(HORIZONS, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Command accuracy")
    ax.set_title("Command Accuracy by Horizon")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_route_pair_map(records: Sequence[FrameRecord], samples: Sequence[RouteSample], path: Path, max_routes: int = 700) -> Optional[str]:
    if plt is None or not samples:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, len(samples) - 1, min(max_routes, len(samples))).astype(int)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    for idx in indices:
        sample = samples[int(idx)]
        start = records[sample.start_idx]
        goal = records[sample.goal_idx]
        ax.plot([start.x, goal.x], [start.y, goal.y], color="#1f77b4", alpha=0.09, lw=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East x (m)")
    ax.set_ylabel("North y (m)")
    ax.set_title("Start-to-Goal Route Priors")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def save_plots(run_dir: Path, history: Sequence[Dict[str, float]], arrays: Dict[str, np.ndarray], state_scale: float, metrics: Dict[str, float], records: Sequence[FrameRecord], test_samples: Sequence[RouteSample]) -> None:
    plot_dir = run_dir / "plots"
    plot_loss_curves(history, plot_dir / "loss_curve.png")
    plot_confusion(arrays, plot_dir / "command_confusion_matrix.png")
    plot_command_distribution(arrays, plot_dir / "command_distribution.png")
    plot_state_scatter(arrays, state_scale, plot_dir / "state_prediction_scatter.png")
    plot_dxdy_scatter(arrays, state_scale, plot_dir / "dxdy_to_goal_scatter.png")
    plot_horizon_accuracy(metrics, plot_dir / "horizon_command_accuracy.png")
    gt = decode_state(arrays["gt_state"], state_scale)
    pred = decode_state(arrays["pred_state"], state_scale)
    plot_hist(np.abs(pred["remaining"] - gt["remaining"]), "Remaining Distance Absolute Error", "Error (m)", plot_dir / "remaining_distance_error_histogram.png")
    plot_hist(np.abs(pred["progress"] - gt["progress"]), "Route Progress Absolute Error", "Absolute error", plot_dir / "progress_error_histogram.png")
    plot_hist(angular_error_deg_np(pred["heading_error"], gt["heading_error"]), "Heading-Error Prediction Error", "Error (deg)", plot_dir / "heading_error_prediction_histogram.png")
    plot_route_pair_map(records, test_samples, plot_dir / "route_pair_map.png")


def make_run_dir(output_root: str, run_name: str) -> Path:
    root = Path(output_root)
    if not run_name:
        run_name = "route_conditioned_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / run_name
    for sub in ["checkpoints", "metrics", "plots", "predictions"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def maybe_limit(samples: List[RouteSample], limit: int, seed: int) -> List[RouteSample]:
    if limit <= 0 or len(samples) <= limit:
        return samples
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    keep = sorted(indices[:limit])
    return [samples[i] for i in keep]


def maybe_limit_eval_with_command_minimum(
    samples: List[RouteSample],
    limit: int,
    seed: int,
    min_per_command: int,
) -> List[RouteSample]:
    if limit <= 0 or len(samples) <= limit:
        return samples
    if min_per_command <= 0:
        return maybe_limit(samples, limit, seed)

    rng = random.Random(seed)
    by_command: Dict[int, List[int]] = {command_id: [] for command_id in range(len(COMMANDS))}
    for idx, sample in enumerate(samples):
        by_command[sample.command_id].append(idx)
    for indices in by_command.values():
        rng.shuffle(indices)

    selected = set()
    for command_id in range(len(COMMANDS)):
        if len(selected) >= limit:
            break
        reserve_count = min(min_per_command, len(by_command[command_id]), limit - len(selected))
        selected.update(by_command[command_id][:reserve_count])

    remaining = [idx for idx in range(len(samples)) if idx not in selected]
    rng.shuffle(remaining)
    selected.update(remaining[: max(0, limit - len(selected))])
    return [samples[i] for i in sorted(selected)]


def eval_sampling_summary(samples: Sequence[RouteSample], min_per_command: int) -> Dict[str, object]:
    counts = command_counts(samples)
    return {
        "min_command_samples_requested": int(min_per_command),
        "selected_command_counts": counts,
        "commands_below_requested_minimum": {
            name: count for name, count in counts.items() if min_per_command > 0 and count < min_per_command
        },
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Route-conditioned visual navigation with known start/goal coordinates.")
    parser.add_argument("--data_root", type=str, default=r"C:\Users\admin\Desktop\spiral_avgso\maps_screenshots_spiral_28150")
    parser.add_argument("--output_root", type=str, default=r"C:\Users\admin\Desktop\spiral_avgso\visual_navigation\results\stage1_route_conditioned")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--short_max_m", type=float, default=100.0)
    parser.add_argument("--medium_max_m", type=float, default=500.0)
    parser.add_argument("--long_max_m", type=float, default=1500.0)
    parser.add_argument(
        "--horizons",
        type=str,
        default="short,medium,long",
        help="Comma-separated active horizons: short, medium, long.",
    )
    parser.add_argument("--stop_threshold_m", type=float, default=8.0)
    parser.add_argument("--forward_angle_deg", type=float, default=15.0)
    parser.add_argument("--turn_around_angle_deg", type=float, default=135.0)
    parser.add_argument("--route_start_stride", type=int, default=8)
    parser.add_argument("--route_states", type=int, default=4)
    parser.add_argument("--targets_per_horizon", type=int, default=1)
    parser.add_argument("--candidate_trials", type=int, default=80)
    parser.add_argument("--max_offset_frames", type=int, default=900)
    parser.add_argument("--max_state_remaining_factor", type=float, default=1.25)
    parser.add_argument("--max_state_remaining_extra_m", type=float, default=100.0)
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

    parser.add_argument("--epochs", type=int, default=10)
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
    parser.add_argument("--w_state", type=float, default=0.35)
    parser.add_argument("--w_reached", type=float, default=0.50)
    parser.add_argument("--w_heading_class", type=float, default=0.20)
    parser.add_argument("--w_steering_sign", type=float, default=0.40)
    parser.add_argument("--w_next_command", type=float, default=0.25)
    parser.add_argument("--w_next_state", type=float, default=0.25)
    parser.add_argument("--w_next_reached", type=float, default=0.20)
    parser.add_argument("--w_next_heading_class", type=float, default=0.10)
    parser.add_argument("--w_next_steering_sign", type=float, default=0.20)
    parser.add_argument("--w_state_rel", type=float, default=1.0)
    parser.add_argument("--w_state_remaining", type=float, default=0.75)
    parser.add_argument("--w_state_progress", type=float, default=0.75)
    parser.add_argument("--w_state_heading", type=float, default=0.50)

    parser.add_argument("--dry_run", type=int, default=0)
    parser.add_argument("--eval_checkpoint", type=str, default="")
    parser.add_argument("--limit_train_samples", type=int, default=0)
    parser.add_argument("--limit_eval_samples", type=int, default=0)
    parser.add_argument(
        "--min_train_command_samples",
        type=int,
        default=0,
        help="Minimum training samples to reserve per command before random filling. Keeps natural imbalance after reservation.",
    )
    parser.add_argument(
        "--min_eval_command_samples",
        type=int,
        default=0,
        help="Minimum validation/test samples to reserve per command before random filling.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    active_horizons = parse_horizons(args.horizons)
    active_horizon_ids = [HORIZON_TO_ID[name] for name in active_horizons]
    active_max_distance_m = max_distance_for_horizons(
        active_horizons,
        args.short_max_m,
        args.medium_max_m,
        args.long_max_m,
    )
    set_seed(args.seed)
    run_dir = make_run_dir(args.output_root, args.run_name)
    save_json(run_dir / "config.json", vars(args))

    print(f"Data root: {args.data_root}")
    print(f"Run directory: {run_dir}")
    print(
        "Route horizons: "
        f"short< {args.short_max_m:.1f} m | "
        f"medium {args.short_max_m:.1f}-{args.medium_max_m:.1f} m | "
        f"long {args.medium_max_m:.1f}-{args.long_max_m:.1f} m"
    )
    print(f"Active horizons: {', '.join(active_horizons)}")

    records = list_frames_one_folder(args.data_root)
    ranges = split_ranges(len(records), args.train_frac, args.val_frac, args.split_gap)
    forward_angle_rad = math.radians(args.forward_angle_deg)
    turn_around_angle_rad = math.radians(args.turn_around_angle_deg)

    build_kwargs = dict(
        seq_len=args.seq_len,
        route_start_stride=args.route_start_stride,
        route_states=args.route_states,
        targets_per_horizon=args.targets_per_horizon,
        candidate_trials=args.candidate_trials,
        max_offset_frames=args.max_offset_frames,
        short_max_m=args.short_max_m,
        medium_max_m=args.medium_max_m,
        long_max_m=args.long_max_m,
        stop_threshold_m=args.stop_threshold_m,
        forward_angle_rad=forward_angle_rad,
        turn_around_angle_rad=turn_around_angle_rad,
        max_state_remaining_factor=args.max_state_remaining_factor,
        max_state_remaining_extra_m=args.max_state_remaining_extra_m,
        max_allowed_state_distance_m=active_max_distance_m,
        active_horizon_ids=active_horizon_ids,
    )
    train_samples = build_route_samples_for_split(records, ranges["train"], seed=args.seed, **build_kwargs)
    val_samples = build_route_samples_for_split(records, ranges["val"], seed=args.seed + 1, **build_kwargs)
    test_samples = build_route_samples_for_split(records, ranges["test"], seed=args.seed + 2, **build_kwargs)
    train_samples = maybe_limit_eval_with_command_minimum(
        train_samples,
        args.limit_train_samples,
        args.seed,
        args.min_train_command_samples,
    )
    val_samples = maybe_limit_eval_with_command_minimum(
        val_samples,
        args.limit_eval_samples,
        args.seed + 1,
        args.min_eval_command_samples,
    )
    test_samples = maybe_limit_eval_with_command_minimum(
        test_samples,
        args.limit_eval_samples,
        args.seed + 2,
        args.min_eval_command_samples,
    )

    normalization = compute_normalization(records, ranges["train"], train_samples, active_max_distance_m)
    train_counts = command_counts(train_samples)
    val_counts = command_counts(val_samples)
    test_counts = command_counts(test_samples)
    train_steering_counts = steering_counts(train_samples)
    val_steering_counts = steering_counts(val_samples)
    test_steering_counts = steering_counts(test_samples)
    summary = {
        "num_frames": len(records),
        "split_ranges": {k: list(v) for k, v in ranges.items()},
        "samples": {"train": len(train_samples), "val": len(val_samples), "test": len(test_samples)},
        "command_counts": {"train": train_counts, "val": val_counts, "test": test_counts},
        "steering_sign_counts": {
            "train": train_steering_counts,
            "val": val_steering_counts,
            "test": test_steering_counts,
        },
        "horizon_counts": {
            "train": horizon_counts(train_samples),
            "val": horizon_counts(val_samples),
            "test": horizon_counts(test_samples),
        },
        "route_state_stats": {
            "train": route_state_stats(train_samples),
            "val": route_state_stats(val_samples),
            "test": route_state_stats(test_samples),
        },
        "command_geometry_stats": {
            "train": command_geometry_stats(train_samples),
            "val": command_geometry_stats(val_samples),
            "test": command_geometry_stats(test_samples),
        },
        "normalization": normalization,
        "command_names": COMMANDS,
        "horizon_names": HORIZONS,
        "steering_sign_names": STEERING_SIGNS,
        "active_horizons": active_horizons,
        "active_max_distance_m": active_max_distance_m,
        "command_minimum_sampling": {
            "train": eval_sampling_summary(train_samples, args.min_train_command_samples),
            "val": eval_sampling_summary(val_samples, args.min_eval_command_samples),
            "test": eval_sampling_summary(test_samples, args.min_eval_command_samples),
        },
        "assumption": "Model input contains start/goal route prior but not intermediate/current coordinates.",
    }
    save_json(run_dir / "metrics" / "dataset_summary.json", summary)
    save_json(run_dir / "metrics" / "normalization_stats.json", normalization)
    print(f"Frames: {len(records)}")
    print(f"Split ranges: {ranges}")
    print(f"Samples: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")
    print(f"Train command counts: {train_counts}")
    print(f"Train steering counts: {train_steering_counts}")
    print(f"Train horizon counts: {horizon_counts(train_samples)}")
    print(f"Val command counts:   {val_counts}")
    print(f"Test command counts:  {test_counts}")

    if args.dry_run:
        print("Dry run complete. No model was trained.")
        return
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not available in this Python environment. Run training from the same Anaconda "
            "environment used for the localization models, or use --dry_run 1 for metadata checks."
        )

    train_ds = RouteConditionedDataset(records, train_samples, normalization, args.seq_len, args.image_size)
    val_ds = RouteConditionedDataset(records, val_samples, normalization, args.seq_len, args.image_size)
    test_ds = RouteConditionedDataset(records, test_samples, normalization, args.seq_len, args.image_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RouteConditionedNavModel(
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
    steering_weights = class_weights_from_named_counts(train_steering_counts, STEERING_SIGNS).to(device)
    reached_pos_weight = torch.tensor(pos_weight_from_samples(train_samples), dtype=torch.float32, device=device)

    print(f"Device: {device}")
    print(f"Command loss weights: {tensor_to_numpy(command_weights)}")
    print(f"Steering-sign loss weights: {tensor_to_numpy(steering_weights)}")
    print(f"Reached positive weight: {float(reached_pos_weight.detach().cpu()):.4f}")
    print("Current coordinates are not model inputs; they are labels only.")

    if args.eval_checkpoint:
        checkpoint_path = Path(args.eval_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"eval_checkpoint not found: {checkpoint_path}")
        print(f"Loading checkpoint for evaluation: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        metrics, rows, arrays = evaluate_full(
            model,
            test_loader,
            device,
            float(normalization["state_distance_scale"]),
            args,
        )
        test_logs = evaluate_loss(model, test_loader, device, command_weights, steering_weights, reached_pos_weight, args)
        metrics["test_loss"] = test_logs["loss"]
        metrics["checkpoint_epoch"] = float(checkpoint.get("epoch", float("nan")))
        metrics["checkpoint_best_val_loss"] = float(checkpoint.get("best_val_loss", float("nan")))
        save_metrics_csv(run_dir / "metrics" / "final_navigation_metrics.csv", metrics)
        save_predictions_csv(run_dir / "predictions" / "test_per_sample_predictions.csv", rows)
        save_plots(run_dir, [], arrays, float(normalization["state_distance_scale"]), metrics, records, test_samples)
        print(f"Saved checkpoint evaluation metrics to {run_dir / 'metrics' / 'final_navigation_metrics.csv'}")
        print(
            f"Command acc: {metrics['command_accuracy']:.4f} | "
            f"Balanced acc: {metrics['balanced_accuracy']:.4f} | "
            f"Current remaining MAE: {metrics['current_remaining_mae_m']:.2f} m"
        )
        return

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
            steering_weights,
            reached_pos_weight,
            args,
            epoch,
            args.epochs,
            live_progress_path,
        )
        val_logs = evaluate_loss(model, val_loader, device, command_weights, steering_weights, reached_pos_weight, args)
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
            f"cmd={val_logs['command']:.4f} state={val_logs['state']:.4f} "
            f"next={val_logs['next_state']:.4f}",
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
    print("Running final test evaluation...")
    metrics, rows, arrays = evaluate_full(
        model,
        test_loader,
        device,
        float(normalization["state_distance_scale"]),
        args,
    )
    metrics["best_val_loss"] = float(best_val)
    test_logs = evaluate_loss(model, test_loader, device, command_weights, steering_weights, reached_pos_weight, args)
    metrics["test_loss"] = test_logs["loss"]
    save_metrics_csv(run_dir / "metrics" / "final_navigation_metrics.csv", metrics)
    save_predictions_csv(run_dir / "predictions" / "test_per_sample_predictions.csv", rows)
    save_plots(run_dir, history, arrays, float(normalization["state_distance_scale"]), metrics, records, test_samples)
    write_live_progress(live_progress_path, "Finished final evaluation.")
    print(f"Saved metrics to {run_dir / 'metrics' / 'final_navigation_metrics.csv'}")
    print(f"Saved plots to {run_dir / 'plots'}")
    print(f"Best val loss: {best_val:.4f} | Test loss: {metrics['test_loss']:.4f}")
    print(
        f"Command acc: {metrics['command_accuracy']:.4f} | "
        f"Balanced acc: {metrics['balanced_accuracy']:.4f} | "
        f"Current remaining MAE: {metrics['current_remaining_mae_m']:.2f} m"
    )


if __name__ == "__main__":
    main()
