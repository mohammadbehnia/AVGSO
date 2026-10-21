# -*- coding: utf-8 -*-
"""
Stage 2B localization-guided continuous heading navigation.

Assumption:
  The model receives a current multi-frame visual sequence, a target image from
  current+horizon, a horizon token, and a reliable current heading prior from
  localization. It does not receive current or target coordinates as inputs.
  Coordinates are used only to build heading-correction and distance labels.

Outputs:
  - continuous heading correction, represented as sin(delta), cos(delta)
  - target distance
  - reached probability

This script intentionally does not use discrete command classification as the
main navigation target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import stage2_multihorizon_visual_nav as mh

TORCH_AVAILABLE = mh.TORCH_AVAILABLE
torch = mh.torch
nn = mh.nn
F = mh.F
DataLoader = mh.DataLoader
Dataset = mh.Dataset
T = mh.T
Image = mh.Image
plt = mh.plt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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


def make_run_dir(output_root: str, run_name: str) -> Path:
    root = Path(output_root)
    if not run_name:
        run_name = "stage2b_heading_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / run_name
    for sub in ["checkpoints", "metrics", "plots", "predictions"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def heading_prior(rec: mh.FrameRecord) -> np.ndarray:
    return np.array([math.sin(rec.heading_rad), math.cos(rec.heading_rad)], dtype=np.float32)


class HeadingNavigationDataset(Dataset):
    def __init__(
        self,
        records: Sequence[mh.FrameRecord],
        samples: Sequence[mh.MultiHorizonSample],
        state_distance_scale: float,
        seq_len: int,
        image_size: int,
    ):
        self.records = records
        self.samples = samples
        self.state_distance_scale = float(max(state_distance_scale, 1.0))
        self.seq_len = seq_len
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        seq_indices = range(sample.current_idx - self.seq_len + 1, sample.current_idx + 1)
        current_images = torch.stack([self._load_norm(self.records[i].path) for i in seq_indices], dim=0)
        target_image = self._load_norm(self.records[sample.target_idx].path)
        current_rec = self.records[sample.current_idx]
        return {
            "current_images": current_images,
            "target_image": target_image,
            "horizon_id": torch.tensor(sample.horizon_id, dtype=torch.long),
            "heading_prior": torch.tensor(heading_prior(current_rec), dtype=torch.float32),
            "heading_target": torch.tensor(
                [math.sin(sample.heading_error_rad), math.cos(sample.heading_error_rad)], dtype=torch.float32
            ),
            "distance_target": torch.tensor(sample.distance_m / self.state_distance_scale, dtype=torch.float32),
            "gt_dx_to_target_m": torch.tensor(sample.dx_to_target, dtype=torch.float32),
            "gt_dy_to_target_m": torch.tensor(sample.dy_to_target, dtype=torch.float32),
            "reached": torch.tensor(sample.reached, dtype=torch.float32),
            "offset": torch.tensor(sample.offset, dtype=torch.long),
            "gt_heading_error": torch.tensor(sample.heading_error_rad, dtype=torch.float32),
            "gt_distance_m": torch.tensor(sample.distance_m, dtype=torch.float32),
            "current_frame_id": torch.tensor(current_rec.frame_id, dtype=torch.long),
            "target_frame_id": torch.tensor(self.records[sample.target_idx].frame_id, dtype=torch.long),
            "current_path": Path(current_rec.path).name,
            "target_path": Path(self.records[sample.target_idx].path).name,
        }


class ContinuousHeadingModel(nn.Module):
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
        self.stem = mh.VGGTokenStem(backbone, d_model, use_pretrained, freeze_backbone)
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
        self.horizon_emb = nn.Embedding(len(mh.HORIZONS), d_model)
        self.heading_mlp = nn.Sequential(nn.LayerNorm(2), nn.Linear(2, d_model), nn.GELU(), nn.Dropout(dropout))

        fusion_dim = d_model * 7
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heading_head = self._head(d_model, 2, dropout)
        self.distance_head = self._head(d_model, 1, dropout)
        self.reached_head = self._head(d_model, 1, dropout)

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

    def forward(
        self,
        current_images: torch.Tensor,
        target_image: torch.Tensor,
        horizon_id: torch.Tensor,
        heading_prior_tensor: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
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
        heading_repr = self.heading_mlp(heading_prior_tensor)
        fusion_input = torch.cat(
            [
                current_repr,
                target_repr,
                current_repr - target_repr,
                torch.abs(current_repr - target_repr),
                current_repr * target_repr,
                horizon_repr,
                heading_repr,
            ],
            dim=-1,
        )
        fused = self.fusion(fusion_input)
        heading = F.normalize(self.heading_head(fused), dim=-1, eps=1e-6)
        distance = self.distance_head(fused).squeeze(-1)
        return {
            "heading": heading,
            "distance": distance,
            "reached_logit": self.reached_head(fused).squeeze(-1),
        }


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved: Dict[str, object] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def circular_heading_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1, eps=1e-6)
    target = F.normalize(target, dim=-1, eps=1e-6)
    cos_sim = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cos_sim).mean()


def total_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    reached_pos_weight: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    heading_loss = circular_heading_loss(out["heading"], batch["heading_target"])
    distance_loss = F.smooth_l1_loss(out["distance"], batch["distance_target"], beta=args.huber_beta)
    reached_loss = F.binary_cross_entropy_with_logits(out["reached_logit"], batch["reached"], pos_weight=reached_pos_weight)
    loss = args.w_heading * heading_loss + args.w_distance * distance_loss + args.w_reached * reached_loss
    return loss, {
        "loss": float(loss.detach().cpu()),
        "heading": float(heading_loss.detach().cpu()),
        "distance": float(distance_loss.detach().cpu()),
        "reached": float(reached_loss.detach().cpu()),
    }


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


def pos_weight_from_samples(samples: Sequence[mh.MultiHorizonSample]) -> float:
    positives = sum(1 for s in samples if s.reached > 0.5)
    negatives = max(len(samples) - positives, 1)
    return float(negatives / max(positives, 1))


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    reached_pos_weight: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch["current_images"], batch["target_image"], batch["horizon_id"], batch["heading_prior"])
        _, logs = total_loss(out, batch, reached_pos_weight, args)
        bs = batch["current_images"].size(0)
        average_logs(totals, logs, bs)
        count += bs
    return finalize_logs(totals, count)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
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
            out = model(batch["current_images"], batch["target_image"], batch["horizon_id"], batch["heading_prior"])
            loss, logs = total_loss(out, batch, reached_pos_weight, args)
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
                f"loss={running['loss']:.4f} heading={running['heading']:.4f} "
                f"dist={running['distance']:.4f} elapsed={elapsed/60:.1f}m eta={remaining/60:.1f}m"
            )
            print(message, flush=True)
            write_live_progress(live_progress_path, message)
    return finalize_logs(totals, count)


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def angle_from_sincos(arr: np.ndarray) -> np.ndarray:
    return np.arctan2(arr[:, 0], arr[:, 1])


def angular_error_deg(pred_angle: np.ndarray, gt_angle: np.ndarray) -> np.ndarray:
    diff = np.arctan2(np.sin(pred_angle - gt_angle), np.cos(pred_angle - gt_angle))
    return np.degrees(np.abs(diff))


def vector_from_heading_distance(heading_prior: np.ndarray, correction: np.ndarray, distance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    current_heading = angle_from_sincos(heading_prior)
    correction_angle = angle_from_sincos(correction)
    target_direction = current_heading + correction_angle
    dx = distance * np.sin(target_direction)
    dy = distance * np.cos(target_direction)
    return dx, dy


def safe_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def safe_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def safe_p90(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, 0.90)) if finite.size else float("nan")


@torch.no_grad()
def evaluate_full(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    distance_scale: float,
    reached_threshold: float,
) -> Tuple[Dict[str, float], List[Dict[str, object]], Dict[str, np.ndarray]]:
    model.eval()
    rows: List[Dict[str, object]] = []
    arrays: Dict[str, List[np.ndarray]] = {
        "gt_heading": [],
        "pred_heading": [],
        "gt_distance": [],
        "pred_distance": [],
        "gt_dx": [],
        "gt_dy": [],
        "pred_dx": [],
        "pred_dy": [],
        "heading_prior": [],
        "gt_reached": [],
        "pred_reached_prob": [],
        "horizon_id": [],
        "offset": [],
    }
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch["current_images"], batch["target_image"], batch["horizon_id"], batch["heading_prior"])
        pred_reached_prob = torch.sigmoid(out["reached_logit"])
        batch_np = {
            "gt_heading": tensor_to_numpy(batch["heading_target"]),
            "pred_heading": tensor_to_numpy(out["heading"]),
            "gt_distance": tensor_to_numpy(batch["distance_target"]) * distance_scale,
            "pred_distance": tensor_to_numpy(out["distance"]) * distance_scale,
            "gt_dx": tensor_to_numpy(batch["gt_dx_to_target_m"]),
            "gt_dy": tensor_to_numpy(batch["gt_dy_to_target_m"]),
            "heading_prior": tensor_to_numpy(batch["heading_prior"]),
            "gt_reached": tensor_to_numpy(batch["reached"]),
            "pred_reached_prob": tensor_to_numpy(pred_reached_prob),
            "horizon_id": tensor_to_numpy(batch["horizon_id"]),
            "offset": tensor_to_numpy(batch["offset"]),
            "current_frame_id": tensor_to_numpy(batch["current_frame_id"]),
            "target_frame_id": tensor_to_numpy(batch["target_frame_id"]),
        }
        batch_np["pred_dx"], batch_np["pred_dy"] = vector_from_heading_distance(
            batch_np["heading_prior"], batch_np["pred_heading"], batch_np["pred_distance"]
        )
        for key in arrays:
            arrays[key].append(np.asarray(batch_np[key]))

        gt_angle = angle_from_sincos(batch_np["gt_heading"])
        pred_angle = angle_from_sincos(batch_np["pred_heading"])
        current_paths = batch["current_path"]
        target_paths = batch["target_path"]
        for i in range(batch_np["horizon_id"].shape[0]):
            distance_abs_error = float(abs(batch_np["pred_distance"][i] - batch_np["gt_distance"][i]))
            relative_distance_error = 100.0 * distance_abs_error / max(abs(float(batch_np["gt_distance"][i])), 1.0)
            rows.append(
                {
                    "current_frame_id": int(batch_np["current_frame_id"][i]),
                    "target_frame_id": int(batch_np["target_frame_id"][i]),
                    "current_path": current_paths[i],
                    "target_path": target_paths[i],
                    "horizon": mh.HORIZONS[int(batch_np["horizon_id"][i])],
                    "offset": int(batch_np["offset"][i]),
                    "gt_heading_error_deg": float(math.degrees(gt_angle[i])),
                    "pred_heading_error_deg": float(math.degrees(pred_angle[i])),
                    "heading_abs_error_deg": float(angular_error_deg(pred_angle[i : i + 1], gt_angle[i : i + 1])[0]),
                    "gt_dx_to_target_m": float(batch_np["gt_dx"][i]),
                    "gt_dy_to_target_m": float(batch_np["gt_dy"][i]),
                    "pred_dx_to_target_m": float(batch_np["pred_dx"][i]),
                    "pred_dy_to_target_m": float(batch_np["pred_dy"][i]),
                    "gt_distance_m": float(batch_np["gt_distance"][i]),
                    "pred_distance_m": float(batch_np["pred_distance"][i]),
                    "distance_abs_error_m": distance_abs_error,
                    "relative_distance_error_pct": relative_distance_error,
                    "gt_reached": float(batch_np["gt_reached"][i]),
                    "pred_reached_prob": float(batch_np["pred_reached_prob"][i]),
                }
            )

    final_arrays = {key: np.concatenate(value, axis=0) if value else np.array([]) for key, value in arrays.items()}
    metrics = compute_metrics(final_arrays, reached_threshold)
    return metrics, rows, final_arrays


def compute_metrics(arrays: Dict[str, np.ndarray], reached_threshold: float) -> Dict[str, float]:
    gt_angle = angle_from_sincos(arrays["gt_heading"])
    pred_angle = angle_from_sincos(arrays["pred_heading"])
    heading_abs = angular_error_deg(pred_angle, gt_angle)
    distance_abs = np.abs(arrays["pred_distance"] - arrays["gt_distance"])
    distance_rel_pct = 100.0 * distance_abs / np.maximum(np.abs(arrays["gt_distance"]), 1.0)
    dxdy_abs = np.sqrt((arrays["pred_dx"] - arrays["gt_dx"]) ** 2 + (arrays["pred_dy"] - arrays["gt_dy"]) ** 2)
    gt_reached = arrays["gt_reached"] > 0.5
    pred_reached = arrays["pred_reached_prob"] >= reached_threshold
    metrics: Dict[str, float] = {
        "num_samples": float(gt_angle.size),
        "heading_error_mean_deg": safe_mean(heading_abs),
        "heading_error_median_deg": safe_median(heading_abs),
        "heading_error_p90_deg": safe_p90(heading_abs),
        "distance_mae_m": safe_mean(distance_abs),
        "distance_median_abs_m": safe_median(distance_abs),
        "distance_p90_abs_m": safe_p90(distance_abs),
        "dxdy_target_mean_m": safe_mean(dxdy_abs),
        "dxdy_target_median_m": safe_median(dxdy_abs),
        "dxdy_target_p90_m": safe_p90(dxdy_abs),
        "relative_distance_error_mean_pct": safe_mean(distance_rel_pct),
        "relative_distance_error_median_pct": safe_median(distance_rel_pct),
        "relative_distance_error_p90_pct": safe_p90(distance_rel_pct),
        "reached_accuracy": float(np.mean(gt_reached == pred_reached)) if gt_reached.size else float("nan"),
        "reached_positive_support": float(gt_reached.sum()),
    }
    for horizon_id, horizon_name in enumerate(mh.HORIZONS):
        mask = arrays["horizon_id"].astype(np.int64) == horizon_id
        if np.any(mask):
            h_err = heading_abs[mask]
            d_err = distance_abs[mask]
            d_rel = distance_rel_pct[mask]
            v_err = dxdy_abs[mask]
            metrics[f"{horizon_name}_samples"] = float(mask.sum())
            metrics[f"{horizon_name}_heading_error_mean_deg"] = safe_mean(h_err)
            metrics[f"{horizon_name}_heading_error_median_deg"] = safe_median(h_err)
            metrics[f"{horizon_name}_heading_error_p90_deg"] = safe_p90(h_err)
            metrics[f"{horizon_name}_distance_mae_m"] = safe_mean(d_err)
            metrics[f"{horizon_name}_dxdy_target_mean_m"] = safe_mean(v_err)
            metrics[f"{horizon_name}_dxdy_target_median_m"] = safe_median(v_err)
            metrics[f"{horizon_name}_dxdy_target_p90_m"] = safe_p90(v_err)
            metrics[f"{horizon_name}_relative_distance_error_mean_pct"] = safe_mean(d_rel)
            metrics[f"{horizon_name}_relative_distance_error_median_pct"] = safe_median(d_rel)
        else:
            metrics[f"{horizon_name}_samples"] = 0.0
            metrics[f"{horizon_name}_heading_error_mean_deg"] = float("nan")
            metrics[f"{horizon_name}_heading_error_median_deg"] = float("nan")
            metrics[f"{horizon_name}_heading_error_p90_deg"] = float("nan")
            metrics[f"{horizon_name}_distance_mae_m"] = float("nan")
            metrics[f"{horizon_name}_dxdy_target_mean_m"] = float("nan")
            metrics[f"{horizon_name}_dxdy_target_median_m"] = float("nan")
            metrics[f"{horizon_name}_dxdy_target_p90_m"] = float("nan")
            metrics[f"{horizon_name}_relative_distance_error_mean_pct"] = float("nan")
            metrics[f"{horizon_name}_relative_distance_error_median_pct"] = float("nan")
    return metrics


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
    ax.set_title("Continuous Heading Navigation Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_heading_scatter(arrays: Dict[str, np.ndarray], path: Path) -> Optional[str]:
    if plt is None or arrays["gt_heading"].size == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    gt = np.degrees(angle_from_sincos(arrays["gt_heading"]))
    pred = np.degrees(angle_from_sincos(arrays["pred_heading"]))
    fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
    ax.scatter(gt, pred, s=8, alpha=0.35, label="Test samples")
    ax.plot([-180, 180], [-180, 180], "r--", linewidth=1, label="Ideal prediction")
    ax.set_xlabel("GT heading correction (deg)")
    ax.set_ylabel("Pred heading correction (deg)")
    ax.set_title("Heading Correction Prediction")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_distance_scatter(arrays: Dict[str, np.ndarray], path: Path) -> Optional[str]:
    if plt is None or arrays["gt_distance"].size == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    gt = arrays["gt_distance"]
    pred = arrays["pred_distance"]
    fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
    ax.scatter(gt, pred, s=8, alpha=0.35, label="Test samples")
    lim = max(float(np.nanmax(gt)), float(np.nanmax(pred)), 1.0)
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="Ideal prediction")
    ax.set_xlabel("GT distance (m)")
    ax.set_ylabel("Pred distance (m)")
    ax.set_title("Target Distance Prediction")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_dxdy_target_scatter(arrays: Dict[str, np.ndarray], path: Path) -> Optional[str]:
    if plt is None or arrays["gt_dx"].size == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    gt_dx = arrays["gt_dx"]
    gt_dy = arrays["gt_dy"]
    pred_dx = arrays["pred_dx"]
    pred_dy = arrays["pred_dy"]
    lim = max(
        float(np.nanmax(np.abs(gt_dx))),
        float(np.nanmax(np.abs(gt_dy))),
        float(np.nanmax(np.abs(pred_dx))),
        float(np.nanmax(np.abs(pred_dy))),
        1.0,
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=160)
    axes[0].scatter(gt_dx, pred_dx, s=8, alpha=0.35, color="#4c78a8", label="Test samples")
    axes[0].plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="Ideal prediction")
    axes[0].set_xlabel("GT dx to target (m)")
    axes[0].set_ylabel("Pred dx to target (m)")
    axes[0].set_title("East-West Target Vector")
    axes[1].scatter(gt_dy, pred_dy, s=8, alpha=0.35, color="#f58518", label="Test samples")
    axes[1].plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="Ideal prediction")
    axes[1].set_xlabel("GT dy to target (m)")
    axes[1].set_ylabel("Pred dy to target (m)")
    axes[1].set_title("North-South Target Vector")
    for ax in axes:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_target_vector_quiver(arrays: Dict[str, np.ndarray], path: Path, max_vectors: int = 160) -> Optional[str]:
    if plt is None or arrays["gt_dx"].size == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    total = arrays["gt_dx"].shape[0]
    if total > max_vectors:
        indices = np.linspace(0, total - 1, max_vectors, dtype=np.int64)
    else:
        indices = np.arange(total, dtype=np.int64)
    gt_dx = arrays["gt_dx"][indices]
    gt_dy = arrays["gt_dy"][indices]
    pred_dx = arrays["pred_dx"][indices]
    pred_dy = arrays["pred_dy"][indices]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ax.quiver(
        np.zeros_like(gt_dx),
        np.zeros_like(gt_dy),
        gt_dx,
        gt_dy,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.003,
        alpha=0.55,
        color="#4c78a8",
        label="GT",
    )
    ax.quiver(
        np.zeros_like(pred_dx),
        np.zeros_like(pred_dy),
        pred_dx,
        pred_dy,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.003,
        alpha=0.55,
        color="#e45756",
        label="Pred",
    )
    lim = max(
        float(np.nanmax(np.abs(gt_dx))) if gt_dx.size else 1.0,
        float(np.nanmax(np.abs(gt_dy))) if gt_dy.size else 1.0,
        float(np.nanmax(np.abs(pred_dx))) if pred_dx.size else 1.0,
        float(np.nanmax(np.abs(pred_dy))) if pred_dy.size else 1.0,
        1.0,
    )
    ax.set_xlim(-lim * 1.1, lim * 1.1)
    ax.set_ylim(-lim * 1.1, lim * 1.1)
    ax.set_xlabel("dx to target (m)")
    ax.set_ylabel("dy to target (m)")
    ax.set_title("Predicted and GT Target Direction Vectors")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_horizon_errors(metrics: Dict[str, float], path: Path) -> Optional[str]:
    if plt is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    heading = [metrics.get(f"{name}_heading_error_median_deg", float("nan")) for name in mh.HORIZONS]
    distance = [metrics.get(f"{name}_distance_mae_m", float("nan")) for name in mh.HORIZONS]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    axes[0].bar(mh.HORIZONS, heading, color="#4c78a8", label="Median heading error")
    axes[0].set_ylabel("Median heading error (deg)")
    axes[0].set_title("Heading by Horizon")
    axes[1].bar(mh.HORIZONS, distance, color="#f58518", label="Distance MAE")
    axes[1].set_ylabel("Distance MAE (m)")
    axes[1].set_title("Distance by Horizon")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_horizon_metric_bar(metrics: Dict[str, float], metric_suffix: str, ylabel: str, title: str, path: Path) -> Optional[str]:
    if plt is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [metrics.get(f"{name}_{metric_suffix}", float("nan")) for name in mh.HORIZONS]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    bars = ax.bar(mh.HORIZONS, values, color=["#4c78a8", "#f58518", "#54a24b", "#e45756"], label=title)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def save_plots(run_dir: Path, history: Sequence[Dict[str, float]], arrays: Dict[str, np.ndarray], metrics: Dict[str, float]) -> None:
    plot_dir = run_dir / "plots"
    plot_loss_curves(history, plot_dir / "loss_curve.png")
    plot_heading_scatter(arrays, plot_dir / "heading_correction_scatter.png")
    plot_distance_scatter(arrays, plot_dir / "distance_prediction_scatter.png")
    plot_dxdy_target_scatter(arrays, plot_dir / "dxdy_to_target_scatter.png")
    plot_target_vector_quiver(arrays, plot_dir / "target_vector_quiver.png")
    plot_horizon_errors(metrics, plot_dir / "horizon_errors.png")
    plot_horizon_metric_bar(
        metrics,
        "distance_mae_m",
        "Distance MAE (m)",
        "Target Distance Error by Horizon",
        plot_dir / "distance_mae_by_horizon.png",
    )
    plot_horizon_metric_bar(
        metrics,
        "heading_error_median_deg",
        "Median heading error (deg)",
        "Heading Correction Error by Horizon",
        plot_dir / "heading_error_by_horizon.png",
    )
    plot_horizon_metric_bar(
        metrics,
        "relative_distance_error_mean_pct",
        "Mean relative distance error (%)",
        "Relative Distance Error by Horizon",
        plot_dir / "relative_distance_error_by_horizon.png",
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2B continuous heading-correction navigation.")
    parser.add_argument("--data_root", type=str, default=r"C:\Users\admin\Desktop\spiral_avgso\maps_screenshots_spiral_28150")
    parser.add_argument("--output_root", type=str, default=r"C:\Users\admin\Desktop\spiral_avgso\visual_navigation\results\stage2_heading")
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
    parser.add_argument("--freeze_backbone", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)

    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--huber_beta", type=float, default=0.1)
    parser.add_argument("--progress_every", type=int, default=0)
    parser.add_argument("--reached_threshold", type=float, default=0.5)

    parser.add_argument("--w_heading", type=float, default=1.0)
    parser.add_argument("--w_distance", type=float, default=0.45)
    parser.add_argument("--w_reached", type=float, default=0.10)

    parser.add_argument("--dry_run", type=int, default=0)
    parser.add_argument("--limit_train_samples", type=int, default=0)
    parser.add_argument("--limit_eval_samples", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    offsets = mh.parse_offsets(args.offsets)
    mh.update_horizon_names(offsets)
    set_seed(args.seed)
    run_dir = make_run_dir(args.output_root, args.run_name)
    save_json(run_dir / "config.json", vars(args))

    print(f"Data root: {args.data_root}")
    print(f"Run directory: {run_dir}")
    print(f"Offsets: {offsets}")
    records = mh.list_frames_one_folder(args.data_root)
    ranges = mh.split_ranges(len(records), args.train_frac, args.val_frac, args.split_gap)
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
    train_samples = mh.build_samples_for_split(records, ranges["train"], **build_kwargs)
    val_samples = mh.build_samples_for_split(records, ranges["val"], **build_kwargs)
    test_samples = mh.build_samples_for_split(records, ranges["test"], **build_kwargs)
    train_samples = mh.maybe_limit(train_samples, args.limit_train_samples, args.seed)
    val_samples = mh.maybe_limit(val_samples, args.limit_eval_samples, args.seed + 1)
    test_samples = mh.maybe_limit(test_samples, args.limit_eval_samples, args.seed + 2)
    distance_scale = max(s.distance_m for s in train_samples)
    normalization = {"state_distance_scale": float(max(distance_scale, 1.0))}
    summary = {
        "num_frames": len(records),
        "split_ranges": {k: list(v) for k, v in ranges.items()},
        "offsets": offsets,
        "samples": {"train": len(train_samples), "val": len(val_samples), "test": len(test_samples)},
        "horizon_counts": {
            "train": mh.horizon_counts(train_samples),
            "val": mh.horizon_counts(val_samples),
            "test": mh.horizon_counts(test_samples),
        },
        "sample_stats": {
            "train": mh.sample_stats(train_samples),
            "val": mh.sample_stats(val_samples),
            "test": mh.sample_stats(test_samples),
        },
        "horizon_stats": {
            "train": mh.horizon_sample_stats(train_samples),
            "val": mh.horizon_sample_stats(val_samples),
            "test": mh.horizon_sample_stats(test_samples),
        },
        "normalization": normalization,
        "horizon_names": mh.HORIZONS,
        "assumption": "Inputs include current visual sequence, target image, horizon token, and current heading prior. Coordinates are labels only.",
    }
    save_json(run_dir / "metrics" / "dataset_summary.json", summary)
    save_json(run_dir / "metrics" / "normalization_stats.json", normalization)
    print(f"Frames: {len(records)}")
    print(f"Split ranges: {ranges}")
    print(f"Samples: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")
    print(f"Train horizon counts: {mh.horizon_counts(train_samples)}")
    print(f"Val horizon counts:   {mh.horizon_counts(val_samples)}")
    print(f"Test horizon counts:  {mh.horizon_counts(test_samples)}")

    if args.dry_run:
        print("Dry run complete. No model was trained.")
        return
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available in this Python environment. Use --dry_run 1 for metadata checks.")

    train_ds = HeadingNavigationDataset(records, train_samples, distance_scale, args.seq_len, args.image_size)
    val_ds = HeadingNavigationDataset(records, val_samples, distance_scale, args.seq_len, args.image_size)
    test_ds = HeadingNavigationDataset(records, test_samples, distance_scale, args.seq_len, args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContinuousHeadingModel(
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
    reached_pos_weight = torch.tensor(pos_weight_from_samples(train_samples), dtype=torch.float32, device=device)

    print(f"Device: {device}")
    print(f"Reached positive weight: {float(reached_pos_weight.detach().cpu()):.4f}")
    print("Current heading prior is a model input; coordinates are labels only.")
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
            reached_pos_weight,
            args,
            epoch,
            args.epochs,
            live_progress_path,
        )
        val_logs = evaluate_loss(model, val_loader, device, reached_pos_weight, args)
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
            f"heading={val_logs['heading']:.4f} dist={val_logs['distance']:.4f}",
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
    metrics, rows, arrays = evaluate_full(model, test_loader, device, distance_scale, args.reached_threshold)
    test_logs = evaluate_loss(model, test_loader, device, reached_pos_weight, args)
    metrics["best_val_loss"] = float(best_val)
    metrics["test_loss"] = test_logs["loss"]
    save_metrics_csv(run_dir / "metrics" / "final_navigation_metrics.csv", metrics)
    save_predictions_csv(run_dir / "predictions" / "test_per_sample_predictions.csv", rows)
    save_plots(run_dir, history, arrays, metrics)
    write_live_progress(live_progress_path, "Finished final evaluation.")
    print(f"Saved metrics to {run_dir / 'metrics' / 'final_navigation_metrics.csv'}")
    print(f"Saved plots to {run_dir / 'plots'}")
    print(f"Best val loss: {best_val:.4f} | Test loss: {metrics['test_loss']:.4f}")
    print(
        f"Heading median error: {metrics['heading_error_median_deg']:.2f} deg | "
        f"Distance MAE: {metrics['distance_mae_m']:.2f} m"
    )


if __name__ == "__main__":
    main()
