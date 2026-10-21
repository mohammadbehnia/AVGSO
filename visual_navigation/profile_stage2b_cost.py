# -*- coding: utf-8 -*-
"""
Computational cost profiler for Stage 2B visual navigation.

Measures:
  - parameter counts and checkpoint size
  - synthetic model-only forward latency
  - real image-load + forward latency
  - peak CUDA memory
  - approximate forward FLOPs from torch.profiler where supported

Run from the repository root in the Anaconda/PyTorch environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import stage2_multihorizon_visual_nav as mh
import stage2b_continuous_heading_nav as s2b

if not s2b.TORCH_AVAILABLE:
    raise RuntimeError("PyTorch is required. Run this script in the Anaconda gso environment.")

torch = s2b.torch
T = s2b.T
Image = s2b.Image


def load_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_run_dir(output_root: str, run_name: str) -> Path:
    root = Path(output_root)
    if not run_name:
        run_name = time.strftime("stage2b_profile_%Y%m%d_%H%M%S")
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def bytes_to_mb(value: float) -> float:
    return float(value) / (1024.0 * 1024.0)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"parameters_total": int(total), "parameters_trainable": int(trainable)}


def load_model(config: Dict[str, object], checkpoint_path: Path, device: torch.device):
    model = s2b.ContinuousHeadingModel(
        seq_len=int(config["seq_len"]),
        d_model=int(config["d_model"]),
        nhead=int(config["nhead"]),
        layers=int(config["layers"]),
        dropout=float(config["dropout"]),
        backbone=str(config["backbone"]),
        use_pretrained=bool(int(config.get("use_pretrained", 1))),
        freeze_backbone=bool(int(config.get("freeze_backbone", 1))),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def make_synthetic_inputs(config: Dict[str, object], batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
    seq_len = int(config["seq_len"])
    image_size = int(config["image_size"])
    return {
        "current_images": torch.randn(batch_size, seq_len, 3, image_size, image_size, device=device),
        "target_image": torch.randn(batch_size, 3, image_size, image_size, device=device),
        "horizon_id": torch.full((batch_size,), 3, dtype=torch.long, device=device),
        "heading_prior": torch.tensor([[0.0, 1.0]], dtype=torch.float32, device=device).repeat(batch_size, 1),
    }


def transform_for_image_size(image_size: int):
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_norm(path: str, transform):
    with Image.open(path) as img:
        return transform(img.convert("RGB"))


def make_real_inputs(
    records: Sequence[mh.FrameRecord],
    config: Dict[str, object],
    current_idx: int,
    target_offset: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    seq_len = int(config["seq_len"])
    transform = transform_for_image_size(int(config["image_size"]))
    target_idx = current_idx + target_offset
    current_images = torch.stack([load_norm(records[i].path, transform) for i in range(current_idx - seq_len + 1, current_idx + 1)])
    target_image = load_norm(records[target_idx].path, transform)
    heading_prior = torch.tensor(s2b.heading_prior(records[current_idx]), dtype=torch.float32)
    offsets = [int(x.strip()) for x in str(config["offsets"]).split(",") if x.strip()]
    horizon_id = int(np.argmin(np.abs(np.asarray(offsets, dtype=np.float32) - float(target_offset))))
    return {
        "current_images": current_images.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).to(device),
        "target_image": target_image.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device),
        "horizon_id": torch.full((batch_size,), horizon_id, dtype=torch.long, device=device),
        "heading_prior": heading_prior.unsqueeze(0).repeat(batch_size, 1).to(device),
    }


def forward_once(model, inputs, amp: bool):
    with torch.inference_mode():
        with torch.amp.autocast("cuda", enabled=(amp and inputs["current_images"].is_cuda)):
            return model(inputs["current_images"], inputs["target_image"], inputs["horizon_id"], inputs["heading_prior"])


def measure_forward_latency(model, inputs, warmup: int, repeats: int, amp: bool) -> Dict[str, float]:
    device = inputs["current_images"].device
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        forward_once(model, inputs, amp)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    times_ms: List[float] = []
    for _ in range(repeats):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            forward_once(model, inputs, amp)
            end.record()
            torch.cuda.synchronize(device)
            times_ms.append(float(start.elapsed_time(end)))
        else:
            t0 = time.perf_counter()
            forward_once(model, inputs, amp)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    peak_mb = bytes_to_mb(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else float("nan")
    arr = np.asarray(times_ms, dtype=np.float64)
    batch_size = int(inputs["current_images"].shape[0])
    return {
        "batch_size": float(batch_size),
        "latency_mean_ms": float(np.mean(arr)),
        "latency_median_ms": float(np.median(arr)),
        "latency_p90_ms": float(np.quantile(arr, 0.90)),
        "latency_std_ms": float(np.std(arr)),
        "throughput_samples_per_s": float(1000.0 * batch_size / max(np.mean(arr), 1e-9)),
        "peak_cuda_memory_mb": peak_mb,
    }


def measure_end_to_end_real_latency(model, records, config, current_idx: int, target_offset: int, repeats: int, amp: bool, device):
    times_ms: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        inputs = make_real_inputs(records, config, current_idx, target_offset, 1, device)
        forward_once(model, inputs, amp)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "real_image_end_to_end_latency_mean_ms": float(np.mean(arr)),
        "real_image_end_to_end_latency_median_ms": float(np.median(arr)),
        "real_image_end_to_end_latency_p90_ms": float(np.quantile(arr, 0.90)),
    }


def estimate_forward_flops(model, inputs, amp: bool) -> Dict[str, float]:
    if inputs["current_images"].device.type != "cuda":
        return {"forward_flops_profiler": float("nan"), "forward_gflops_profiler": float("nan")}
    try:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities, with_flops=True, record_shapes=False) as prof:
            forward_once(model, inputs, amp)
        flops = 0
        for evt in prof.key_averages():
            value = getattr(evt, "flops", 0)
            if value:
                flops += int(value)
        return {"forward_flops_profiler": float(flops), "forward_gflops_profiler": float(flops) / 1e9}
    except Exception as exc:
        return {
            "forward_flops_profiler": float("nan"),
            "forward_gflops_profiler": float("nan"),
            "flops_profiler_error": repr(exc),
        }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile Stage 2B computational cost.")
    parser.add_argument("--stage2b_run_dir", type=str, default=r"C:\Users\admin\Desktop\spiral_avgso\visual_navigation\results\stage2_heading\stage2b_heading_40k_plot_upgrade")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--output_root", type=str, default=r"C:\Users\admin\Desktop\spiral_avgso\visual_navigation\results\computational_profile")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--batch_sizes", type=str, default="1,4")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--real_repeats", type=int, default=8)
    parser.add_argument("--target_offset", type=int, default=40)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    run_dir = make_run_dir(args.output_root, args.run_name)
    stage2b_run_dir = Path(args.stage2b_run_dir)
    config = load_json(stage2b_run_dir / "config.json")
    normalization = load_json(stage2b_run_dir / "metrics" / "normalization_stats.json")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else stage2b_run_dir / "checkpoints" / "best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(config, checkpoint_path, device)

    records = mh.list_frames_one_folder(str(config["data_root"]))
    ranges = mh.split_ranges(len(records), float(config["train_frac"]), float(config["val_frac"]), int(config["split_gap"]))
    test_start, test_end = ranges["test"]
    current_idx = test_start + int(config["seq_len"]) + 8
    if current_idx + int(args.target_offset) >= test_end:
        current_idx = test_start + int(config["seq_len"])

    param_counts = count_parameters(model)
    checkpoint_size_mb = bytes_to_mb(checkpoint_path.stat().st_size)
    device_info = {
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else [],
        "gpu_total_memory_mb": bytes_to_mb(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else float("nan"),
        "torch_num_threads": int(torch.get_num_threads()),
    }
    input_info = {
        "seq_len": int(config["seq_len"]),
        "image_size": int(config["image_size"]),
        "images_per_sample": int(config["seq_len"]) + 1,
        "input_pixels_per_sample": int((int(config["seq_len"]) + 1) * 3 * int(config["image_size"]) * int(config["image_size"])),
        "state_distance_scale": float(normalization["state_distance_scale"]),
    }

    rows: List[Dict[str, object]] = []
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    for batch_size in batch_sizes:
        inputs = make_synthetic_inputs(config, batch_size, device)
        latency = measure_forward_latency(model, inputs, args.warmup, args.repeats, bool(args.amp))
        flops = estimate_forward_flops(model, inputs, bool(args.amp)) if batch_size == 1 else {}
        row = {"mode": "synthetic_model_forward", **latency, **flops}
        rows.append(row)

    real_inputs = make_real_inputs(records, config, current_idx, int(args.target_offset), 1, device)
    real_forward = measure_forward_latency(model, real_inputs, max(2, args.warmup // 2), args.repeats, bool(args.amp))
    real_e2e = measure_end_to_end_real_latency(
        model, records, config, current_idx, int(args.target_offset), args.real_repeats, bool(args.amp), device
    )
    rows.append({"mode": "real_preloaded_forward", **real_forward})
    rows.append({"mode": "real_image_load_plus_forward", "batch_size": 1.0, **real_e2e})

    summary: Dict[str, object] = {
        "stage2b_run_dir": str(stage2b_run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "checkpoint_size_mb": checkpoint_size_mb,
        **param_counts,
        **device_info,
        **input_info,
        "amp_enabled": bool(args.amp),
        "latency_rows": rows,
    }
    save_json(run_dir / "stage2b_profile_summary.json", summary)
    save_csv(run_dir / "stage2b_profile_latency.csv", rows)
    print(json.dumps(summary, indent=2))
    print(f"Saved profile to {run_dir}")


if __name__ == "__main__":
    main()
