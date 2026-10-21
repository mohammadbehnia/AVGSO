# -*- coding: utf-8 -*-
"""
Computational cost profiler for the Chapter 5 Spiral VGG/V3 localization model.

The script profiles the saved final localization checkpoint using synthetic
inputs and one real four-frame window. It reports parameters, checkpoint size,
latency, throughput, peak CUDA memory, and PyTorch-profiler FLOP estimates.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def import_localization_module(repo_root: Path):
    path = repo_root / "gso_multiframe_spiral_v3_backbones.py"
    spec = importlib.util.spec_from_file_location("ch5loc", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import localization script from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        run_name = time.strftime("ch5_localization_profile_%Y%m%d_%H%M%S")
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def bytes_to_mb(value: float) -> float:
    return float(value) / (1024.0 * 1024.0)


def count_parameters(model) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"parameters_total": int(total), "parameters_trainable": int(trainable)}


def make_synthetic_inputs(torch, cfg, batch_size: int, device):
    return torch.randn(batch_size, int(cfg.T_frames), 3, int(cfg.image_size), int(cfg.image_size), device=device)


def make_real_inputs(torch, loc, cfg, start_idx: int, batch_size: int, device):
    frames = loc.list_frames_one_folder(cfg.data_root)
    ds = loc.GSOSlidingWindowDataset(frames, cfg, start_idx, start_idx + 1, xy_mean=None, xy_std=None)
    item = ds[0]
    images = item["images"].unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).to(device)
    return images


def forward_once(torch, model, images, amp: bool):
    with torch.inference_mode():
        with torch.amp.autocast("cuda", enabled=(amp and images.is_cuda)):
            return model(images)


def measure_forward_latency(torch, model, images, warmup: int, repeats: int, amp: bool) -> Dict[str, float]:
    device = images.device
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        forward_once(torch, model, images, amp)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    times_ms: List[float] = []
    for _ in range(repeats):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            forward_once(torch, model, images, amp)
            end.record()
            torch.cuda.synchronize(device)
            times_ms.append(float(start.elapsed_time(end)))
        else:
            t0 = time.perf_counter()
            forward_once(torch, model, images, amp)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times_ms, dtype=np.float64)
    batch_size = int(images.shape[0])
    peak_mb = bytes_to_mb(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else float("nan")
    return {
        "batch_size": float(batch_size),
        "latency_mean_ms": float(np.mean(arr)),
        "latency_median_ms": float(np.median(arr)),
        "latency_p90_ms": float(np.quantile(arr, 0.90)),
        "latency_std_ms": float(np.std(arr)),
        "throughput_samples_per_s": float(1000.0 * batch_size / max(np.mean(arr), 1e-9)),
        "peak_cuda_memory_mb": peak_mb,
    }


def measure_real_e2e_latency(torch, loc, cfg, model, start_idx: int, repeats: int, amp: bool, device):
    times_ms: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        images = make_real_inputs(torch, loc, cfg, start_idx, 1, device)
        forward_once(torch, model, images, amp)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "real_image_end_to_end_latency_mean_ms": float(np.mean(arr)),
        "real_image_end_to_end_latency_median_ms": float(np.median(arr)),
        "real_image_end_to_end_latency_p90_ms": float(np.quantile(arr, 0.90)),
    }


def estimate_forward_flops(torch, model, images, amp: bool) -> Dict[str, float]:
    if images.device.type != "cuda":
        return {"forward_flops_profiler": float("nan"), "forward_gflops_profiler": float("nan")}
    try:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities, with_flops=True, record_shapes=False) as prof:
            forward_once(torch, model, images, amp)
        flops = 0
        for evt in prof.key_averages():
            value = getattr(evt, "flops", 0)
            if value:
                flops += int(value)
        return {"forward_flops_profiler": float(flops), "forward_gflops_profiler": float(flops) / 1e9}
    except Exception as exc:
        return {"forward_flops_profiler": float("nan"), "forward_gflops_profiler": float("nan"), "flops_profiler_error": repr(exc)}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile Chapter 5 localization computational cost.")
    parser.add_argument("--checkpoint", type=str, default=r"C:\Users\admin\Desktop\spiral_avgso\localization_spiral_vgg\v3_results\checkpoints_v3\best.pt")
    parser.add_argument("--output_root", type=str, default=r"C:\Users\admin\Desktop\spiral_avgso\visual_navigation\results\computational_profile")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--batch_sizes", type=str, default="1,4")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--real_repeats", type=int, default=8)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    loc = import_localization_module(repo_root)
    torch = loc.torch
    torch.manual_seed(args.seed)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg_dict = checkpoint.get("cfg", {})
    cfg = loc.CFG(data_root=cfg_dict["data_root"])
    for key, value in cfg_dict.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    if not hasattr(cfg, "backbone"):
        cfg.backbone = "vgg"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = loc.SpatioTemporalTransformer(cfg, embed_dim=256, nhead=8, layers=4, dropout=0.1, backbone=getattr(cfg, "backbone", "vgg")).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    run_dir = make_run_dir(args.output_root, args.run_name)
    num_windows = len(loc.list_frames_one_folder(cfg.data_root)) - (int(cfg.T_frames) - 1) * int(cfg.stride)
    tr, ev, te = loc.chronological_split(num_windows, float(cfg.train_frac), float(cfg.eval_frac), float(cfg.test_frac))
    start_idx = te[0] + 8

    rows: List[Dict[str, object]] = []
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    for batch_size in batch_sizes:
        images = make_synthetic_inputs(torch, cfg, batch_size, device)
        latency = measure_forward_latency(torch, model, images, args.warmup, args.repeats, bool(args.amp))
        flops = estimate_forward_flops(torch, model, images, bool(args.amp)) if batch_size == 1 else {}
        rows.append({"mode": "synthetic_model_forward", **latency, **flops})

    real_images = make_real_inputs(torch, loc, cfg, start_idx, 1, device)
    rows.append({"mode": "real_preloaded_forward", **measure_forward_latency(torch, model, real_images, max(2, args.warmup // 2), args.repeats, bool(args.amp))})
    rows.append({"mode": "real_image_load_plus_forward", "batch_size": 1.0, **measure_real_e2e_latency(torch, loc, cfg, model, start_idx, args.real_repeats, bool(args.amp), device)})

    input_info = {
        "seq_len": int(cfg.T_frames),
        "image_size": int(cfg.image_size),
        "images_per_sample": int(cfg.T_frames),
        "input_pixels_per_sample": int(int(cfg.T_frames) * 3 * int(cfg.image_size) * int(cfg.image_size)),
        "num_windows": int(num_windows),
        "split_windows": {"train": int(tr[1] - tr[0]), "eval": int(ev[1] - ev[0]), "test": int(te[1] - te[0])},
    }
    summary: Dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_size_mb": bytes_to_mb(checkpoint_path.stat().st_size),
        **count_parameters(model),
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else [],
        "gpu_total_memory_mb": bytes_to_mb(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else float("nan"),
        "torch_num_threads": int(torch.get_num_threads()),
        **input_info,
        "amp_enabled": bool(args.amp),
        "latency_rows": rows,
    }
    save_json(run_dir / "ch5_localization_profile_summary.json", summary)
    save_csv(run_dir / "ch5_localization_profile_latency.csv", rows)
    print(json.dumps(summary, indent=2))
    print(f"Saved profile to {run_dir}")


if __name__ == "__main__":
    main()
