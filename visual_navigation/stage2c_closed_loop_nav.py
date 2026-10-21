# -*- coding: utf-8 -*-
"""
Stage 2C closed-loop visual navigation simulation.

This script evaluates whether a trained Stage 2B continuous-heading model can
be used iteratively as a navigation policy. The target image remains fixed for
an episode. At each step, the model predicts a heading correction and target
distance from the current visual sequence, target image, horizon token, and
current heading prior. The simulator converts that prediction into a desired
local waypoint and moves to the nearest feasible visual frame in the AVGSO
frame graph.

This is not a flight dynamics simulator. It is a dataset-grounded closed-loop
visual navigation test over available geo-referenced visual observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import stage2_multihorizon_visual_nav as mh
import stage2b_continuous_heading_nav as s2b

TORCH_AVAILABLE = s2b.TORCH_AVAILABLE
torch = s2b.torch
T = s2b.T
Image = s2b.Image
plt = s2b.plt


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_run_dir(output_root: str, run_name: str) -> Path:
    root = Path(output_root)
    if not run_name:
        run_name = "stage2c_closed_loop_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / run_name
    for sub in ["metrics", "plots", "trajectories"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def load_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def horizon_id_from_offset(offset: int, offsets: Sequence[int]) -> int:
    arr = np.asarray(offsets, dtype=np.float32)
    return int(np.argmin(np.abs(arr - float(offset))))


def horizon_id_from_distance(distance_m: float, offsets: Sequence[int], approx_m_per_frame: float) -> int:
    expected = np.asarray(offsets, dtype=np.float32) * float(approx_m_per_frame)
    return int(np.argmin(np.abs(expected - float(distance_m))))


def local_distance(a: mh.FrameRecord, b: mh.FrameRecord) -> float:
    return math.hypot(b.x - a.x, b.y - a.y)


def build_episodes(
    records: Sequence[mh.FrameRecord],
    split_range: Tuple[int, int],
    seq_len: int,
    target_offsets: Sequence[int],
    max_episodes: int,
    seed: int,
    approx_m_per_frame: float,
    max_offset_distance_factor: float,
    max_offset_distance_extra_m: float,
) -> List[Dict[str, int]]:
    start, end = split_range
    candidates: List[Dict[str, int]] = []
    max_offset = max(target_offsets)
    first_current = start + seq_len - 1
    last_current = end - max_offset - 1
    for current_idx in range(first_current, max(first_current, last_current + 1)):
        for offset in target_offsets:
            target_idx = current_idx + offset
            if target_idx >= end:
                continue
            distance = local_distance(records[current_idx], records[target_idx])
            expected_distance = approx_m_per_frame * offset
            max_allowed = max(expected_distance * max_offset_distance_factor, expected_distance + max_offset_distance_extra_m)
            if distance <= max_allowed:
                candidates.append({"start_idx": current_idx, "target_idx": target_idx, "target_offset": offset})
    rng = random.Random(seed)
    rng.shuffle(candidates)
    if max_episodes > 0:
        candidates = candidates[:max_episodes]
    return candidates


def transform_for_image_size(image_size: int):
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_image_tensor(path: str, transform):
    with Image.open(path) as img:
        return transform(img.convert("RGB"))


def choose_next_frame(
    records: Sequence[mh.FrameRecord],
    current_idx: int,
    target_idx: int,
    desired_x: float,
    desired_y: float,
    split_range: Tuple[int, int],
    visited: Dict[int, int],
    candidate_radius_m: float,
    success_threshold_m: float,
) -> int:
    current = records[current_idx]
    target = records[target_idx]
    if local_distance(current, target) <= success_threshold_m:
        return target_idx

    start, end = split_range
    best_idx = current_idx
    best_score = float("inf")
    fallback_idx = current_idx
    fallback_score = float("inf")

    for idx in range(start, end):
        if idx == current_idx:
            continue
        rec = records[idx]
        move_dist = local_distance(current, rec)
        if move_dist > candidate_radius_m:
            continue
        waypoint_error = math.hypot(rec.x - desired_x, rec.y - desired_y)
        target_error = local_distance(rec, target)
        repeat_penalty = 35.0 * visited.get(idx, 0)
        score = waypoint_error + 0.15 * target_error + repeat_penalty
        if score < best_score:
            best_idx = idx
            best_score = score
        if target_error < fallback_score:
            fallback_idx = idx
            fallback_score = target_error

    if best_idx == current_idx:
        return fallback_idx
    return best_idx


def tensor_batch_for_step(records, current_idx: int, target_idx: int, seq_len: int, transform, device):
    seq_indices = range(current_idx - seq_len + 1, current_idx + 1)
    current_images = torch.stack([load_image_tensor(records[i].path, transform) for i in seq_indices], dim=0)
    target_image = load_image_tensor(records[target_idx].path, transform)
    heading_prior = torch.tensor(s2b.heading_prior(records[current_idx]), dtype=torch.float32)
    return {
        "current_images": current_images.unsqueeze(0).to(device),
        "target_image": target_image.unsqueeze(0).to(device),
        "heading_prior": heading_prior.unsqueeze(0).to(device),
    }


def load_stage2b_model(checkpoint_path: Path, config: Dict[str, object], device):
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
    return model


@torch.no_grad() if TORCH_AVAILABLE else (lambda fn: fn)
def simulate_episode(
    model,
    records: Sequence[mh.FrameRecord],
    episode: Dict[str, int],
    split_range: Tuple[int, int],
    config: Dict[str, object],
    distance_scale: float,
    args,
    transform,
    device,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    offsets = [int(x.strip()) for x in str(config["offsets"]).split(",") if x.strip()]
    current_idx = int(episode["start_idx"])
    target_idx = int(episode["target_idx"])
    target_offset = int(episode["target_offset"])
    initial_distance = local_distance(records[current_idx], records[target_idx])
    horizon_id = horizon_id_from_offset(target_offset, offsets)
    visited: Dict[int, int] = {}
    step_rows: List[Dict[str, object]] = []
    path_length = 0.0
    success = False
    stop_reason = "max_steps"

    for step in range(int(args.max_steps) + 1):
        current = records[current_idx]
        target = records[target_idx]
        true_distance = local_distance(current, target)
        if true_distance <= args.success_threshold_m or current_idx == target_idx:
            success = True
            stop_reason = "reached"
            break

        batch = tensor_batch_for_step(records, current_idx, target_idx, int(config["seq_len"]), transform, device)
        horizon_tensor = torch.tensor([horizon_id], dtype=torch.long, device=device)
        out = model(batch["current_images"], batch["target_image"], horizon_tensor, batch["heading_prior"])
        pred_heading = s2b.tensor_to_numpy(out["heading"])[0:1]
        pred_distance = float(s2b.tensor_to_numpy(out["distance"])[0] * distance_scale)
        heading_prior = s2b.tensor_to_numpy(batch["heading_prior"])[0:1]
        pred_dx, pred_dy = s2b.vector_from_heading_distance(heading_prior, pred_heading, np.asarray([pred_distance]))
        step_length = min(max(pred_distance, args.min_step_m), args.max_step_m)
        vector_norm = max(math.hypot(float(pred_dx[0]), float(pred_dy[0])), 1e-6)
        desired_x = current.x + step_length * float(pred_dx[0]) / vector_norm
        desired_y = current.y + step_length * float(pred_dy[0]) / vector_norm
        next_idx = choose_next_frame(
            records,
            current_idx,
            target_idx,
            desired_x,
            desired_y,
            split_range,
            visited,
            args.candidate_radius_m,
            args.success_threshold_m,
        )
        move_distance = local_distance(records[current_idx], records[next_idx])
        path_length += move_distance
        visited[current_idx] = visited.get(current_idx, 0) + 1
        step_rows.append(
            {
                "episode_id": int(episode["episode_id"]),
                "step": step,
                "current_frame_id": int(current.frame_id),
                "next_frame_id": int(records[next_idx].frame_id),
                "target_frame_id": int(target.frame_id),
                "horizon": mh.HORIZONS[horizon_id],
                "true_distance_m": true_distance,
                "pred_distance_m": pred_distance,
                "move_distance_m": move_distance,
                "desired_x_m": desired_x,
                "desired_y_m": desired_y,
                "current_x_m": current.x,
                "current_y_m": current.y,
                "next_x_m": records[next_idx].x,
                "next_y_m": records[next_idx].y,
                "target_x_m": target.x,
                "target_y_m": target.y,
            }
        )
        if next_idx == current_idx:
            stop_reason = "stuck"
            break
        current_idx = next_idx
        horizon_id = horizon_id_from_distance(pred_distance, offsets, float(config["approx_m_per_frame"]))

    final_distance = local_distance(records[current_idx], records[target_idx])
    direct_distance = max(initial_distance, 1e-6)
    summary = {
        "episode_id": int(episode["episode_id"]),
        "start_frame_id": int(records[int(episode["start_idx"])].frame_id),
        "target_frame_id": int(records[target_idx].frame_id),
        "target_offset": target_offset,
        "success": int(success),
        "stop_reason": stop_reason,
        "steps": len(step_rows),
        "initial_distance_m": initial_distance,
        "final_distance_m": final_distance,
        "progress_m": initial_distance - final_distance,
        "progress_ratio": (initial_distance - final_distance) / direct_distance,
        "path_length_m": path_length,
        "path_efficiency": direct_distance / max(path_length, direct_distance),
    }
    return summary, step_rows


def compute_summary_metrics(episode_rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    if not episode_rows:
        return {}
    success = np.asarray([float(r["success"]) for r in episode_rows], dtype=np.float32)
    final_distance = np.asarray([float(r["final_distance_m"]) for r in episode_rows], dtype=np.float32)
    progress_ratio = np.asarray([float(r["progress_ratio"]) for r in episode_rows], dtype=np.float32)
    steps = np.asarray([float(r["steps"]) for r in episode_rows], dtype=np.float32)
    metrics = {
        "num_episodes": float(len(episode_rows)),
        "success_rate": float(np.mean(success)),
        "final_distance_mean_m": float(np.mean(final_distance)),
        "final_distance_median_m": float(np.median(final_distance)),
        "final_distance_p90_m": float(np.quantile(final_distance, 0.90)),
        "progress_ratio_mean": float(np.mean(progress_ratio)),
        "steps_mean": float(np.mean(steps)),
        "steps_median": float(np.median(steps)),
    }
    for offset in sorted({int(r["target_offset"]) for r in episode_rows}):
        mask = np.asarray([int(r["target_offset"]) == offset for r in episode_rows], dtype=bool)
        metrics[f"offset_{offset}_episodes"] = float(mask.sum())
        metrics[f"offset_{offset}_success_rate"] = float(np.mean(success[mask]))
        metrics[f"offset_{offset}_final_distance_median_m"] = float(np.median(final_distance[mask]))
    return metrics


def plot_final_distance(episode_rows: Sequence[Dict[str, object]], path: Path) -> None:
    if plt is None or not episode_rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [float(r["final_distance_m"]) for r in episode_rows]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    ax.hist(values, bins=25, color="#4c78a8", alpha=0.85, label="Closed-loop episodes")
    ax.set_xlabel("Final distance to target (m)")
    ax.set_ylabel("Episodes")
    ax.set_title("Closed-Loop Final Target Error")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_success_by_offset(episode_rows: Sequence[Dict[str, object]], path: Path) -> None:
    if plt is None or not episode_rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = sorted({int(r["target_offset"]) for r in episode_rows})
    rates = []
    for offset in offsets:
        subset = [r for r in episode_rows if int(r["target_offset"]) == offset]
        rates.append(np.mean([float(r["success"]) for r in subset]))
    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    ax.bar([f"+{o}" for o in offsets], rates, color="#54a24b", label="Success rate")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Success rate")
    ax.set_title("Closed-Loop Success by Target Offset")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_trajectory_examples(step_rows: Sequence[Dict[str, object]], path: Path, max_episodes: int = 8) -> None:
    if plt is None or not step_rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    episode_ids = sorted({int(r["episode_id"]) for r in step_rows})[:max_episodes]
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    for episode_id in episode_ids:
        rows = [r for r in step_rows if int(r["episode_id"]) == episode_id]
        xs = [float(rows[0]["current_x_m"])] + [float(r["next_x_m"]) for r in rows]
        ys = [float(rows[0]["current_y_m"])] + [float(r["next_y_m"]) for r in rows]
        tx = float(rows[-1]["target_x_m"])
        ty = float(rows[-1]["target_y_m"])
        ax.plot(xs, ys, linewidth=1.2, alpha=0.75, label="Navigated path" if episode_id == episode_ids[0] else None)
        ax.scatter([xs[0]], [ys[0]], s=12, color="#4c78a8", label="Start" if episode_id == episode_ids[0] else None)
        ax.scatter([tx], [ty], s=18, marker="x", color="#e45756", label="GT target" if episode_id == episode_ids[0] else None)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Closed-Loop Trajectory Examples")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_final_position_vs_target(step_rows: Sequence[Dict[str, object]], path: Path) -> None:
    if plt is None or not step_rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    final_by_episode: Dict[int, Dict[str, object]] = {}
    for row in step_rows:
        final_by_episode[int(row["episode_id"])] = row
    final_rows = [final_by_episode[k] for k in sorted(final_by_episode)]
    final_x = np.asarray([float(r["next_x_m"]) for r in final_rows], dtype=np.float32)
    final_y = np.asarray([float(r["next_y_m"]) for r in final_rows], dtype=np.float32)
    target_x = np.asarray([float(r["target_x_m"]) for r in final_rows], dtype=np.float32)
    target_y = np.asarray([float(r["target_y_m"]) for r in final_rows], dtype=np.float32)
    errors = np.sqrt((final_x - target_x) ** 2 + (final_y - target_y) ** 2)
    lim = max(
        float(np.nanmax(np.abs(final_x))),
        float(np.nanmax(np.abs(final_y))),
        float(np.nanmax(np.abs(target_x))),
        float(np.nanmax(np.abs(target_y))),
        1.0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=160)
    axes[0].scatter(target_x, final_x, c=errors, cmap="viridis", s=16, alpha=0.8, label="Closed-loop endpoints")
    axes[0].plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="Ideal target match")
    axes[0].set_xlabel("GT target x (m)")
    axes[0].set_ylabel("Final navigated x (m)")
    axes[0].set_title("Final X vs GT Target X")
    scatter = axes[1].scatter(target_y, final_y, c=errors, cmap="viridis", s=16, alpha=0.8, label="Closed-loop endpoints")
    axes[1].plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="Ideal target match")
    axes[1].set_xlabel("GT target y (m)")
    axes[1].set_ylabel("Final navigated y (m)")
    axes[1].set_title("Final Y vs GT Target Y")
    for ax in axes:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
    fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.8, label="Final error (m)")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Closed-loop simulation for Stage 2B continuous heading navigation.")
    parser.add_argument("--stage2b_run_dir", type=str, default=r"C:\Users\admin\Desktop\spiral_avgso\visual_navigation\results\stage2_heading\stage2b_heading_40k_plot_upgrade")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--output_root", type=str, default=r"C:\Users\admin\Desktop\spiral_avgso\visual_navigation\results\stage2_closed_loop")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--target_offsets", type=str, default="4,10,40")
    parser.add_argument("--max_episodes", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument("--success_threshold_m", type=float, default=45.0)
    parser.add_argument("--candidate_radius_m", type=float, default=95.0)
    parser.add_argument("--min_step_m", type=float, default=12.0)
    parser.add_argument("--max_step_m", type=float, default=70.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    stage2b_run_dir = Path(args.stage2b_run_dir)
    config = load_json(stage2b_run_dir / "config.json")
    normalization = load_json(stage2b_run_dir / "metrics" / "normalization_stats.json")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else stage2b_run_dir / "checkpoints" / "best.pt"
    run_dir = make_run_dir(args.output_root, args.run_name)
    target_offsets = [int(x.strip()) for x in args.target_offsets.split(",") if x.strip()]

    records = mh.list_frames_one_folder(str(config["data_root"]))
    ranges = mh.split_ranges(len(records), float(config["train_frac"]), float(config["val_frac"]), int(config["split_gap"]))
    test_range = ranges["test"]
    episodes = build_episodes(
        records,
        test_range,
        int(config["seq_len"]),
        target_offsets,
        int(args.max_episodes),
        int(args.seed),
        float(config["approx_m_per_frame"]),
        float(config["max_offset_distance_factor"]),
        float(config["max_offset_distance_extra_m"]),
    )
    for idx, episode in enumerate(episodes):
        episode["episode_id"] = idx

    setup = {
        "stage2b_run_dir": str(stage2b_run_dir),
        "checkpoint": str(checkpoint_path),
        "test_range": test_range,
        "target_offsets": target_offsets,
        "num_episodes": len(episodes),
        "args": vars(args),
    }
    save_json(run_dir / "metrics" / "closed_loop_setup.json", setup)
    print(f"Run directory: {run_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes: {len(episodes)}")
    print(f"Target offsets: {target_offsets}")

    if args.dry_run:
        print("Dry run complete. No checkpoint simulation was executed.")
        return
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available in this Python environment. Run this in the Anaconda gso environment.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transform_for_image_size(int(config["image_size"]))
    model = load_stage2b_model(checkpoint_path, config, device)
    distance_scale = float(normalization["state_distance_scale"])
    print(f"Device: {device}")

    episode_rows: List[Dict[str, object]] = []
    all_step_rows: List[Dict[str, object]] = []
    for episode in episodes:
        summary, step_rows = simulate_episode(
            model,
            records,
            episode,
            test_range,
            config,
            distance_scale,
            args,
            transform,
            device,
        )
        episode_rows.append(summary)
        all_step_rows.extend(step_rows)
        if (int(summary["episode_id"]) + 1) % 25 == 0:
            print(f"Simulated {int(summary['episode_id']) + 1}/{len(episodes)} episodes", flush=True)

    metrics = compute_summary_metrics(episode_rows)
    save_json(run_dir / "metrics" / "closed_loop_metrics.json", metrics)
    save_csv(run_dir / "trajectories" / "closed_loop_episode_summary.csv", episode_rows)
    save_csv(run_dir / "trajectories" / "closed_loop_step_trace.csv", all_step_rows)
    plot_final_distance(episode_rows, run_dir / "plots" / "final_distance_histogram.png")
    plot_success_by_offset(episode_rows, run_dir / "plots" / "success_by_offset.png")
    plot_trajectory_examples(all_step_rows, run_dir / "plots" / "trajectory_examples.png")
    plot_final_position_vs_target(all_step_rows, run_dir / "plots" / "final_position_vs_gt_target.png")
    print(f"Saved metrics to {run_dir / 'metrics' / 'closed_loop_metrics.json'}")
    print(
        f"Success rate: {metrics.get('success_rate', float('nan')):.3f} | "
        f"Median final distance: {metrics.get('final_distance_median_m', float('nan')):.1f} m"
    )


if __name__ == "__main__":
    main()
