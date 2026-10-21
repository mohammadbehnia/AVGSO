# AVGSO: Visual Geo-Localization and Navigation for UAVs

This repository contains the research code for the **AVGSO** project: learning visual geo-localization and goal-directed navigation from sequential oblique aerial imagery.

The project uses geo-referenced simulated UAV views to study two connected problems:

1. **Multi-frame localization:** estimate the UAV position and heading from a short sequence of images.
2. **Visual navigation:** use the current visual sequence, target view, and route/horizon information to predict how the UAV should move toward the target.

The code is organized around the final AVGSO research workflow rather than a single end-to-end black-box policy:

```text
AVGSO frames + geographic labels
              |
              v
Multi-frame visual localization
              |
              v
Route-conditioned navigation baselines
              |
              v
Continuous heading-distance navigation
              |
              v
Closed-loop navigation over the AVGSO frame graph
```

## Main components

| Component | Purpose | Main source |
|---|---|---|
| AVGSO frame generation | Create spiral, lawnmower, circular, line, and oblique routes | `map_automator_no_text_spiral.py`, `map_automator_straight_line.py` |
| Multi-frame localization | Predict metric position and heading from four-frame windows | `gso_multiframe_spiral_v3_backbones.py` |
| Localization analysis | Produce meter-level plots and trajectory/generalization checks | `gso_multiframe_spiral_v3_meters_graphs.py`, `test_line_trajectory_generalization.py` |
| Immediate goal navigation | Predict target-relative direction, distance, and navigation command | `visual_navigation/stage1_goal_nav_immediate.py` |
| Route-conditioned navigation | Use a start-goal route prior with hidden intermediate state labels | `visual_navigation/stage1_route_conditioned_nav.py` |
| Multi-horizon navigation | Predict navigation state at several future horizons | `visual_navigation/stage2_multihorizon_visual_nav.py` |
| Continuous navigation | Predict heading correction, target distance, and reached probability | `visual_navigation/stage2b_continuous_heading_nav.py` |
| Closed-loop evaluation | Repeatedly apply the navigation model on the AVGSO frame graph | `visual_navigation/stage2c_closed_loop_nav.py` |

## Research summary

The localization models process four consecutive views with a convolutional visual backbone and temporal/spatial aggregation. Position is evaluated in local metric coordinates, while heading is represented with circular targets such as sine and cosine components.

The navigation experiments progress from immediate target-relative commands to route-conditioned and multi-horizon prediction. The final navigation formulation uses continuous heading correction and target-distance prediction instead of treating discrete commands as the primary output. Stage 2C then evaluates the learned Stage 2B primitive iteratively in a dataset-grounded closed-loop simulator.

## Verified results

The metrics below are from saved experiment artifacts in the working project. They describe the current research baselines and are not a claim that the models are ready for real-flight deployment.

| Experiment | Result |
|---|---|
| Stage 1C route-conditioned navigation | 90.6% command accuracy; 86.3% macro F1; 90.0% reached F1 on 2,000 test samples |
| Stage 2B continuous heading navigation | 2.31° mean heading error; 24.8 m distance MAE on 4,000 test samples |
| Stage 2C closed-loop navigation | 89.5% success over 200 episodes with a 45 m arrival threshold |
| Stage 2C short/medium horizons | 100.0% success for offsets +4 and +10 |
| Stage 2C long horizon | 72.0% success for offset +40 |

The long-horizon result remains the principal limitation. The simulator follows feasible visual observations in the AVGSO frame graph; it is not a full aircraft dynamics, obstacle-avoidance, or onboard flight-control simulator.

## Installation

Create a Python environment with PyTorch and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

For GPU training, install the PyTorch build appropriate for your CUDA version before installing the remaining packages. The scripts support CPU-side dry runs without PyTorch, but model training and checkpoint evaluation require PyTorch and torchvision.

## Dataset format

The primary scripts expect a flat directory of images with geographic metadata encoded in each filename:

```text
frame_<frame_id>_<latitude>_<longitude>_<heading>.jpg
```

Example:

```text
frame_000001_36.045944_-79.960988_90.jpg
```

The repository currently contains large AVGSO image folders and archives for local experimentation. These raw assets are intentionally excluded by `.gitignore`; publish the code and documentation separately from the full imagery, or provide a dataset download location and checksum in your project release.

## Reproduce the main workflows

### 1. Multi-frame localization

Run the VGG-based localization baseline on a folder that follows the filename convention above:

```bash
python gso_multiframe_spiral_v3_backbones.py \
  --data_root /path/to/avgso_frames \
  --epochs 20 \
  --batch_size 4 \
  --backbone vgg \
  --output_dir results/localization
```

### 2. Immediate goal-conditioned navigation dry run

Build the 15--80 m immediate-distance samples and verify the split/output structure before training:

```bash
python visual_navigation/stage1_goal_nav_immediate.py \
  --data_root /path/to/avgso_frames \
  --output_root visual_navigation/results/stage1_immediate \
  --run_name dry_run \
  --dry_run 1
```

### 3. Continuous heading-distance navigation dry run

```bash
python visual_navigation/stage2b_continuous_heading_nav.py \
  --data_root /path/to/avgso_frames \
  --output_root visual_navigation/results/stage2_heading \
  --run_name dry_run_stage2b \
  --dry_run 1
```

### 4. Closed-loop evaluation

After training Stage 2B, pass its run directory to the Stage 2C simulator:

```bash
python visual_navigation/stage2c_closed_loop_nav.py \
  --stage2b_run_dir visual_navigation/results/stage2_heading/<run_name> \
  --output_root visual_navigation/results/stage2_closed_loop \
  --run_name closed_loop_check \
  --max_episodes 50 \
  --dry_run 1
```

Remove `--dry_run 1` for checkpoint-backed closed-loop evaluation.

## Outputs

Training and evaluation runs save structured artifacts under the selected output directory:

- `config.json`: run configuration and data settings
- `metrics/`: final metrics, normalization statistics, and loss logs
- `predictions/`: per-sample predictions for detailed analysis
- `plots/`: error, trajectory, confusion, and horizon plots
- `checkpoints/`: best model checkpoints
- `trajectories/`: closed-loop episode and step traces

## Important limitations

- The AVGSO imagery is simulated or map-derived aerial imagery, not a replacement for field data.
- Geographic coordinates are used to construct labels and evaluation geometry; deployment requires a real localization and sensing strategy.
- The closed-loop simulator operates over available dataset frames and does not model aircraft dynamics, wind, latency, collision risk, or actuator limits.
- Route and command distributions are imbalanced. In particular, rare left-turn cases remain difficult in some route-conditioned experiments.
- Checkpoint, dataset, dissertation, and presentation files are not required to understand or run the core source code and should be published separately when repository size or licensing requires it.

## Citation

If you use this code or the AVGSO benchmark in academic work, cite the associated dissertation or project publication when it becomes available. Until then, please retain the project name **AVGSO: Visual Geo-Localization and Navigation for UAVs** and link back to this repository.

## License

No license is asserted by this repository yet. Add a license file before accepting external contributions or redistributing the dataset and generated imagery.
