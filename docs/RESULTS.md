# AVGSO Results Summary

This page records the current saved baselines for the localization and navigation pipeline.

## Localization

The final localization narrative uses a VGG-based multi-frame model that aggregates four consecutive oblique aerial views and predicts metric position and heading. The implementation and analysis scripts are:

- `gso_multiframe_spiral_v3_backbones.py`
- `gso_multiframe_spiral_v3_meters_graphs.py`
- `test_line_trajectory_generalization.py`
- `test_line_trajectory_generalization_center_radius.py`

The localization experiments are intended to test both route interpolation and generalization across line and spiral geometries. Reported localization numbers should always identify the route split, frame spacing, coordinate conversion, and whether the result is measured on the same route family or on a held-out route geometry.

## Navigation

### Stage 1C: route-conditioned navigation

The Stage 1C checkpoint evaluation contains 2,000 test samples and reports 90.6% command accuracy, 86.3% macro F1, 65.8% balanced accuracy, and 90.0% reached F1. Performance is strongest for the common forward and stop classes. The left-turn class remains underrepresented and is not reliably recalled.

### Stage 2B: continuous heading-distance navigation

The Stage 2B evaluation contains 4,000 test samples. The model reports 2.31° mean heading error, 0.42° median heading error, and 24.8 m distance MAE. The +4 and +10 horizons are substantially easier than the +40 horizon, which has 85.6 m distance MAE and 4.21° mean heading error.

### Stage 2C: closed-loop evaluation

Stage 2C evaluates the Stage 2B model iteratively over 200 episodes with a 45 m success threshold. Overall success is 89.5%, with 100.0% success at offsets +4 and +10 and 72.0% success at offset +40. The median final distance is 27.8 m, while the long-tail mean and p90 distances show that long-horizon failures still matter.

## Interpretation

These results support the use of continuous heading and distance as the main navigation representation. Discrete commands remain useful as a diagnostic, but class imbalance makes them a weaker primary objective for long or geometrically diverse routes.

The closed-loop result is dataset-grounded rather than a physical flight demonstration. It shows whether the learned visual primitive can make repeated progress through the available AVGSO frame graph; it does not establish safety or real-world flight readiness.
