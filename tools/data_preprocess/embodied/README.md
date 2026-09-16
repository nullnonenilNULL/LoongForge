# Embodied Data Preprocessing

Dataset preparation and data generation for embodied / VLA training.

| Path | Purpose |
|---|---|
| [`ego2robot/`](./ego2robot/) | Generate robot training data from first-person human videos — MuJoCo retargeting to **16 dual-arm morphologies**, hand removal, and background compositing into a **LeRobot v3.0** dataset. Standalone pipeline with its own dependencies; see its local README. |
| `dreamzero` (below) | Prepare an already-collected LeRobot dataset and precompute frozen-condition features for **DreamZero** (Wan2.2) training. |

## dreamzero

Prepare a LeRobot dataset and precompute frozen-condition features for
LoongForge **DreamZero** (Wan2.2) training.

| Script | Purpose |
|---|---|
| `prepare_dataset.py` | Validate a source LeRobot layout and write the DreamZero/GEAR-ready dataset. |
| `precompute_features.py` | Precompute frozen-condition (e.g. text/vision) features into structured cache artifacts. |
| `validate_precomputed_feature_artifact.py` | Sanity-check the precomputed feature cache before training. |
| `cache_precompute/` | Cache config, feature computation, and storage backends used by `precompute_features.py`. |

Run each script with `--help` for the full argument list. Precomputing features
lets DreamZero training skip frozen-module forward passes at run time via
cache-aware data loading.
