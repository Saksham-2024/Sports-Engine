# Graph Report - src  (2026-05-13)

## Corpus Check
- 52 files · ~277,825 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1265 nodes · 6626 edges · 21 communities detected
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 1013 edges (avg confidence: 0.78)
- Token cost: 1,200 input · 800 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Plotly JS Bundle (minified)|Plotly JS Bundle (minified)]]
- [[_COMMUNITY_Plotly Core Internals|Plotly Core Internals]]
- [[_COMMUNITY_TrackNetV3 Error Analysis UI|TrackNetV3 Error Analysis UI]]
- [[_COMMUNITY_Plotly Rendering Engine|Plotly Rendering Engine]]
- [[_COMMUNITY_Dataset Creation Pipeline|Dataset Creation Pipeline]]
- [[_COMMUNITY_jQuery Animation Utilities|jQuery Animation Utilities]]
- [[_COMMUNITY_Project Docs & Visual Results|Project Docs & Visual Results]]
- [[_COMMUNITY_Physics & Camera Geometry|Physics & Camera Geometry]]
- [[_COMMUNITY_XY Anomaly Cleaning|XY Anomaly Cleaning]]
- [[_COMMUNITY_Plotly Graph Construction|Plotly Graph Construction]]
- [[_COMMUNITY_Coordinate Normalizer|Coordinate Normalizer]]
- [[_COMMUNITY_Homography & Calibration|Homography & Calibration]]
- [[_COMMUNITY_Hit Frame Detection|Hit Frame Detection]]
- [[_COMMUNITY_Training Window Generator|Training Window Generator]]
- [[_COMMUNITY_venv Activation Scripts|venv Activation Scripts]]
- [[_COMMUNITY_Naive Bayes Predictor (archive)|Naive Bayes Predictor (archive)]]
- [[_COMMUNITY_Plotly Remote Entry|Plotly Remote Entry]]
- [[_COMMUNITY_Shuttle Coordinate Verifier|Shuttle Coordinate Verifier]]
- [[_COMMUNITY_Next-Shot Inference Mode|Next-Shot Inference Mode]]
- [[_COMMUNITY_Post-Hit Positioning Mode|Post-Hit Positioning Mode]]
- [[_COMMUNITY_Shuttle Coord Sanity Check|Shuttle Coord Sanity Check]]

## God Nodes (most connected - your core abstractions)
1. `o()` - 185 edges
2. `E()` - 165 edges
3. `N()` - 152 edges
4. `i()` - 124 edges
5. `T()` - 86 edges
6. `r()` - 85 edges
7. `c()` - 76 edges
8. `t()` - 73 edges
9. `h()` - 73 edges
10. `f()` - 71 edges

## Surprising Connections (you probably didn't know these)
- `TrackNetV3 — Shuttlecock Tracking Model` --references--> `TrackNetV3 Network Architecture Diagram`  [EXTRACTED]
  src/TrackNetV3/README.md → src/TrackNetV3/figure/NetArch.png
- `TrackNetV3 — Shuttlecock Tracking Model` --references--> `TrackNetV3 vs YOLOv7 vs TrackNetV2 Comparison Chart`  [EXTRACTED]
  src/TrackNetV3/README.md → src/TrackNetV3/figure/Comparison.png
- `TranSPORTmer Pipeline — Trajectory Transformer` --references--> `TranSPORTmer Test Results — Ground Truth vs Predictions`  [EXTRACTED]
  src/tranSPORTmer/README.md → src/tranSPORTmer/test results.png
- `d()` --calls--> `run()`  [INFERRED]
  /home/saksham/projects and programming/BTech_Project/src/TrackNetV3/tracknet/share/jupyter/nbextensions/jupyterlab-plotly/index.js → /home/saksham/projects and programming/BTech_Project/src/dataset_creation/apply_physics.py
- `d()` --calls--> `run()`  [INFERRED]
  /home/saksham/projects and programming/BTech_Project/src/TrackNetV3/tracknet/share/jupyter/labextensions/jupyterlab-plotly/static/478.f9a7116e09cbfb956212.js → /home/saksham/projects and programming/BTech_Project/src/dataset_creation/apply_physics.py

## Hyperedges (group relationships)
- **End-to-End Dataset Creation Pipeline** — readme_download_videos, readme_segment_pass, readme_homography, readme_shuttle_pass, readme_apply_physics, readme_transformer_dataset [EXTRACTED 1.00]
- **TranSPORTmer Training Pipeline** — readme_clean_xy, readme_normalize_masks, readme_training_windows, readme_transformer_arch, readme_train_loop, readme_inference_modes [EXTRACTED 1.00]

## Communities

### Community 0 - "Plotly JS Bundle (minified)"
Cohesion: 0.05
Nodes (294): a(), r(), E(), _(), _a(), aa(), ac(), accessToken() (+286 more)

### Community 1 - "Plotly Core Internals"
Cohesion: 0.05
Nodes (268): _(), a(), ac(), accessToken(), ae(), Ai(), Al(), an() (+260 more)

### Community 2 - "TrackNetV3 Error Analysis UI"
Cohesion: 0.02
Nodes (123): change_dropdown(), save_corrected_result(), show_frame(), Dataset, Return the rally index-path mapping dictionary., Return the corresponding rally index of the rally directory., Shuttlecock_Trajectory_Dataset             Dataset description: https://hackmd.i, Parse the split from the rally directory. (+115 more)

### Community 3 - "Plotly Rendering Engine"
Cohesion: 0.05
Nodes (169): u(), ei(), hi(), kn(), ae(), ai(), Ao(), At() (+161 more)

### Community 4 - "Dataset Creation Pipeline"
Cohesion: 0.05
Nodes (37): To(), demo(), load_test_sequence(), predict_opponent_movement(), predict_trajectory(), Inference Module: Two Modes  Mode 1: Next-Shot Trajectory Prediction   - Given p, Infer shot type from shuttle trajectory profile., Confidence based on shuttle observability in history. (+29 more)

### Community 5 - "jQuery Animation Utilities"
Cohesion: 0.09
Nodes (10): c(), d, f(), g(), h(), j(), m(), p() (+2 more)

### Community 6 - "Project Docs & Visual Results"
Cohesion: 0.08
Nodes (31): TrackNetV3 vs YOLOv7 vs TrackNetV2 Comparison Chart, TrackNetV3 Error Analysis UI Screenshot, TrackNetV3 Network Architecture Diagram, TranSPORTmer Test Results — Ground Truth vs Predictions, apply_physics.py — 3D Physics Projection (BVP), Background Median Estimation, BVP Solver — Gravity + Air Drag Physics Model, 01_clean_xy_anomalies.py — XY Outlier Handler (+23 more)

### Community 7 - "Physics & Camera Geometry"
Cohesion: 0.14
Nodes (24): assign_hitters(), _build_row(), bvp_objective(), camera_ray(), get_hitter_bbox(), get_player_feet_y(), intersect_ray_y_plane(), pixel_to_court_homography() (+16 more)

### Community 9 - "XY Anomaly Cleaning"
Cohesion: 0.15
Nodes (11): main(), STEP 1: Clean XY Anomalies in Shuttle Coordinates  Strategy:   - Bands ≤5 frames, Process single segment for XY anomalies.                  Returns:             L, Process entire dataframe.                  Returns:             Cleaned datafram, Handle scattered XY anomalies in shuttle coordinates., Main preprocessing pipeline., Args:             x_bounds: Reasonable bounds for X             y_bounds: Reason, Check if XY coordinate is anomalous. (+3 more)

### Community 10 - "Plotly Graph Construction"
Cohesion: 0.18
Nodes (3): h(), s(), y()

### Community 11 - "Coordinate Normalizer"
Cohesion: 0.21
Nodes (8): BadmintonNormalizer, main(), STEP 2: Normalize to [0, 1] with NaN-Mask Awareness  Strategy:   - Normalize val, Validate normalized data is in [0, 1] or NaN.                  Returns:, Main normalization pipeline., Normalize badminton data to [0, 1] preserving NaN/masks.          Bounds (from d, Normalize single value to [0, 1].                  Args:             value: floa, Normalize entire dataframe.                  Args:             df: DataFrame wit

### Community 12 - "Homography & Calibration"
Cohesion: 0.33
Nodes (8): approximate_camera_matrix(), _cached_basenames(), calibrate(), gather_matrices(), Return set of basenames from a dict whose keys may be relative or absolute paths, Runs calibrate() only for videos NOT already in the caches.      Parameters, Reasonable intrinsic approximation for broadcast cameras., Opens one frame and asks the user to click 6 points:       Points 1-4 : court co

### Community 13 - "Hit Frame Detection"
Cohesion: 0.39
Nodes (7): detect_hit_frames(), main(), natural_sort_key(), process_match(), Combine step: merge shuttle tracks + player positions per segment, detect hit fr, Process a single match: load segments, shuttle CSVs, player JSON.     Returns a, Detect hit frames from raw shuttle pixel tracks.      A hit is flagged when BOTH

### Community 14 - "Training Window Generator"
Cohesion: 0.42
Nodes (3): main(), save_windows(), TrainingWindowGenerator

### Community 15 - "venv Activation Scripts"
Cohesion: 0.53
Nodes (4): Get-PyVenvConfig(), global:deactivate(), global:_OLD_VIRTUAL_PROMPT(), global:prompt()

### Community 16 - "Naive Bayes Predictor (archive)"
Cohesion: 0.6
Nodes (3): extract_usable_data(), get_data(), prob()

### Community 17 - "Plotly Remote Entry"
Cohesion: 0.67
Nodes (2): P(), u()

### Community 18 - "Shuttle Coordinate Verifier"
Cohesion: 0.67
Nodes (1): main()

### Community 27 - "Next-Shot Inference Mode"
Cohesion: 1.0
Nodes (1): Mode 1: Predict shuttle trajectory for next shot.                  Args:

### Community 28 - "Post-Hit Positioning Mode"
Cohesion: 1.0
Nodes (1): Mode 2: Predict opponent's positioning after they respond to shot.

### Community 37 - "Shuttle Coord Sanity Check"
Cohesion: 1.0
Nodes (1): verify_shuttle_coords.py — Coordinate Sanity Checker

## Knowledge Gaps
- **75 isolated node(s):** `Detect hit frames from raw shuttle pixel tracks.      A hit is flagged when BOTH`, `Process a single match: load segments, shuttle CSVs, player JSON.     Returns a`, `Get video duration in seconds using ffprobe.`, `return sa, cond in {0, 1, 2, 3}         cond :  last sa  ->   sa           0  :`, `Reasonable intrinsic approximation for broadcast cameras.` (+70 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Plotly Remote Entry`** (4 nodes): `remoteEntry.e2bf80f6515beeab9026.js`, `P()`, `u()`, `remoteEntry.e2bf80f6515beeab9026.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Shuttle Coordinate Verifier`** (3 nodes): `verify_shuttle_coords.py`, `verify_shuttle_coords.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Next-Shot Inference Mode`** (1 nodes): `Mode 1: Predict shuttle trajectory for next shot.                  Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Post-Hit Positioning Mode`** (1 nodes): `Mode 2: Predict opponent's positioning after they respond to shot.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Shuttle Coord Sanity Check`** (1 nodes): `verify_shuttle_coords.py — Coordinate Sanity Checker`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `test()` connect `Plotly JS Bundle (minified)` to `Plotly Core Internals`, `TrackNetV3 Error Analysis UI`, `Plotly Rendering Engine`?**
  _High betweenness centrality (0.130) - this node is a cross-community bridge._
- **Why does `i()` connect `Plotly Core Internals` to `Plotly JS Bundle (minified)`, `Plotly Graph Construction`, `Plotly Rendering Engine`, `jQuery Animation Utilities`?**
  _High betweenness centrality (0.087) - this node is a cross-community bridge._
- **Why does `o()` connect `Plotly Core Internals` to `Plotly JS Bundle (minified)`, `Plotly Rendering Engine`, `Dataset Creation Pipeline`, `jQuery Animation Utilities`, `Plotly Figure Layout`, `Plotly Graph Construction`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Are the 104 inferred relationships involving `o()` (e.g. with `ln()` and `jn()`) actually correct?**
  _`o()` has 104 INFERRED edges - model-reasoned connections that need verification._
- **Are the 103 inferred relationships involving `E()` (e.g. with `ye()` and `xe()`) actually correct?**
  _`E()` has 103 INFERRED edges - model-reasoned connections that need verification._
- **Are the 79 inferred relationships involving `N()` (e.g. with `ze()` and `an()`) actually correct?**
  _`N()` has 79 INFERRED edges - model-reasoned connections that need verification._
- **Are the 76 inferred relationships involving `i()` (e.g. with `Ne()` and `Hn()`) actually correct?**
  _`i()` has 76 INFERRED edges - model-reasoned connections that need verification._