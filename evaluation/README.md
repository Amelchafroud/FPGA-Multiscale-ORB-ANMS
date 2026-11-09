# Evaluation

This directory contains the evaluation and validation material associated with the FPGA multiscale ORB keypoint detector, its fixed-point software reference, and the OpenCV ORB comparison baseline.

## Directory Structure

```text
evaluation/
|-- baselines/
|   `-- opencv_orb_benchmark.cpp
|-- metrics/
|   `-- keypoint_metrics.ipynb
|-- parameter_selection/
|   `-- nms_threshold_selection.ipynb
|-- validation/
|   |-- anms_validation.ipynb
|   |-- anms_sw_hw_agreement.py
|   `-- anms_selection_audit.py
`-- results/
    |-- sw_fixed/
    |-- hw_local_nms/
    |-- hw_anms/
    `-- threshold_sweep/
```

## OpenCV Baseline

`baselines/opencv_orb_benchmark.cpp` provides the OpenCV ORB comparison baseline corresponding to the evaluation configuration:

```text
nfeatures   = 1000
scaleFactor = 1.2
nlevels     = 8
```

The remaining ORB parameters use OpenCV defaults. This baseline is a practical comparison reference and is not algorithm-equivalent to the fixed-point FPGA detector.

## Evaluation Metrics

`metrics/keypoint_metrics.ipynb` contains the keypoint-quality metrics used in the full-dataset evaluation, including homography-based localization/repeatability, Ripley-based spatial coverage analysis, and C3I.

The notebook operates on exported keypoint files and the Oxford VGG ground-truth homographies. Dataset organization is documented in [`../docs/DATASET.md`](../docs/DATASET.md).

## ANMS Validation

The files under `validation/` support software/hardware validation of the detector output:

- `anms_validation.ipynb` provides the ANMS validation workflow.
- `anms_sw_hw_agreement.py` compares fixed-point software and FPGA outputs at the local-NMS and final ANMS stages.
- `anms_selection_audit.py` analyzes the effect of Top-K and ANMS selection.

The FPGA ANMS implementation itself is under `src/fpga/anms/`.

## Parameter Selection

`parameter_selection/nms_threshold_selection.ipynb` records the auxiliary analysis used for NMS/Harris threshold selection. It is supporting evaluation material rather than a separate implementation of the final multiscale accelerator.

## Results

The stored result directories correspond to the following roles:

| Directory | Purpose |
|---|---|
| `sw_fixed/` | Fixed-point software-reference outputs |
| `hw_local_nms/` | FPGA local-NMS outputs |
| `hw_anms/` | Final FPGA ANMS outputs |
| `threshold_sweep/` | Optional threshold-selection outputs |

The final detector configuration represented by the ANMS result files is:

```text
Pyramid levels    = 8
Harris threshold  = 100
K_top             = 2048
N_final           = 1000
c_robust          = 0.90
```

The included notebooks and scripts expose the evaluation methodology. They should not be interpreted as a frozen reproduction of every numerical table in the associated article.

Generated binaries, FPGA build products, and the Oxford VGG dataset are not stored in this directory.

## Reproduction

Build and execution instructions are provided in [`../docs/BUILD.md`](../docs/BUILD.md). Dataset layout and sequence mapping are provided in [`../docs/DATASET.md`](../docs/DATASET.md).
