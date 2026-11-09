# FPGA Multiscale ORB Keypoint Extraction with Adaptive Non-Maximum Suppression

This repository contains the FPGA detector implementation, OpenCL host application, fixed-point software reference, and evaluation material associated with the research work:

**"FPGA Accelerator for Multiscale ORB Keypoint Extraction with Adaptive Non-Maximum Suppression"**

The released design targets detector-side acceleration of the ORB feature-extraction pipeline on AMD/Xilinx Alveo FPGA accelerators. It combines multiscale Harris keypoint detection, local non-maximum suppression, coordinate rescaling and merging across pyramid levels, bounded Top-K candidate reduction, and adaptive non-maximum suppression (ANMS).

Orientation estimation and BRIEF descriptor generation remain CPU-side in the complete heterogeneous pipeline and are outside the detector kernels released here.

## Overview

The detector uses two HLS kernels:

- `ms_harris_nms_merge`
- `anms_kernel`

The processing flow is:

```text
Input image
    |
    v
8-level image pyramid
    |
    v
Fixed-point Harris response
    |
    v
Thresholding + 5x5 local NMS
    |
    v
Coordinate rescaling to level 0
    |
    v
Multiscale candidate merge
    |
    v
Top-K candidate reduction
    |
    v
ANMS
    |
    v
Final keypoints
```

The first kernel implements the multiscale detector front-end and produces a merged keypoint list in level-0 coordinates. The second performs bounded candidate reduction and ANMS selection.

## Final Detector Configuration

| Parameter | Value |
|---|---:|
| Pyramid levels | 8 |
| Scale factor | 1.2 |
| Harris threshold | 100 |
| Local NMS window | 5x5 |
| Top-K input to ANMS (`K_top`) | 2048 |
| Final keypoints (`N_final`) | 1000 |
| ANMS robustness coefficient (`c`) | 0.9 |

## Repository Structure

```text
FPGA-Multiscale-ORB-ANMS/
|-- src/
|   |-- fpga/
|   |   |-- multiscale_harris_nms/
|   |   `-- anms/
|   |-- host/
|   `-- software_reference/
|-- evaluation/
|   |-- baselines/
|   |-- metrics/
|   |-- parameter_selection/
|   |-- validation/
|   `-- results/
|-- docs/
|   |-- BUILD.md
|   `-- DATASET.md
|-- CITATION.cff
`-- README.md
```

## Code Components

- `src/fpga/multiscale_harris_nms/` contains the multiscale Harris/NMS merge kernel.
- `src/fpga/anms/` contains the ANMS kernel.
- `src/host/` contains the OpenCL host application. XRT supplies the underlying Alveo runtime infrastructure; the host is not implemented with the native XRT API.
- `src/software_reference/fixed_point_harris_nms_anms.cpp` provides the fixed-point software reference used for controlled software/hardware comparison.
- `evaluation/baselines/opencv_orb_benchmark.cpp` provides the OpenCV ORB comparison baseline. It is a practical comparison reference and is not algorithm-equivalent to the fixed-point FPGA detector.

## FPGA Platforms and Toolchains

The implementation was evaluated on:

| Platform | Vitis environment | XRT environment |
|---|---|---|
| Alveo U50 | Vitis 2022.2 | XRT 2.15 |
| Alveo U55C | Vitis 2023.2 | XRT 2.18 |

ANMS synthesis and scheduling behavior was found to depend on the HLS toolchain version. Hardware results should therefore be reproduced using the corresponding validated platform and toolchain configuration.

## Dataset

Evaluation uses eight sequences from the Oxford VGG Affine Covariant Features benchmark, with six images per sequence for a total of 48 images.

The dataset is not redistributed with the repository. The evaluation workflow expects the local dataset under `data/VGG/`.

## Evaluation and Validation

The released evaluation material includes:

- fixed-point software-reference outputs;
- FPGA outputs after local NMS and multiscale merging;
- final FPGA outputs after ANMS;
- homography-based correspondence, repeatability, and localization analysis;
- Ripley-based spatial coverage analysis;
- C3I spatial consistency analysis;
- ANMS software/hardware agreement checks;
- ANMS selection auditing;
- supporting threshold-selection analysis;
- the OpenCV ORB comparison baseline.

The threshold-selection notebook is supporting parameter-selection material; its historical standalone generator is not part of the released final multiscale accelerator source.

## Reproducibility Scope

The repository supports inspection and reproduction of the detector-side implementation and its evaluation workflow. It includes the HLS kernels, OpenCL host source, fixed-point software reference, comparison baseline, stored keypoint outputs, evaluation notebooks, and validation scripts.

Generated FPGA binaries and implementation products are not included.

Exact FPGA performance can depend on synthesis, scheduling, placement, routing, platform configuration, compute-unit replication, and toolchain version.

## Documentation

| Document | Scope |
|---|---|
| [`docs/BUILD.md`](docs/BUILD.md) | FPGA compilation, host build, command-line arguments, and execution |
| [`docs/DATASET.md`](docs/DATASET.md) | Oxford VGG sequence mapping, image formats, homographies, and local layout |
| [`evaluation/README.md`](evaluation/README.md) | Evaluation outputs, baseline, metrics, parameter selection, and validation |

## Citation

Machine-readable software citation metadata is provided in [`CITATION.cff`](CITATION.cff).
