# Build and Execution Guide

This document describes how to build and run the software reference, OpenCV baseline, FPGA kernels, and OpenCL host application provided in this repository.

The commands below assume a Linux development host. FPGA compilation requires a supported AMD/Xilinx Vitis installation, XRT, and an installed Alveo platform definition.

## 1. Validated Environments

The reported hardware experiments used the following principal environments:

| Platform | Vitis | XRT |
|---|---|---|
| Alveo U50 | Vitis 2022.2 | XRT 2.15 |
| Alveo U55C | Vitis 2023.2 | XRT 2.18 |

HLS scheduling can change across tool versions. During validation of the ANMS implementation, different scheduling behavior was observed between Vitis HLS 2022.2 and 2023.2. Corrected U55C validation therefore also included an ANMS implementation synthesized with the 2022.2 HLS flow.

For hardware reproduction, use the toolchain associated with the target experiment. The repository does not treat the observed scheduling difference as a confirmed Vitis software defect.

## 2. Repository Paths Used by the Build

The relevant sources are:

```text
src/fpga/multiscale_harris_nms/
    multiscale_harris_nms.cpp
    multiscale_harris_nms.h

src/fpga/anms/
    anms.cpp
    anms.h

src/host/
    host.cpp
    read_image.cpp
    read_image.h
    color_conversion.cpp
    color_conversion.h
    config.h

src/software_reference/
    fixed_point_harris_nms_anms.cpp

evaluation/baselines/
    opencv_orb_benchmark.cpp
```

Generated Vitis build products are intentionally not tracked.

## 3. Software Reference

The fixed-point software reference can be compiled independently of the FPGA toolchain.

```bash
g++ -O3 -std=c++17 \
    src/software_reference/fixed_point_harris_nms_anms.cpp \
    -o fixed_point_harris_nms_anms
```

Run it with an explicit dataset root and output directory:

```bash
./fixed_point_harris_nms_anms <dataset_root> <output_directory>
```

For the repository dataset layout:

```bash
./fixed_point_harris_nms_anms \
    ./data/VGG \
    ./software_reference_outputs
```

## 4. OpenCV ORB Baseline

OpenCV 4 development files are required.

A typical Linux build is:

```bash
g++ -O3 -std=c++17 \
    evaluation/baselines/opencv_orb_benchmark.cpp \
    -o opencv_orb_benchmark \
    $(pkg-config --cflags --libs opencv4)
```

Run:

```bash
./opencv_orb_benchmark \
    <dataset_root> \
    <output_directory>
```

For example:

```bash
./opencv_orb_benchmark \
    ./data/VGG \
    ./opencv_orb_outputs
```

The baseline configuration used by the program is:

```text
nfeatures   = 1000
scaleFactor = 1.2
nlevels     = 8
```

The OpenCV program is a comparison baseline, not an algorithm-equivalent implementation of the FPGA detector.

## 5. FPGA Environment

Load the Vitis and XRT environments corresponding to the target platform. Installation paths depend on the local system.

```bash
source <VITIS_INSTALLATION>/settings64.sh
source <XRT_INSTALLATION>/setup.sh
```

Confirm that the required tools are available:

```bash
which v++
which xbutil
```

Set the platform name or `.xpfm` path:

```bash
export PLATFORM=<platform-name-or-xpfm-path>
```

Create a build directory:

```bash
mkdir -p build
```

## 6. Kernel Compilation

### 6.1 Multiscale Harris/NMS Kernel

Compile the multiscale detector kernel to an FPGA object:

```bash
v++ -c -t hw \
    --platform "$PLATFORM" \
    -k ms_harris_nms_merge \
    src/fpga/multiscale_harris_nms/multiscale_harris_nms.cpp \
    -o build/ms_harris_nms_merge.xo
```

### 6.2 ANMS Kernel

Compile the ANMS kernel:

```bash
v++ -c -t hw \
    --platform "$PLATFORM" \
    -k anms_kernel \
    src/fpga/anms/anms.cpp \
    -o build/anms_kernel.xo
```

These commands show the minimal source-to-`.xo` flow. Clock constraints and other implementation options used for a specific hardware experiment must be supplied through the corresponding Vitis build settings.

## 7. FPGA Linking

A basic single-instance link is:

```bash
v++ -l -t hw \
    --platform "$PLATFORM" \
    build/ms_harris_nms_merge.xo \
    build/anms_kernel.xo \
    -o build/orb_anms.xclbin
```

Multi-compute-unit experiments require replicated instances of both kernels and the associated connectivity settings at link time. Those settings are platform- and experiment-specific and are not implied by the minimal command above.

Do not use a single-instance link command to claim reproduction of a reported multi-CU result.

## 8. OpenCL Host Compilation

The released host uses the OpenCL API. XRT supplies the runtime and OpenCL libraries used by the Alveo platform; the host is not implemented with the native XRT API.

A representative build command is:

```bash
g++ -std=c++14 -O2 -Wall \
    -D__USE_XOPEN2K8 \
    -I"${XILINX_XRT}/include" \
    src/host/host.cpp \
    src/host/read_image.cpp \
    src/host/color_conversion.cpp \
    -L"${XILINX_XRT}/lib" \
    -Wl,-rpath-link,"${XILINX_XRT}/lib" \
    -lxilinxopencl \
    -lz \
    -lpng \
    -lpthread \
    -lrt \
    -o multiscale_harris_nms_anms
```

The exact include and library paths depend on the installed XRT environment.

## 9. Host Command-Line Interface

The current host accepts:

```text
multiscale_harris_nms_anms
    <xclbin>
    [vgg_root]
    [thresh]
    [max_kp_merge_req]
    [dev]
    [save_outputs]
    [K_top]
    [N_final]
    [c]
    [active_levels]
    [max_kp_l0_req]
    [max_kp_l1_req]
    [max_kp_l2_req]
    [max_kp_l3_req]
    [max_kp_l4_req]
    [max_kp_l5_req]
    [max_kp_l6_req]
    [max_kp_l7_req]
    [clean_benchmark]
    [num_slots]
    [flush_each_frame]
    [trace_host]
    [warmup_frames]
    [num_lanes]
    [force_single_cu]
    [power_repeats]
```

The arguments are interpreted in this order:

| Position | Argument | Default | Purpose |
|---:|---|---:|---|
| 1 | `xclbin` | required | FPGA binary |
| 2 | `vgg_root` | source-defined | Dataset root |
| 3 | `thresh` | 100 | Harris threshold |
| 4 | `max_kp_merge_req` | 0 | Merged-keypoint capacity; `0` selects automatic sizing |
| 5 | `dev` | 0 | Accelerator device index |
| 6 | `save_outputs` | 0 | Enable detector-output export |
| 7 | `K_top` | 2048 | Candidate count passed to ANMS |
| 8 | `N_final` | 1000 | Maximum final ANMS keypoints |
| 9 | `c` | 0.9 | ANMS robustness coefficient |
| 10 | `active_levels` | 8 | Number of active pyramid levels |
| 11-18 | `max_kp_l0_req` ... `max_kp_l7_req` | 0 | Per-level capacities; `0` selects automatic sizing |
| 19 | `clean_benchmark` | 1 | Benchmark-control flag |
| 20 | `num_slots` | 2 | Host pipeline slots |
| 21 | `flush_each_frame` | 0 | Force per-frame command completion when enabled |
| 22 | `trace_host` | 0 | Enable host tracing |
| 23 | `warmup_frames` | 1 | Warm-up frames excluded from measured execution |
| 24 | `num_lanes` | 1 | Number of accelerator lanes requested by the host |
| 25 | `force_single_cu` | 0 | Request single-CU execution; `0` disables forcing |
| 26 | `power_repeats` | 1 | Dataset repetitions used by power-oriented runs |

The current host clamps `active_levels` to 1-8, `num_slots` to 2-12, `warmup_frames` to 0-8, and `num_lanes` to 1-6. A requested lane count must also be compatible with the compute-unit instances present in the loaded `.xclbin`.

For reproducible repository usage, pass `vgg_root` explicitly rather than relying on a machine-specific source default.

## 10. Final Detector Run

For the final detector configuration, a concise single-lane invocation is:

```bash
./multiscale_harris_nms_anms \
    <xclbin> \
    ./data/VGG \
    100 \
    0 \
    0 \
    0 \
    2048 \
    1000 \
    0.9 \
    8
```

This sets the scientific detector parameters explicitly while leaving the capacity arguments in automatic mode.

The three zeros after the dataset path correspond to:

```text
max_kp_merge_req = 0   -> automatic capacity
dev              = 0   -> first accelerator device
save_outputs     = 0   -> no output export
```

A zero capacity request does not mean zero keypoints.

## 11. Full Benchmark-Style Invocation

The remaining host controls can be supplied explicitly when benchmark behavior must be fixed:

```bash
./multiscale_harris_nms_anms \
    <xclbin> \
    ./data/VGG \
    100 \
    0 \
    0 \
    0 \
    2048 \
    1000 \
    0.9 \
    8 \
    0 0 0 0 0 0 0 0 \
    1 \
    2 \
    0 \
    0 \
    1 \
    1 \
    0 \
    1
```

This example uses automatic per-level capacities, clean benchmark mode, two host slots, one warm-up frame, one accelerator lane, no forced CU, and one dataset repetition.

Multi-CU runs require an `.xclbin` containing the corresponding replicated kernel instances. Set `num_lanes` only to a value supported by that binary.

## 12. Output Export

When `save_outputs=1`, the host creates detector output files under its configured output directory.

Stored reference outputs used by the repository evaluation are already organized under:

```text
evaluation/results/sw_fixed/
evaluation/results/hw_local_nms/
evaluation/results/hw_anms/
```

Do not mix newly generated experimental outputs with the stored reference files unless the purpose is to replace a validated result set deliberately.

## 13. Reproducibility Notes

- The U50 and U55C results were obtained with different validated Vitis/XRT environments.
- HLS scheduling and routed implementation results can vary across Vitis versions.
- Reported multi-CU results require the corresponding replicated kernel topology; a single-CU `.xclbin` is not equivalent.
- FPGA timing and resource results depend on the target platform, clock constraints, connectivity, compute-unit count, and routed implementation.
- The OpenCV baseline is a comparison reference and does not reproduce the FPGA detector algorithm.
- The fixed-point software reference is intended for controlled software-hardware comparison; universal bit-exact equivalence of the complete detector is not claimed.
- The Oxford VGG dataset is not distributed with the repository. See [`DATASET.md`](DATASET.md) for preparation instructions.
