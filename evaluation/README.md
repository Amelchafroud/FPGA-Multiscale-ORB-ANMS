# Evaluation

This directory contains the evaluation and validation material associated with the FPGA multiscale ORB keypoint detector.

## Structure

- `metrics/`
  - `keypoint_metrics.ipynb`: implementation of the keypoint quality metrics used during evaluation.

- `validation/`
  - `anms_validation.ipynb`: ANMS validation notebook.
  - `anms_sw_hw_agreement.py`: compares fixed-point software and FPGA outputs at the local-NMS and final ANMS stages.
  - `anms_selection_audit.py`: analyzes differences introduced by Top-K and ANMS selection.

- `parameter_selection/`
  - `nms_threshold_selection.ipynb`: threshold-selection analysis.

- `results/`
  - `sw_fixed/`: fixed-point software reference outputs used by the SW/HW validation scripts.
  - `hw_local_nms/`: FPGA local-NMS outputs for the 48 Oxford VGG images.
  - `hw_anms/`: final FPGA ANMS outputs for the 48 Oxford VGG images.
  - `threshold_sweep/`: location expected by the threshold-selection notebook for optional sweep data.

The final configuration represented by the ANMS result files is:

- 8 pyramid levels
- Harris threshold: 100
- `K_top = 2048`
- `N_final = 1000`
- `c_robust = 0.90`

The included metric notebooks expose the evaluation methodology. They should not be interpreted as a frozen reproduction of every numerical table in the published article.
