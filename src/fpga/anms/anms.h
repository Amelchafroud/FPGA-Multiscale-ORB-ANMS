#pragma once
#include <ap_int.h>

// Packed keypoint:
// [15:0]   = x
// [31:16]  = y
// [63:32]  = integer Harris/NMS score

typedef ap_uint<64> kp_t;

extern "C" void anms_kernel(
    const kp_t*  nms_kp_in,      // input keypoints from fixed HN kernel
    const int*   nms_kp_count,   // number of valid input keypoints
    kp_t*        anms_kp_out,    // output keypoints after ANMS
    int*         anms_kp_count,  // number of output keypoints
    int          K_top,          // top candidates kept by score
    int          N_final,        // final keypoints after ANMS
    float        c,              // ANMS strength factor (0.9)
    int          max_in          // safety cap from host
);
