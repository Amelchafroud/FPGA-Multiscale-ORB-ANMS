#pragma once
#include <hls_stream.h>
#include <ap_int.h>


// - 8 pyramid levels generated on device - scale factor between levels = 1.2 = 6/5
// - Each level runs the same Harris+NMS strategy
// - Each level streams variable-length keypoints internally
// - all NMS keypoints are rescaled to level-0 coordinates
// - one merge/writer block creates one contiguous DDR list for ANMS


constexpr int MAX_W = 1000;
constexpr int MS_LEVELS = 8;

// 8-bit grayscale pixel
typedef ap_uint<8>  u8;

//integer Harris score
typedef int   score_t;

// Final keypoint word written to DDR and consumed by ANMS: [15:0]   = x, [31:16]  = y, [63:32]  = score
typedef ap_uint<64> kp_word_t;

// Internal packet used only inside the multiscale kernel - scale_id needs 3 bits because we have levels 0..7.
struct kp_pkt_t {
    kp_word_t   kp;
    ap_uint<3>  scale_id;   // 0..7
    ap_uint<1>  last;       // 1 only for end-of-stream token
};

// stats_out layout:
// stats_out[0..7]   = Harris candidate count per level L0..L7
// stats_out[8..15]  = merged NMS accepted count per level L0..L7
// stats_out[16]     = merged NMS count total


extern "C" void ms_harris_nms_merge(
    const u8*    src,
    kp_word_t*   nms_kp_out,
    int*         nms_kp_count_total,
    int*         stats_out,
    score_t      thresh,
    int          W,
    int          H,
    int          active_levels,
    int          max_kp_merge,
    int          max_kp_l0,
    int          max_kp_l1,
    int          max_kp_l2,
    int          max_kp_l3,
    int          max_kp_l4,
    int          max_kp_l5,
    int          max_kp_l6,
    int          max_kp_l7);
