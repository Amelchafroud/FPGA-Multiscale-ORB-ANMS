#include "anms.h"
#include <stdint.h>


// unpack / pack

static inline ap_uint<16> kp_x(kp_t w) { return w.range(15, 0); }
static inline ap_uint<16> kp_y(kp_t w) { return w.range(31, 16); }
static inline ap_uint<32> kp_sbits(kp_t w) { return w.range(63, 32); }

static inline kp_t pack_kp(ap_uint<16> x, ap_uint<16> y, ap_uint<32> sbits) {
#pragma HLS INLINE
    kp_t w = 0;
    w.range(15, 0)  = x;
    w.range(31, 16) = y;
    w.range(63, 32) = sbits;
    return w;
}

// Parallel minimum search over score array.
// It finds the minimum score and its index using PF lanes.
// Tie-break: keep the smallest index, matching the original sequential behavior.

template<int KMAX, int PF>
static void find_min_score_pf(ap_uint<32> ss[KMAX],
                              int K,
                              int &min_idx,
                              ap_uint<32> &min_s)
{
	#pragma HLS INLINE off

    ap_uint<32> lane_min_s[PF];
    int         lane_min_i[PF];
    #pragma HLS ARRAY_PARTITION variable=lane_min_s complete dim=1
    #pragma HLS ARRAY_PARTITION variable=lane_min_i complete dim=1

InitLaneMin:
    for (int p = 0; p < PF; p++) {
    	#pragma HLS UNROLL
        lane_min_s[p] = 0xFFFFFFFFu;
        lane_min_i[p] = KMAX;
    }

BlockMinLoop:
    for (int base = 0; base < K; base += PF) {
    	#pragma HLS PIPELINE II=1

LaneMinLoop:
        for (int p = 0; p < PF; p++) {
        	#pragma HLS UNROLL
            int idx = base + p;

            if (idx < K) {
                ap_uint<32> v = ss[idx];

                if ((v < lane_min_s[p]) ||
                    ((v == lane_min_s[p]) && (idx < lane_min_i[p]))) {
                    lane_min_s[p] = v;
                    lane_min_i[p] = idx;
                }
            }
        }
    }

    min_s   = lane_min_s[0];
    min_idx = lane_min_i[0];

ReduceLaneMin:
    for (int p = 1; p < PF; p++) {
    	#pragma HLS UNROLL
        if ((lane_min_s[p] < min_s) ||
            ((lane_min_s[p] == min_s) && (lane_min_i[p] < min_idx))) {
            min_s   = lane_min_s[p];
            min_idx = lane_min_i[p];
        }
    }
}
// Top-K selection by score bits

template<int KMAX>
static int select_topK(const kp_t* in, int in_count,
                       ap_uint<16> xs[KMAX],
                       ap_uint<16> ys[KMAX],
                       ap_uint<32> ss[KMAX],
                       int K)
{
	#pragma HLS INLINE off

    const int PF_TOPK = 8;  // parallel factor for minimum-score search

    int n = (in_count < K) ? in_count : K;

InitFill:
    for (int i = 0; i < K; i++) {
    	#pragma HLS PIPELINE II=1
        if (i < n) {
            kp_t w = in[i];
            xs[i] = kp_x(w);
            ys[i] = kp_y(w);
            ss[i] = kp_sbits(w);
        } else {
            xs[i] = 0;
            ys[i] = 0;
            ss[i] = 0;
        }
    }

    if (n == 0) return 0;

    // Find current minimum score in the initial top-K set.
    int min_idx = 0;
    ap_uint<32> min_s = 0;
    find_min_score_pf<KMAX, PF_TOPK>(ss, n, min_idx, min_s);

ScanRest:
    for (int i = n; i < in_count; i++) {
    	#pragma HLS PIPELINE II=1
        kp_t w = in[i];
        ap_uint<32> s = kp_sbits(w);

        if (s > min_s) {
            xs[min_idx] = kp_x(w);
            ys[min_idx] = kp_y(w);
            ss[min_idx] = s;

            // Recompute the minimum using PF_TOPK parallel lanes.
            find_min_score_pf<KMAX, PF_TOPK>(ss, K, min_idx, min_s);
        }
    }

    return n;
}

// Radix sort by score bits (descending)

template<int KMAX>
static void radix_sort_desc(ap_uint<16> xs[KMAX],
                            ap_uint<16> ys[KMAX],
                            ap_uint<32> ss[KMAX],
                            int K)
{
#pragma HLS INLINE off

    static ap_uint<16> tx[KMAX];
    static ap_uint<16> ty[KMAX];
    static ap_uint<32> ts[KMAX];
#pragma HLS BIND_STORAGE variable=tx type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=ty type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=ts type=ram_2p impl=bram

    static ap_uint<16> count[256];
    static ap_uint<16> pos[256];
#pragma HLS BIND_STORAGE variable=count type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=pos   type=ram_2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=count cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=pos   cyclic factor=16

PassLoop:
    for (int pass = 0; pass < 4; pass++) {

        // Clear histogram
ClearHist:
        for (int b = 0; b < 256; b++) {
        	#pragma HLS PIPELINE II=1
            count[b] = 0;
        }

        int shift = pass * 8;

        // Histogram
HistLoop:
        for (int i = 0; i < K; i++) {
        	#pragma HLS PIPELINE II=1
            ap_uint<8> key = (ap_uint<8>)((ss[i] >> shift) & 0xFF);
            count[key] = count[key] + 1;
        }

        // Prefix sums for DESCENDING order: buckets 255..0
        ap_uint<16> running = 0;
PrefixLoop:
        for (int b = 255; b >= 0; b--) {
        	#pragma HLS PIPELINE II=1
            pos[b] = running;
            running = running + count[b];
        }

        // Scatter
ScatterLoop:
        for (int i = 0; i < K; i++) {
        	#pragma HLS PIPELINE II=1
            ap_uint<8> key = (ap_uint<8>)((ss[i] >> shift) & 0xFF);
            ap_uint<16> p = pos[key];
            pos[key] = p + 1;

            tx[p] = xs[i];
            ty[p] = ys[i];
            ts[p] = ss[i];
        }

        // Copy back
CopyBackLoop:
        for (int i = 0; i < K; i++) {
        	#pragma HLS PIPELINE II=1
            xs[i] = tx[i];
            ys[i] = ty[i];
            ss[i] = ts[i];
        }
    }
}


// ANMS core
// - INTEGER score
// - we keep host interface c as float, but convert it ONCE to fixed Q12
// - stronger-than test uses integer arithmetic:
//
//      sj > c * si
//
//   implemented as:
//
//      sj * SCALE > c_q * si

template<int KMAX, int NMAX>
static int anms_core(ap_uint<16> xs[KMAX],
                     ap_uint<16> ys[KMAX],
                     ap_uint<32> ss[KMAX],
                     int K,
                     kp_t* out,
                     int N,
                     ap_uint<16> c_q12)
{
	#pragma HLS INLINE off

    // 1) Sort descending by score
    radix_sort_desc<KMAX>(xs, ys, ss, K);

    // 2) Compute suppression radius^2
    const uint32_t INF = 0xFFFFFFFFu;
    static uint32_t r2[KMAX];
    #pragma HLS BIND_STORAGE variable=r2 type=ram_2p impl=bram

    const uint32_t SCALE = 4096u; // Q12 scale
    const int PF_RADIUS = 16; //Parallel Factor for the ANMS radius computation.

    RadiusLoop:
        for (int i = 0; i < K; i++) {
            uint32_t best_lane[PF_RADIUS];
    #pragma HLS ARRAY_PARTITION variable=best_lane complete dim=1

    InitRadiusLanes:
            for (int p = 0; p < PF_RADIUS; p++) {
    #pragma HLS UNROLL
                best_lane[p] = INF;
            }

            ap_uint<16> xi = xs[i];
            ap_uint<16> yi = ys[i];
            ap_uint<32> si = ss[i];

            uint64_t rhs = (uint64_t)((uint32_t)c_q12) *
                           (uint64_t)((uint32_t)si);

    StrongerBlockLoop:
            for (int j0 = 0; j0 < i; j0 += PF_RADIUS) {
    #pragma HLS PIPELINE II=1

    CompareRadiusLanes:
                for (int p = 0; p < PF_RADIUS; p++) {
    #pragma HLS UNROLL
                    int j = j0 + p;

                    if (j < i) {
                        uint64_t lhs = ((uint64_t)((uint32_t)ss[j])) << 12;

                        if (lhs > rhs) {
                            int dx = (int)xi - (int)xs[j];
                            int dy = (int)yi - (int)ys[j];
                            uint32_t d2 = (uint32_t)(dx * dx + dy * dy);

                            if (d2 < best_lane[p]) {
                                best_lane[p] = d2;
                            }
                        }
                    }
                }
            }

            uint32_t best = best_lane[0];

    ReduceRadiusLanes:
            for (int p = 1; p < PF_RADIUS; p++) {
    #pragma HLS UNROLL
                if (best_lane[p] < best) {
                    best = best_lane[p];
                }
            }

            r2[i] = best;
        }

    // 3) Keep top-N by largest radius^2
    static ap_uint<16> ox[NMAX];
    static ap_uint<16> oy[NMAX];
    static ap_uint<32> os[NMAX];
    static uint32_t    or2[NMAX];
    #pragma HLS BIND_STORAGE variable=ox  type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=oy  type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=os  type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=or2 type=ram_2p impl=bram

    int filled = (K < N) ? K : N;

    FillInitialTopN:
    for (int i = 0; i < filled; i++) {
    	#pragma HLS PIPELINE II=1
        ox[i]  = xs[i];
        oy[i]  = ys[i];
        os[i]  = ss[i];
        or2[i] = r2[i];
    }

    if (filled == 0) {
        return 0;
    }

    // If K <= N, all candidates are already selected.
    if (K > N) {

        // Compute the current minimum radius only once after the initial fill.
        int min_i = 0;
        uint32_t minv = or2[0];

        FindInitialMinR:
		for (int t = 1; t < N; t++) {
        	#pragma HLS PIPELINE II=1
            if (or2[t] < minv) {
                minv = or2[t];
                min_i = t;
            }
        }
        ScanRemainingCandidates:
		for (int i = N; i < K; i++) {
            uint32_t val = r2[i];

            // Only if the new radius is better than the current minimum
            //replace and recompute the minimum.
            if (val > minv) {
                ox[min_i]  = xs[i];
                oy[min_i]  = ys[i];
                os[min_i]  = ss[i];
                or2[min_i] = val;

                min_i = 0;
                minv = or2[0];

                RecomputeMinR:
                for (int t = 1; t < N; t++) {
                	#pragma HLS PIPELINE II=1
                    if (or2[t] < minv) {
                        minv = or2[t];
                        min_i = t;
                    }
                }
            }
        }
    }

    WriteOut:
    for (int i = 0; i < filled; i++) {
    	#pragma HLS PIPELINE II=1
        out[i] = pack_kp(ox[i], oy[i], os[i]);
    }

    return filled;
}

// TOP kernel
extern "C" void anms_kernel(
    const kp_t*  nms_kp_in,
    const int*   nms_kp_count,
    kp_t*        anms_kp_out,
    int*         anms_kp_count,
    int          K_top,
    int          N_final,
    float        c,
    int          max_in)
{
#pragma HLS INTERFACE m_axi port=nms_kp_in      offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=nms_kp_count   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=anms_kp_out    offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=anms_kp_count  offset=slave bundle=gmem3

#pragma HLS INTERFACE s_axilite port=nms_kp_in     bundle=control
#pragma HLS INTERFACE s_axilite port=nms_kp_count  bundle=control
#pragma HLS INTERFACE s_axilite port=anms_kp_out   bundle=control
#pragma HLS INTERFACE s_axilite port=anms_kp_count bundle=control
#pragma HLS INTERFACE s_axilite port=K_top         bundle=control
#pragma HLS INTERFACE s_axilite port=N_final       bundle=control
#pragma HLS INTERFACE s_axilite port=c             bundle=control
#pragma HLS INTERFACE s_axilite port=max_in        bundle=control
#pragma HLS INTERFACE s_axilite port=return        bundle=control

    const int KMAX = 2048;  // compile-time max for Top-K
    const int NMAX = 1024;  // compile-time max for output

    if (K_top <= 0 || N_final <= 0 || K_top > KMAX || N_final > NMAX || N_final > K_top) {
        *anms_kp_count = 0;
        return;
    }

    int in_count = *nms_kp_count;
    if (in_count < 0) in_count = 0;
    if (in_count > max_in) in_count = max_in;

    if (in_count == 0) {
        *anms_kp_count = 0;
        return;
    }

    // Clamp c to [0,1], then quantize once to Q12
    float c_clamped = c;
    if (c_clamped < 0.0f) c_clamped = 0.0f;
    if (c_clamped > 1.0f) c_clamped = 1.0f;

    ap_uint<16> c_q12 = (ap_uint<16>)(c_clamped * 4096.0f + 0.5f);

    static ap_uint<16> xs[KMAX];
    static ap_uint<16> ys[KMAX];
    static ap_uint<32> ss[KMAX];
    #pragma HLS BIND_STORAGE variable=xs type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=ys type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=ss type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=xs cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=ys cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=ss cyclic factor=16 dim=1

    int Kuse = (in_count < K_top) ? in_count : K_top;

    if (in_count <= K_top) {
CopyAll:
        for (int i = 0; i < Kuse; i++) {
        	#pragma HLS PIPELINE II=1
            kp_t w = nms_kp_in[i];
            xs[i] = kp_x(w);
            ys[i] = kp_y(w);
            ss[i] = kp_sbits(w);
        }
Pad:
        for (int i = Kuse; i < K_top; i++) {
        	#pragma HLS PIPELINE II=1
            xs[i] = 0;
            ys[i] = 0;
            ss[i] = 0;
        }
    } else {
        select_topK<KMAX>(nms_kp_in, in_count, xs, ys, ss, K_top);
    }

    int outn = anms_core<KMAX, NMAX>(xs, ys, ss, Kuse, anms_kp_out, N_final, c_q12);
    *anms_kp_count = outn;
}
