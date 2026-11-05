#include "multiscale_harris_nms.h"


// 1) Generic pixel-stream helpers

// DDR -> pixel stream
static void read_image_to_stream(
    const u8* src,
    hls::stream<u8>& s_out,
    int W,
    int H)
{
#pragma HLS INLINE off
ReadRows:
    for (int y = 0; y < H; ++y) {
    ReadCols:
        for (int x = 0; x < W; ++x) {
#pragma HLS PIPELINE II=1
            s_out.write(src[y * W + x]);
        }
    }
}

// One input pixel stream -> two identical output streams
static void duplicate_pixel_stream(
    hls::stream<u8>& s_in,
    hls::stream<u8>& s_out_a,
    hls::stream<u8>& s_out_b,
    int W,
    int H)
{
#pragma HLS INLINE off
DupRows:
    for (int y = 0; y < H; ++y) {
    DupCols:
        for (int x = 0; x < W; ++x) {
#pragma HLS PIPELINE II=1
            u8 p = s_in.read();
            s_out_a.write(p);
            s_out_b.write(p);
        }
    }
}

// Drain an entire pixel stream: Used when a level is disabled but the upstream producer remains active.
static void discard_pixel_stream(
    hls::stream<u8>& s_in,
    int W,
    int H)
{
#pragma HLS INLINE off
DiscardPixelRows:
    for (int y = 0; y < H; ++y) {
    DiscardPixelCols:
        for (int x = 0; x < W; ++x) {
#pragma HLS PIPELINE II=1
            (void)s_in.read();
        }
    }
}

static void discard_score_stream(
    hls::stream<score_t>& s_in,
    int W,
    int H)
{
#pragma HLS INLINE off
DiscardScoreRows:
    for (int y = 0; y < H; ++y) {
    DiscardScoreCols:
        for (int x = 0; x < W; ++x) {
#pragma HLS PIPELINE II=1
            (void)s_in.read();
        }
    }
}

// Gaussian 5x5 smoothing followed by 5/6 decimation.
// This implements one pyramid step with scale factor 1.2:
// next level size = floor(current size * 5 / 6).
static void gaussian5x5_downsample_5over6(
    hls::stream<u8>& s_in,
    hls::stream<u8>& s_out,
    int W,
    int H)
{
	#pragma HLS INLINE off

	// 5-tap sampled Gaussian kernel, sigma approximately 1.
	// Fixed-point Q8 weights, normalized so that sum(g)=256.
	const ap_uint<8> g[5] = {14, 63, 102, 63, 14};
	#pragma HLS ARRAY_PARTITION variable=g complete

    u8 lb1[MAX_W], lb2[MAX_W], lb3[MAX_W], lb4[MAX_W];
#pragma HLS BIND_STORAGE variable=lb1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=lb2 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=lb3 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=lb4 type=ram_2p impl=bram

    u8 win[5][5];
#pragma HLS ARRAY_PARTITION variable=win complete dim=0

InitLB_56:
    for (int x = 0; x < W; ++x) {
#pragma HLS PIPELINE II=1
        lb1[x] = 0;
        lb2[x] = 0;
        lb3[x] = 0;
        lb4[x] = 0;
    }

InitWin_56:
    for (int r = 0; r < 5; ++r) {
        for (int c = 0; c < 5; ++c) {
#pragma HLS PIPELINE II=1
            win[r][c] = 0;
        }
    }

    int y_acc = 0;

Rows_56:
    for (int y = 0; y < H; ++y) {

        // Bresenham-like keep rule:
        // keep exactly floor(H*5/6) rows over the full image.
        y_acc += 5;
        bool keep_y = false;
        if (y_acc >= 6) {
            keep_y = true;
            y_acc -= 6;
        }

        int x_acc = 0;

    Cols_56:
        for (int x = 0; x < W; ++x) {
#pragma HLS PIPELINE II=1
            u8 pix = s_in.read();

            // Shift 5x5 window left
            for (int r = 0; r < 5; ++r) {
#pragma HLS UNROLL
                win[r][0] = win[r][1];
                win[r][1] = win[r][2];
                win[r][2] = win[r][3];
                win[r][3] = win[r][4];
            }

            // Bring new rightmost column from line buffers + current pixel
            win[0][4] = lb4[x];
            win[1][4] = lb3[x];
            win[2][4] = lb2[x];
            win[3][4] = lb1[x];
            win[4][4] = pix;

            // Update line buffers
            lb4[x] = lb3[x];
            lb3[x] = lb2[x];
            lb2[x] = lb1[x];
            lb1[x] = pix;

            // Keep exactly floor(W*5/6) columns over the row.
            x_acc += 5;
            bool keep_x = false;
            if (x_acc >= 6) {
                keep_x = true;
                x_acc -= 6;
            }

            if (keep_x && keep_y) {
            	ap_uint<32> sum = 0;

                for (int ry = 0; ry < 5; ++ry) {
#pragma HLS UNROLL
                    for (int rx = 0; rx < 5; ++rx) {
#pragma HLS UNROLL
                    	sum += (ap_uint<32>)win[ry][rx] *
                    	       (ap_uint<32>)g[ry] *
                    	       (ap_uint<32>)g[rx];
                    }
                }

                u8 outp = (u8)((sum + 32768) >> 16); // divide by 65536 with rounding
                s_out.write(outp);
            }
        }
    }
}

// 2) Harris score pipeline

static score_t harris_score_7x7(const u8 window[7][7])
{
	#pragma HLS INLINE

    const int g1d[7] = {1, 6, 15, 20, 15, 6, 1};
    #pragma HLS ARRAY_PARTITION variable=g1d complete

    long long rowIxx[5];
    long long rowIyy[5];
    long long rowIxy[5];
    #pragma HLS ARRAY_PARTITION variable=rowIxx complete
    #pragma HLS ARRAY_PARTITION variable=rowIyy complete
    #pragma HLS ARRAY_PARTITION variable=rowIxy complete

    HarrisY:
    for (int wy = 1; wy <= 5; ++wy) {
    	#pragma HLS UNROLL
        long long acc_xx = 0;
        long long acc_yy = 0;
        long long acc_xy = 0;

        HarrisX:
        for (int wx = 1; wx <= 5; ++wx) {
        	#pragma HLS UNROLL
            int gx = (int)window[wy][wx + 1] - (int)window[wy][wx - 1];
            int gy = (int)window[wy + 1][wx] - (int)window[wy - 1][wx];

            int gxx = gx * gx;
            int gyy = gy * gy;
            int gxy = gx * gy;

            int gw = g1d[wy] * g1d[wx];

            acc_xx += (long long)gw * gxx;
            acc_yy += (long long)gw * gyy;
            acc_xy += (long long)gw * gxy;
        }

        rowIxx[wy - 1] = acc_xx;
        rowIyy[wy - 1] = acc_yy;
        rowIxy[wy - 1] = acc_xy;
    }

    long long Ixx01 = rowIxx[0] + rowIxx[1];
    long long Ixx23 = rowIxx[2] + rowIxx[3];
    long long Ixx   = Ixx01 + Ixx23 + rowIxx[4];

    long long Iyy01 = rowIyy[0] + rowIyy[1];
    long long Iyy23 = rowIyy[2] + rowIyy[3];
    long long Iyy   = Iyy01 + Iyy23 + rowIyy[4];

    long long Ixy01 = rowIxy[0] + rowIxy[1];
    long long Ixy23 = rowIxy[2] + rowIxy[3];
    long long Ixy   = Ixy01 + Ixy23 + rowIxy[4];

    long long detM  = Ixx * Iyy - Ixy * Ixy;
    long long trace = Ixx + Iyy;

    long long R = detM - ((trace * trace) >> 5);
    R >>= 20;

    if (R < 0)          R = 0;
    if (R > 0x7FFFFFFF) R = 0x7FFFFFFF;

    return (score_t)R;
}

static void process_harris7x7( hls::stream<u8>& s_in, hls::stream<score_t>& s_score, int W, int H)
{
	#pragma HLS INLINE off

    u8 lb1[MAX_W], lb2[MAX_W], lb3[MAX_W], lb4[MAX_W], lb5[MAX_W], lb6[MAX_W];
    #pragma HLS BIND_STORAGE variable=lb1 type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=lb2 type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=lb3 type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=lb4 type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=lb5 type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=lb6 type=ram_2p impl=bram

    u8 window[7][7];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0

    InitLB:
    for (int x = 0; x < W; ++x) {
    	#pragma HLS PIPELINE II=1
        lb1[x] = 0; lb2[x] = 0; lb3[x] = 0;
        lb4[x] = 0; lb5[x] = 0; lb6[x] = 0;
    }

    InitWin:
    for (int r = 0; r < 7; ++r) {
        for (int c = 0; c < 7; ++c) {
        	#pragma HLS PIPELINE II=1
            window[r][c] = 0;
        }
    }

    Rows:
    for (int y = 0; y < H; ++y) {
    Cols:
        for (int x = 0; x < W; ++x) {
        	#pragma HLS PIPELINE II=1
            u8 pix = s_in.read();

            for (int r = 0; r < 7; ++r) {
            	#pragma HLS UNROLL
                window[r][0] = window[r][1];
                window[r][1] = window[r][2];
                window[r][2] = window[r][3];
                window[r][3] = window[r][4];
                window[r][4] = window[r][5];
                window[r][5] = window[r][6];
            }

            window[0][6] = lb6[x];
            window[1][6] = lb5[x];
            window[2][6] = lb4[x];
            window[3][6] = lb3[x];
            window[4][6] = lb2[x];
            window[5][6] = lb1[x];
            window[6][6] = pix;

            lb6[x] = lb5[x];
            lb5[x] = lb4[x];
            lb4[x] = lb3[x];
            lb3[x] = lb2[x];
            lb2[x] = lb1[x];
            lb1[x] = pix;

            score_t out_score = 0;
            if (y >= 6 && x >= 6) {
                out_score = harris_score_7x7(window);
            }
            s_score.write(out_score);
        }
    }
}

static void duplicate_score_stream(
    hls::stream<score_t>& s_in,
    hls::stream<score_t>& s_out_a,
    hls::stream<score_t>& s_out_b,
    int W,
    int H)
{
	#pragma HLS INLINE off
	Rows:
    for (int y = 0; y < H; ++y) {
    Cols:
        for (int x = 0; x < W; ++x) {
        	#pragma HLS PIPELINE II=1
            score_t v = s_in.read();
            s_out_a.write(v);
            s_out_b.write(v);
        }
    }
}

static void count_harris_candidates(
    hls::stream<score_t>& s_in,
    int* kp_count,
    score_t thresh,
    int W,
    int H)
{
	#pragma HLS INLINE off

    int count = 0;

    Rows:
    for (int y = 0; y < H; ++y) {
    	Cols:
        for (int x = 0; x < W; ++x) {
        	#pragma HLS PIPELINE II=1
            score_t s = s_in.read();
            if (y >= 6 && x >= 6 && s > thresh) {
                count++;
            }
        }
    }

    *kp_count = count;
}
/////
static void process_harris7x7_or_zero(
    hls::stream<u8>& s_in,
    hls::stream<score_t>& s_score,
    int W,
    int H,
    bool level_enabled)
{
#pragma HLS INLINE off

    if (level_enabled) {
        process_harris7x7(s_in, s_score, W, H);
    } else {
        discard_pixel_stream(s_in, W, H);

ZeroScoreRows:
        for (int y = 0; y < H; ++y) {
        ZeroScoreCols:
            for (int x = 0; x < W; ++x) {
#pragma HLS PIPELINE II=1
                s_score.write((score_t)0);
            }
        }
    }
}


// 3) Keypoint stream generation from NMS

static void write_end_of_stream_token(
    hls::stream<kp_pkt_t>& s_kp_out,
    ap_uint<3> scale_id)
{
	#pragma HLS INLINE
    kp_pkt_t eos;
    eos.kp       = 0;
    eos.scale_id = scale_id;
    eos.last     = 1;
    s_kp_out.write(eos);
}

// 5x5 NMS -> internal keypoint stream, Coordinates are rescaled back to the level-0 frame before streaming.
template<int SCALE_NUM, int SCALE_DEN, int SCALE_ID>
static void nms5x5_stream_keypoints_level(
    hls::stream<score_t>& s_in,
    hls::stream<kp_pkt_t>& s_kp_out,
    score_t thresh,
    int W,
    int H,
    int max_kp_local,
    int W_ref,
    int H_ref)
{
	#pragma HLS INLINE off

    score_t lb1[MAX_W], lb2[MAX_W], lb3[MAX_W], lb4[MAX_W];
    #pragma HLS BIND_STORAGE variable=lb1 type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=lb2 type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=lb3 type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=lb4 type=ram_2p impl=bram

    score_t hwin[5][5];
#pragma HLS ARRAY_PARTITION variable=hwin complete dim=0

    const int W_local_limit = W;
    const int H_local_limit = H;

    InitLB2:
    for (int x = 0; x < W; ++x) {
    	#pragma HLS PIPELINE II=1
        lb1[x] = 0; lb2[x] = 0; lb3[x] = 0; lb4[x] = 0;
    }
    InitWin2:
    for (int r = 0; r < 5; ++r) {
        for (int c = 0; c < 5; ++c) {
        	#pragma HLS PIPELINE II=1
            hwin[r][c] = 0;
        }
    }

    int count = 0;

    Rows:
    for (int y = 0; y < H; ++y) {
    	Cols:
        for (int x = 0; x < W; ++x) {
        	#pragma HLS PIPELINE II=1
            score_t h = s_in.read();

            for (int r = 0; r < 5; ++r) {
            	#pragma HLS UNROLL
                hwin[r][0] = hwin[r][1];
                hwin[r][1] = hwin[r][2];
                hwin[r][2] = hwin[r][3];
                hwin[r][3] = hwin[r][4];
            }

            hwin[0][4] = lb4[x];
            hwin[1][4] = lb3[x];
            hwin[2][4] = lb2[x];
            hwin[3][4] = lb1[x];
            hwin[4][4] = h;

            lb4[x] = lb3[x];
            lb3[x] = lb2[x];
            lb2[x] = lb1[x];
            lb1[x] = h;

            if (y >= 8 && x >= 8) {
                score_t center = hwin[2][2];

                if (center > thresh) {
                    bool is_max = true;

                    CheckMax:
                    for (int ry = 0; ry < 5; ++ry) {
                    	#pragma HLS UNROLL
                        for (int rx = 0; rx < 5; ++rx) {
                        	#pragma HLS UNROLL
                            if (ry == 2 && rx == 2) continue;
                            if (hwin[ry][rx] >= center) is_max = false;
                        }
                    }

                    if (is_max && count < max_kp_local) {
                        int ox = x - 5;
                        int oy = y - 5;

                        if (ox >= 0 && ox < W_local_limit &&
                            oy >= 0 && oy < H_local_limit) {

                        	int ox_ref = (int)(((long long)ox * SCALE_NUM + SCALE_DEN / 2) / SCALE_DEN);
                        	int oy_ref = (int)(((long long)oy * SCALE_NUM + SCALE_DEN / 2) / SCALE_DEN);

                        	if (ox_ref < 0) ox_ref = 0;
                        	if (oy_ref < 0) oy_ref = 0;
                        	if (ox_ref >= W_ref) ox_ref = W_ref - 1;
                        	if (oy_ref >= H_ref) oy_ref = H_ref - 1;

                            kp_pkt_t pkt;
                            pkt.kp = 0;
                            pkt.kp.range(15, 0)  = (ap_uint<16>)ox_ref;
                            pkt.kp.range(31, 16) = (ap_uint<16>)oy_ref;
                            pkt.kp.range(63, 32) = (ap_uint<32>)((unsigned int)center);
                            pkt.scale_id = (ap_uint<3>)SCALE_ID;
                            pkt.last = 0;
                            s_kp_out.write(pkt);
                            count++;
                        }
                    }
                }
            }
        }
    }

    write_end_of_stream_token(s_kp_out, (ap_uint<3>)SCALE_ID);
}


template<int SCALE_NUM, int SCALE_DEN, int SCALE_ID>
static void nms5x5_stream_keypoints_or_eos_level(
    hls::stream<score_t>& s_in,
    hls::stream<kp_pkt_t>& s_kp_out,
    score_t thresh,
    int W,
    int H,
    bool level_enabled,
    int max_kp_local,
    int W_ref,
    int H_ref)
{
#pragma HLS INLINE off

	if (level_enabled) {
	    nms5x5_stream_keypoints_level<SCALE_NUM, SCALE_DEN, SCALE_ID>(
	        s_in, s_kp_out,
	        thresh, W, H,
	        max_kp_local,
	        W_ref, H_ref
	    );
    } else {
        discard_score_stream(s_in, W, H);
        write_end_of_stream_token(s_kp_out, (ap_uint<3>)SCALE_ID);
    }
}


// 5) Merge block: eight variable-length keypoint streams -> one DDR buffer.
// Each input stream ends with one explicit EOS packet.
// The writer performs non-blocking round-robin arbitration so that faster
// levels do not permanently block slower levels.
static bool try_read_kp_stream(
    hls::stream<kp_pkt_t>& s,
    bool& done,
    int& count_level,
    int& total,
    int max_kp_merge,
    kp_word_t* nms_kp_out)
{
#pragma HLS INLINE

    kp_pkt_t pkt;

    if (!done && s.read_nb(pkt)) {
        if (pkt.last) {
            done = true;
        } else {
            if (total < max_kp_merge) {
                nms_kp_out[total] = pkt.kp;
                total++;
                count_level++;
            }
        }
        return true;
    }

    return false;
}

static void merge_eight_level_keypoint_streams_to_ddr(
    hls::stream<kp_pkt_t>& s_l0,
    hls::stream<kp_pkt_t>& s_l1,
    hls::stream<kp_pkt_t>& s_l2,
    hls::stream<kp_pkt_t>& s_l3,
    hls::stream<kp_pkt_t>& s_l4,
    hls::stream<kp_pkt_t>& s_l5,
    hls::stream<kp_pkt_t>& s_l6,
    hls::stream<kp_pkt_t>& s_l7,
    kp_word_t* nms_kp_out,
    int* merged_written_nms_count_l0,
    int* merged_written_nms_count_l1,
    int* merged_written_nms_count_l2,
    int* merged_written_nms_count_l3,
    int* merged_written_nms_count_l4,
    int* merged_written_nms_count_l5,
    int* merged_written_nms_count_l6,
    int* merged_written_nms_count_l7,
    int* merged_written_nms_count_total,
    int max_kp_merge)
{
#pragma HLS INLINE off

    bool done0 = false;
    bool done1 = false;
    bool done2 = false;
    bool done3 = false;
    bool done4 = false;
    bool done5 = false;
    bool done6 = false;
    bool done7 = false;

    ap_uint<3> rr = 0;

    int total = 0;
    int c0 = 0;
    int c1 = 0;
    int c2 = 0;
    int c3 = 0;
    int c4 = 0;
    int c5 = 0;
    int c6 = 0;
    int c7 = 0;

MergeLoop8:
    while (!(done0 && done1 && done2 && done3 &&
             done4 && done5 && done6 && done7)) {
#pragma HLS PIPELINE II=1

        bool progress = false;

        switch (rr) {
        case 0:
            progress = try_read_kp_stream(s_l0, done0, c0, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l1, done1, c1, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l2, done2, c2, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l3, done3, c3, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l4, done4, c4, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l5, done5, c5, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l6, done6, c6, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l7, done7, c7, total, max_kp_merge, nms_kp_out);
            break;

        case 1:
            progress = try_read_kp_stream(s_l1, done1, c1, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l2, done2, c2, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l3, done3, c3, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l4, done4, c4, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l5, done5, c5, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l6, done6, c6, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l7, done7, c7, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l0, done0, c0, total, max_kp_merge, nms_kp_out);
            break;

        case 2:
            progress = try_read_kp_stream(s_l2, done2, c2, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l3, done3, c3, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l4, done4, c4, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l5, done5, c5, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l6, done6, c6, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l7, done7, c7, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l0, done0, c0, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l1, done1, c1, total, max_kp_merge, nms_kp_out);
            break;

        case 3:
            progress = try_read_kp_stream(s_l3, done3, c3, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l4, done4, c4, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l5, done5, c5, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l6, done6, c6, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l7, done7, c7, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l0, done0, c0, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l1, done1, c1, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l2, done2, c2, total, max_kp_merge, nms_kp_out);
            break;

        case 4:
            progress = try_read_kp_stream(s_l4, done4, c4, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l5, done5, c5, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l6, done6, c6, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l7, done7, c7, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l0, done0, c0, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l1, done1, c1, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l2, done2, c2, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l3, done3, c3, total, max_kp_merge, nms_kp_out);
            break;

        case 5:
            progress = try_read_kp_stream(s_l5, done5, c5, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l6, done6, c6, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l7, done7, c7, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l0, done0, c0, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l1, done1, c1, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l2, done2, c2, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l3, done3, c3, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l4, done4, c4, total, max_kp_merge, nms_kp_out);
            break;

        case 6:
            progress = try_read_kp_stream(s_l6, done6, c6, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l7, done7, c7, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l0, done0, c0, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l1, done1, c1, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l2, done2, c2, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l3, done3, c3, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l4, done4, c4, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l5, done5, c5, total, max_kp_merge, nms_kp_out);
            break;

        default:
            progress = try_read_kp_stream(s_l7, done7, c7, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l0, done0, c0, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l1, done1, c1, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l2, done2, c2, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l3, done3, c3, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l4, done4, c4, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l5, done5, c5, total, max_kp_merge, nms_kp_out);
            if (!progress) progress = try_read_kp_stream(s_l6, done6, c6, total, max_kp_merge, nms_kp_out);
            break;
        }

        if (progress) {
            rr++;
        }
    }

    *merged_written_nms_count_l0 = c0;
    *merged_written_nms_count_l1 = c1;
    *merged_written_nms_count_l2 = c2;
    *merged_written_nms_count_l3 = c3;
    *merged_written_nms_count_l4 = c4;
    *merged_written_nms_count_l5 = c5;
    *merged_written_nms_count_l6 = c6;
    *merged_written_nms_count_l7 = c7;
    *merged_written_nms_count_total = total;
}

//pack all scalar counters into one metadata buffer: This reduces the number of top-level AXI masters versus one pointer per count.
// stats_out layout:
// stats_out[0..7]   = Harris candidate count per level L0..L7
// stats_out[8..15]  = merged NMS accepted count per level L0..L7
// stats_out[16]     = merged NMS count total
static void write_kernel_stats(
    int* stats_out,
    int harris_candidate_count_l0,
    int harris_candidate_count_l1,
    int harris_candidate_count_l2,
    int harris_candidate_count_l3,
    int harris_candidate_count_l4,
    int harris_candidate_count_l5,
    int harris_candidate_count_l6,
    int harris_candidate_count_l7,
    int merged_written_nms_count_l0,
    int merged_written_nms_count_l1,
    int merged_written_nms_count_l2,
    int merged_written_nms_count_l3,
    int merged_written_nms_count_l4,
    int merged_written_nms_count_l5,
    int merged_written_nms_count_l6,
    int merged_written_nms_count_l7,
    int merged_written_nms_count_total)
{
#pragma HLS INLINE off

WriteStats8:
    for (int i = 0; i < 17; ++i) {
#pragma HLS PIPELINE II=1
        int v = 0;

        if      (i == 0)  v = harris_candidate_count_l0;
        else if (i == 1)  v = harris_candidate_count_l1;
        else if (i == 2)  v = harris_candidate_count_l2;
        else if (i == 3)  v = harris_candidate_count_l3;
        else if (i == 4)  v = harris_candidate_count_l4;
        else if (i == 5)  v = harris_candidate_count_l5;
        else if (i == 6)  v = harris_candidate_count_l6;
        else if (i == 7)  v = harris_candidate_count_l7;

        else if (i == 8)  v = merged_written_nms_count_l0;
        else if (i == 9)  v = merged_written_nms_count_l1;
        else if (i == 10) v = merged_written_nms_count_l2;
        else if (i == 11) v = merged_written_nms_count_l3;
        else if (i == 12) v = merged_written_nms_count_l4;
        else if (i == 13) v = merged_written_nms_count_l5;
        else if (i == 14) v = merged_written_nms_count_l6;
        else if (i == 15) v = merged_written_nms_count_l7;

        else              v = merged_written_nms_count_total;

        stats_out[i] = v;
    }
}

static void multiscale_pipeline_wrapper(
    const u8*    src,
    kp_word_t*   nms_kp_out,
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
    int          max_kp_l7,
    int*         harris_candidate_count_l0,
    int*         harris_candidate_count_l1,
    int*         harris_candidate_count_l2,
    int*         harris_candidate_count_l3,
    int*         harris_candidate_count_l4,
    int*         harris_candidate_count_l5,
    int*         harris_candidate_count_l6,
    int*         harris_candidate_count_l7,
    int*         merged_written_nms_count_l0,
    int*         merged_written_nms_count_l1,
    int*         merged_written_nms_count_l2,
    int*         merged_written_nms_count_l3,
    int*         merged_written_nms_count_l4,
    int*         merged_written_nms_count_l5,
    int*         merged_written_nms_count_l6,
    int*         merged_written_nms_count_l7,
    int*         merged_written_nms_count_total)
{
#pragma HLS INLINE off

    // Pyramid dimensions for scale factor 1.2.
    // Downscale ratio between consecutive levels is 1/1.2 = 5/6.
    const int W0 = W;
    const int H0 = H;

    const int W1 = (W0 * 5) / 6;
    const int H1 = (H0 * 5) / 6;

    const int W2 = (W1 * 5) / 6;
    const int H2 = (H1 * 5) / 6;

    const int W3 = (W2 * 5) / 6;
    const int H3 = (H2 * 5) / 6;

    const int W4 = (W3 * 5) / 6;
    const int H4 = (H3 * 5) / 6;

    const int W5 = (W4 * 5) / 6;
    const int H5 = (H4 * 5) / 6;

    const int W6 = (W5 * 5) / 6;
    const int H6 = (H5 * 5) / 6;

    const int W7 = (W6 * 5) / 6;
    const int H7 = (H6 * 5) / 6;

    const bool enable_l0 = (active_levels >= 1) && (W0 > 8) && (H0 > 8);
    const bool enable_l1 = (active_levels >= 2) && (W1 > 8) && (H1 > 8);
    const bool enable_l2 = (active_levels >= 3) && (W2 > 8) && (H2 > 8);
    const bool enable_l3 = (active_levels >= 4) && (W3 > 8) && (H3 > 8);
    const bool enable_l4 = (active_levels >= 5) && (W4 > 8) && (H4 > 8);
    const bool enable_l5 = (active_levels >= 6) && (W5 > 8) && (H5 > 8);
    const bool enable_l6 = (active_levels >= 7) && (W6 > 8) && (H6 > 8);
    const bool enable_l7 = (active_levels >= 8) && (W7 > 8) && (H7 > 8);

    // Pixel streams for pyramid generation.
    hls::stream<u8> s_l0_raw;
    hls::stream<u8> s_l0_for_branch;
    hls::stream<u8> s_l0_for_downsample;

    hls::stream<u8> s_l1_raw;
    hls::stream<u8> s_l1_for_branch;
    hls::stream<u8> s_l1_for_downsample;

    hls::stream<u8> s_l2_raw;
    hls::stream<u8> s_l2_for_branch;
    hls::stream<u8> s_l2_for_downsample;

    hls::stream<u8> s_l3_raw;
    hls::stream<u8> s_l3_for_branch;
    hls::stream<u8> s_l3_for_downsample;

    hls::stream<u8> s_l4_raw;
    hls::stream<u8> s_l4_for_branch;
    hls::stream<u8> s_l4_for_downsample;

    hls::stream<u8> s_l5_raw;
    hls::stream<u8> s_l5_for_branch;
    hls::stream<u8> s_l5_for_downsample;

    hls::stream<u8> s_l6_raw;
    hls::stream<u8> s_l6_for_branch;
    hls::stream<u8> s_l6_for_downsample;

    hls::stream<u8> s_l7_for_branch;

    // Keypoint streams.
    hls::stream<kp_pkt_t> s_kp_l0;
    hls::stream<kp_pkt_t> s_kp_l1;
    hls::stream<kp_pkt_t> s_kp_l2;
    hls::stream<kp_pkt_t> s_kp_l3;
    hls::stream<kp_pkt_t> s_kp_l4;
    hls::stream<kp_pkt_t> s_kp_l5;
    hls::stream<kp_pkt_t> s_kp_l6;
    hls::stream<kp_pkt_t> s_kp_l7;

    // Score streams for each level.
    hls::stream<score_t> s_score_l0;
    hls::stream<score_t> s_score_l0_for_harris;
    hls::stream<score_t> s_score_l0_for_nms;

    hls::stream<score_t> s_score_l1;
    hls::stream<score_t> s_score_l1_for_harris;
    hls::stream<score_t> s_score_l1_for_nms;

    hls::stream<score_t> s_score_l2;
    hls::stream<score_t> s_score_l2_for_harris;
    hls::stream<score_t> s_score_l2_for_nms;

    hls::stream<score_t> s_score_l3;
    hls::stream<score_t> s_score_l3_for_harris;
    hls::stream<score_t> s_score_l3_for_nms;

    hls::stream<score_t> s_score_l4;
    hls::stream<score_t> s_score_l4_for_harris;
    hls::stream<score_t> s_score_l4_for_nms;

    hls::stream<score_t> s_score_l5;
    hls::stream<score_t> s_score_l5_for_harris;
    hls::stream<score_t> s_score_l5_for_nms;

    hls::stream<score_t> s_score_l6;
    hls::stream<score_t> s_score_l6_for_harris;
    hls::stream<score_t> s_score_l6_for_nms;

    hls::stream<score_t> s_score_l7;
    hls::stream<score_t> s_score_l7_for_harris;
    hls::stream<score_t> s_score_l7_for_nms;

#pragma HLS STREAM variable=s_l0_raw            depth=1024
#pragma HLS STREAM variable=s_l0_for_branch     depth=1024
#pragma HLS STREAM variable=s_l0_for_downsample depth=1024

#pragma HLS STREAM variable=s_l1_raw            depth=1024
#pragma HLS STREAM variable=s_l1_for_branch     depth=1024
#pragma HLS STREAM variable=s_l1_for_downsample depth=1024

#pragma HLS STREAM variable=s_l2_raw            depth=1024
#pragma HLS STREAM variable=s_l2_for_branch     depth=1024
#pragma HLS STREAM variable=s_l2_for_downsample depth=1024

#pragma HLS STREAM variable=s_l3_raw            depth=1024
#pragma HLS STREAM variable=s_l3_for_branch     depth=1024
#pragma HLS STREAM variable=s_l3_for_downsample depth=1024

#pragma HLS STREAM variable=s_l4_raw            depth=1024
#pragma HLS STREAM variable=s_l4_for_branch     depth=1024
#pragma HLS STREAM variable=s_l4_for_downsample depth=1024

#pragma HLS STREAM variable=s_l5_raw            depth=1024
#pragma HLS STREAM variable=s_l5_for_branch     depth=1024
#pragma HLS STREAM variable=s_l5_for_downsample depth=1024

#pragma HLS STREAM variable=s_l6_raw            depth=1024
#pragma HLS STREAM variable=s_l6_for_branch     depth=1024
#pragma HLS STREAM variable=s_l6_for_downsample depth=1024

#pragma HLS STREAM variable=s_l7_for_branch     depth=1024

#pragma HLS STREAM variable=s_kp_l0 depth=1024
#pragma HLS STREAM variable=s_kp_l1 depth=1024
#pragma HLS STREAM variable=s_kp_l2 depth=1024
#pragma HLS STREAM variable=s_kp_l3 depth=1024
#pragma HLS STREAM variable=s_kp_l4 depth=1024
#pragma HLS STREAM variable=s_kp_l5 depth=1024
#pragma HLS STREAM variable=s_kp_l6 depth=1024
#pragma HLS STREAM variable=s_kp_l7 depth=1024

#pragma HLS STREAM variable=s_score_l0            depth=1024
#pragma HLS STREAM variable=s_score_l0_for_harris depth=1024
#pragma HLS STREAM variable=s_score_l0_for_nms    depth=1024

#pragma HLS STREAM variable=s_score_l1            depth=1024
#pragma HLS STREAM variable=s_score_l1_for_harris depth=1024
#pragma HLS STREAM variable=s_score_l1_for_nms    depth=1024

#pragma HLS STREAM variable=s_score_l2            depth=1024
#pragma HLS STREAM variable=s_score_l2_for_harris depth=1024
#pragma HLS STREAM variable=s_score_l2_for_nms    depth=1024

#pragma HLS STREAM variable=s_score_l3            depth=1024
#pragma HLS STREAM variable=s_score_l3_for_harris depth=1024
#pragma HLS STREAM variable=s_score_l3_for_nms    depth=1024

#pragma HLS STREAM variable=s_score_l4            depth=1024
#pragma HLS STREAM variable=s_score_l4_for_harris depth=1024
#pragma HLS STREAM variable=s_score_l4_for_nms    depth=1024

#pragma HLS STREAM variable=s_score_l5            depth=1024
#pragma HLS STREAM variable=s_score_l5_for_harris depth=1024
#pragma HLS STREAM variable=s_score_l5_for_nms    depth=1024

#pragma HLS STREAM variable=s_score_l6            depth=1024
#pragma HLS STREAM variable=s_score_l6_for_harris depth=1024
#pragma HLS STREAM variable=s_score_l6_for_nms    depth=1024

#pragma HLS STREAM variable=s_score_l7            depth=1024
#pragma HLS STREAM variable=s_score_l7_for_harris depth=1024
#pragma HLS STREAM variable=s_score_l7_for_nms    depth=1024

#pragma HLS DATAFLOW

    // Pyramid generation chain.
    read_image_to_stream(src, s_l0_raw, W0, H0);
    duplicate_pixel_stream(s_l0_raw, s_l0_for_branch, s_l0_for_downsample, W0, H0);

    gaussian5x5_downsample_5over6(s_l0_for_downsample, s_l1_raw, W0, H0);
    duplicate_pixel_stream(s_l1_raw, s_l1_for_branch, s_l1_for_downsample, W1, H1);

    gaussian5x5_downsample_5over6(s_l1_for_downsample, s_l2_raw, W1, H1);
    duplicate_pixel_stream(s_l2_raw, s_l2_for_branch, s_l2_for_downsample, W2, H2);

    gaussian5x5_downsample_5over6(s_l2_for_downsample, s_l3_raw, W2, H2);
    duplicate_pixel_stream(s_l3_raw, s_l3_for_branch, s_l3_for_downsample, W3, H3);

    gaussian5x5_downsample_5over6(s_l3_for_downsample, s_l4_raw, W3, H3);
    duplicate_pixel_stream(s_l4_raw, s_l4_for_branch, s_l4_for_downsample, W4, H4);

    gaussian5x5_downsample_5over6(s_l4_for_downsample, s_l5_raw, W4, H4);
    duplicate_pixel_stream(s_l5_raw, s_l5_for_branch, s_l5_for_downsample, W5, H5);

    gaussian5x5_downsample_5over6(s_l5_for_downsample, s_l6_raw, W5, H5);
    duplicate_pixel_stream(s_l6_raw, s_l6_for_branch, s_l6_for_downsample, W6, H6);

    gaussian5x5_downsample_5over6(s_l6_for_downsample, s_l7_for_branch, W6, H6);

    // Level 0
    process_harris7x7_or_zero(s_l0_for_branch, s_score_l0, W0, H0, enable_l0);
    duplicate_score_stream(s_score_l0, s_score_l0_for_harris, s_score_l0_for_nms, W0, H0);
    count_harris_candidates(s_score_l0_for_harris, harris_candidate_count_l0, thresh, W0, H0);
    nms5x5_stream_keypoints_or_eos_level<1,1,0>(
        s_score_l0_for_nms, s_kp_l0,
        thresh, W0, H0,
        enable_l0, max_kp_l0,
        W0, H0
    );

    // Level 1
    process_harris7x7_or_zero(s_l1_for_branch, s_score_l1, W1, H1, enable_l1);
    duplicate_score_stream(s_score_l1, s_score_l1_for_harris, s_score_l1_for_nms, W1, H1);
    count_harris_candidates(s_score_l1_for_harris, harris_candidate_count_l1, thresh, W1, H1);
    nms5x5_stream_keypoints_or_eos_level<6,5,1>(
        s_score_l1_for_nms, s_kp_l1,
        thresh, W1, H1,
        enable_l1, max_kp_l1,
        W0, H0
    );

    // Level 2
    process_harris7x7_or_zero(s_l2_for_branch, s_score_l2, W2, H2, enable_l2);
    duplicate_score_stream(s_score_l2, s_score_l2_for_harris, s_score_l2_for_nms, W2, H2);
    count_harris_candidates(s_score_l2_for_harris, harris_candidate_count_l2, thresh, W2, H2);
    nms5x5_stream_keypoints_or_eos_level<36,25,2>(
        s_score_l2_for_nms, s_kp_l2,
        thresh, W2, H2,
        enable_l2, max_kp_l2,
        W0, H0
    );

    // Level 3
    process_harris7x7_or_zero(s_l3_for_branch, s_score_l3, W3, H3, enable_l3);
    duplicate_score_stream(s_score_l3, s_score_l3_for_harris, s_score_l3_for_nms, W3, H3);
    count_harris_candidates(s_score_l3_for_harris, harris_candidate_count_l3, thresh, W3, H3);
    nms5x5_stream_keypoints_or_eos_level<216,125,3>(
        s_score_l3_for_nms, s_kp_l3,
        thresh, W3, H3,
        enable_l3, max_kp_l3,
        W0, H0
    );

    // Level 4
    process_harris7x7_or_zero(s_l4_for_branch, s_score_l4, W4, H4, enable_l4);
    duplicate_score_stream(s_score_l4, s_score_l4_for_harris, s_score_l4_for_nms, W4, H4);
    count_harris_candidates(s_score_l4_for_harris, harris_candidate_count_l4, thresh, W4, H4);
    nms5x5_stream_keypoints_or_eos_level<1296,625,4>(
        s_score_l4_for_nms, s_kp_l4,
        thresh, W4, H4,
        enable_l4, max_kp_l4,
        W0, H0
    );

    // Level 5
    process_harris7x7_or_zero(s_l5_for_branch, s_score_l5, W5, H5, enable_l5);
    duplicate_score_stream(s_score_l5, s_score_l5_for_harris, s_score_l5_for_nms, W5, H5);
    count_harris_candidates(s_score_l5_for_harris, harris_candidate_count_l5, thresh, W5, H5);
    nms5x5_stream_keypoints_or_eos_level<7776,3125,5>(
        s_score_l5_for_nms, s_kp_l5,
        thresh, W5, H5,
        enable_l5, max_kp_l5,
        W0, H0
    );

    // Level 6
    process_harris7x7_or_zero(s_l6_for_branch, s_score_l6, W6, H6, enable_l6);
    duplicate_score_stream(s_score_l6, s_score_l6_for_harris, s_score_l6_for_nms, W6, H6);
    count_harris_candidates(s_score_l6_for_harris, harris_candidate_count_l6, thresh, W6, H6);
    nms5x5_stream_keypoints_or_eos_level<46656,15625,6>(
        s_score_l6_for_nms, s_kp_l6,
        thresh, W6, H6,
        enable_l6, max_kp_l6,
        W0, H0
    );

    // Level 7
    process_harris7x7_or_zero(s_l7_for_branch, s_score_l7, W7, H7, enable_l7);
    duplicate_score_stream(s_score_l7, s_score_l7_for_harris, s_score_l7_for_nms, W7, H7);
    count_harris_candidates(s_score_l7_for_harris, harris_candidate_count_l7, thresh, W7, H7);
    nms5x5_stream_keypoints_or_eos_level<279936,78125,7>(
        s_score_l7_for_nms, s_kp_l7,
        thresh, W7, H7,
        enable_l7, max_kp_l7,
        W0, H0
    );

    merge_eight_level_keypoint_streams_to_ddr(
        s_kp_l0, s_kp_l1, s_kp_l2, s_kp_l3,
        s_kp_l4, s_kp_l5, s_kp_l6, s_kp_l7,
        nms_kp_out,
        merged_written_nms_count_l0,
        merged_written_nms_count_l1,
        merged_written_nms_count_l2,
        merged_written_nms_count_l3,
        merged_written_nms_count_l4,
        merged_written_nms_count_l5,
        merged_written_nms_count_l6,
        merged_written_nms_count_l7,
        merged_written_nms_count_total,
        max_kp_merge
    );
}

// Top kernel: src is the level-0 image only.
// Levels 1..7 are generated on device using a 5/6 downscale ratio,
// scale factor of 1.2 between levels.

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
    int          max_kp_l7)
{
#pragma HLS INTERFACE m_axi     port=src                offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi     port=nms_kp_out         offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=nms_kp_count_total offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi     port=stats_out          offset=slave bundle=gmem3

#pragma HLS INTERFACE s_axilite port=src                bundle=control
#pragma HLS INTERFACE s_axilite port=nms_kp_out         bundle=control
#pragma HLS INTERFACE s_axilite port=nms_kp_count_total bundle=control
#pragma HLS INTERFACE s_axilite port=stats_out          bundle=control
#pragma HLS INTERFACE s_axilite port=thresh             bundle=control
#pragma HLS INTERFACE s_axilite port=W                  bundle=control
#pragma HLS INTERFACE s_axilite port=H                  bundle=control
#pragma HLS INTERFACE s_axilite port=active_levels      bundle=control
#pragma HLS INTERFACE s_axilite port=max_kp_merge       bundle=control
#pragma HLS INTERFACE s_axilite port=max_kp_l0          bundle=control
#pragma HLS INTERFACE s_axilite port=max_kp_l1          bundle=control
#pragma HLS INTERFACE s_axilite port=max_kp_l2          bundle=control
#pragma HLS INTERFACE s_axilite port=max_kp_l3          bundle=control
#pragma HLS INTERFACE s_axilite port=max_kp_l4          bundle=control
#pragma HLS INTERFACE s_axilite port=max_kp_l5          bundle=control
#pragma HLS INTERFACE s_axilite port=max_kp_l6          bundle=control
#pragma HLS INTERFACE s_axilite port=max_kp_l7          bundle=control
#pragma HLS INTERFACE s_axilite port=return             bundle=control

    if (active_levels < 1) active_levels = 1;
    if (active_levels > 8) active_levels = 8;

    if (W <= 0 || H <= 0 || W > MAX_W || max_kp_merge <= 0) {
        *nms_kp_count_total = 0;

        write_kernel_stats(
            stats_out,
            0, 0, 0, 0, 0, 0, 0, 0,   // Harris counts L0..L7
            0, 0, 0, 0, 0, 0, 0, 0,   // NMS merged counts L0..L7
            0                         // total
        );

        return;
    }

    int harris_candidate_count_l0 = 0;
    int harris_candidate_count_l1 = 0;
    int harris_candidate_count_l2 = 0;
    int harris_candidate_count_l3 = 0;
    int harris_candidate_count_l4 = 0;
    int harris_candidate_count_l5 = 0;
    int harris_candidate_count_l6 = 0;
    int harris_candidate_count_l7 = 0;

    int merged_written_nms_count_l0 = 0;
    int merged_written_nms_count_l1 = 0;
    int merged_written_nms_count_l2 = 0;
    int merged_written_nms_count_l3 = 0;
    int merged_written_nms_count_l4 = 0;
    int merged_written_nms_count_l5 = 0;
    int merged_written_nms_count_l6 = 0;
    int merged_written_nms_count_l7 = 0;
    int merged_written_nms_count_total = 0;

    multiscale_pipeline_wrapper(
        src,
        nms_kp_out,
        thresh,
        W, H,
        active_levels,
        max_kp_merge,
        max_kp_l0,
        max_kp_l1,
        max_kp_l2,
        max_kp_l3,
        max_kp_l4,
        max_kp_l5,
        max_kp_l6,
        max_kp_l7,
        &harris_candidate_count_l0,
        &harris_candidate_count_l1,
        &harris_candidate_count_l2,
        &harris_candidate_count_l3,
        &harris_candidate_count_l4,
        &harris_candidate_count_l5,
        &harris_candidate_count_l6,
        &harris_candidate_count_l7,
        &merged_written_nms_count_l0,
        &merged_written_nms_count_l1,
        &merged_written_nms_count_l2,
        &merged_written_nms_count_l3,
        &merged_written_nms_count_l4,
        &merged_written_nms_count_l5,
        &merged_written_nms_count_l6,
        &merged_written_nms_count_l7,
        &merged_written_nms_count_total
    );

    *nms_kp_count_total = merged_written_nms_count_total;

    write_kernel_stats(
        stats_out,
        harris_candidate_count_l0,
        harris_candidate_count_l1,
        harris_candidate_count_l2,
        harris_candidate_count_l3,
        harris_candidate_count_l4,
        harris_candidate_count_l5,
        harris_candidate_count_l6,
        harris_candidate_count_l7,
        merged_written_nms_count_l0,
        merged_written_nms_count_l1,
        merged_written_nms_count_l2,
        merged_written_nms_count_l3,
        merged_written_nms_count_l4,
        merged_written_nms_count_l5,
        merged_written_nms_count_l6,
        merged_written_nms_count_l7,
        merged_written_nms_count_total
    );
}

