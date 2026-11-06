#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

static const int MAX_W = 1000;
static const int MAX_H = 1000;
static const int LEVELS = 8;
static const int K_TOP = 2048;
static const int N_FINAL = 1000;
static const uint16_t C_Q12 = (uint16_t)(0.90f * 4096.0f + 0.5f);

struct Image {
    int w = 0;
    int h = 0;
    vector<uint8_t> pix;
};

struct KP {
    uint16_t x;
    uint16_t y;
    uint32_t s;
};

static inline uint8_t get_px(const Image& img, int x, int y) {
    if (x < 0 || y < 0 || x >= img.w || y >= img.h) return 0;
    return img.pix[y * img.w + x];
}

static bool read_image_auto(const string& base_no_ext, Image& img) {
    string names[2] = {base_no_ext + ".ppm", base_no_ext + ".pgm"};

    for (const string& filename : names) {
        ifstream f(filename, ios::binary);
        if (!f) continue;

        string magic;
        f >> magic;
        if (magic != "P6" && magic != "P5") continue;

        while (f >> ws && f.peek() == '#') {
            string header_line;
            getline(f, header_line);
        }

        int maxval;
        f >> img.w >> img.h >> maxval;
        f.ignore(1);

        if (img.w <= 0 || img.h <= 0 || img.w > MAX_W || img.h > MAX_H) {
            cerr << "[ERR] Invalid image size: " << filename << "\n";
            return false;
        }

        img.pix.assign(img.w * img.h, 0);

        if (magic == "P5") {
            for (int i = 0; i < img.w * img.h; ++i) {
                char v;
                f.read(&v, 1);
                img.pix[i] = (uint8_t)v;
            }
        } else {
            for (int i = 0; i < img.w * img.h; ++i) {
                char r, g, b;
                f.read(&r, 1);
                f.read(&g, 1);
                f.read(&b, 1);
                uint8_t ur = (uint8_t)r;
                uint8_t ug = (uint8_t)g;
                uint8_t ub = (uint8_t)b;

                // Same style as host grayscale conversion: integer output.
                img.pix[i] = (uint8_t)(0.299 * ur + 0.587 * ug + 0.114 * ub);
            }
        }

        cout << "[READ] " << filename << " " << img.w << "x" << img.h << "\n";
        return true;
    }

    return false;
}

// HW-equivalent 5x5 Gaussian + 5/6 downsample.
// HW uses Q8 weights {14,63,102,63,14} and (sum+32768)>>16.
static Image gaussian5x5_downsample_5over6_cpu(const Image& in) {
    static const uint32_t g[5] = {14, 63, 102, 63, 14};

    Image out;
    out.w = (in.w * 5) / 6;
    out.h = (in.h * 5) / 6;
    out.pix.assign(out.w * out.h, 0);

    int out_y = 0;
    int y_acc = 0;

    for (int y = 0; y < in.h; ++y) {
        y_acc += 5;
        bool keep_y = false;
        if (y_acc >= 6) {
            keep_y = true;
            y_acc -= 6;
        }

        int out_x = 0;
        int x_acc = 0;

        for (int x = 0; x < in.w; ++x) {
            x_acc += 5;
            bool keep_x = false;
            if (x_acc >= 6) {
                keep_x = true;
                x_acc -= 6;
            }

            if (keep_x && keep_y) {
                uint32_t sum = 0;

                for (int ry = 0; ry < 5; ++ry) {
                    for (int rx = 0; rx < 5; ++rx) {
                        int yy = y + ry - 4;
                        int xx = x + rx - 4;
                        uint8_t p = get_px(in, xx, yy);
                        sum += (uint32_t)p * g[ry] * g[rx];
                    }
                }

                uint8_t outp = (uint8_t)((sum + 32768u) >> 16);

                if (out_y < out.h && out_x < out.w) {
                    out.pix[out_y * out.w + out_x] = outp;
                }
                out_x++;
            }
        }

        if (keep_y) out_y++;
    }

    return out;
}

static vector<Image> build_pyramid_fixed(const Image& base) {
    vector<Image> pyr;
    pyr.reserve(LEVELS);
    pyr.push_back(base);

    for (int l = 1; l < LEVELS; ++l) {
        pyr.push_back(gaussian5x5_downsample_5over6_cpu(pyr[l - 1]));
    }

    return pyr;
}

// HW-equivalent integer Harris score.
// Same logic as harris_score_7x7 in the FPGA kernel:
// g1d={1,6,15,20,15,6,1}, gx = right-left, gy = down-up,
// R = det - (trace^2 >> 5), then R >>= 20, clamp to uint31.
static uint32_t harris_score_7x7_cpu(const Image& img, int x, int y) {
    static const int g1d[7] = {1, 6, 15, 20, 15, 6, 1};

    long long rowIxx[5];
    long long rowIyy[5];
    long long rowIxy[5];

    for (int wy = 1; wy <= 5; ++wy) {
        long long acc_xx = 0;
        long long acc_yy = 0;
        long long acc_xy = 0;

        for (int wx = 1; wx <= 5; ++wx) {
            int px_r = get_px(img, x + wx + 1 - 6, y + wy - 6);
            int px_l = get_px(img, x + wx - 1 - 6, y + wy - 6);
            int px_d = get_px(img, x + wx - 6, y + wy + 1 - 6);
            int px_u = get_px(img, x + wx - 6, y + wy - 1 - 6);

            int gx = px_r - px_l;
            int gy = px_d - px_u;

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

    long long Ixx = rowIxx[0] + rowIxx[1] + rowIxx[2] + rowIxx[3] + rowIxx[4];
    long long Iyy = rowIyy[0] + rowIyy[1] + rowIyy[2] + rowIyy[3] + rowIyy[4];
    long long Ixy = rowIxy[0] + rowIxy[1] + rowIxy[2] + rowIxy[3] + rowIxy[4];

    long long detM = Ixx * Iyy - Ixy * Ixy;
    long long trace = Ixx + Iyy;

    long long R = detM - ((trace * trace) >> 5);
    R >>= 20;

    if (R < 0) R = 0;
    if (R > 0x7FFFFFFFLL) R = 0x7FFFFFFFLL;

    return (uint32_t)R;
}

static vector<uint32_t> compute_harris_map_fixed(const Image& img) {
    vector<uint32_t> H(img.w * img.h, 0);

    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            if (y >= 6 && x >= 6) {
                H[y * img.w + x] = harris_score_7x7_cpu(img, x, y);
            }
        }
    }

    return H;
}

static vector<KP> nms5x5_level_fixed(
    const Image& img,
    const vector<uint32_t>& H,
    uint32_t thresh,
    int scale_num,
    int scale_den,
    int W_ref,
    int H_ref,
    int max_kp_local)
{
    vector<KP> out;
    out.reserve(max_kp_local);

    int count = 0;

    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {

            if (y < 8 || x < 8) continue;

            // HW center is hwin[2][2] when current stream position is (x,y).
            // The emitted coordinate is ox=x-5, oy=y-5.
            int cx = x - 2;
            int cy = y - 2;

            if (cx < 0 || cy < 0 || cx >= img.w || cy >= img.h) continue;

            uint32_t center = H[cy * img.w + cx];

            if (center <= thresh) continue;

            bool is_max = true;

            for (int ry = -2; ry <= 2; ++ry) {
                for (int rx = -2; rx <= 2; ++rx) {
                    if (ry == 0 && rx == 0) continue;

                    int xx = cx + rx;
                    int yy = cy + ry;

                    uint32_t v = 0;
                    if (xx >= 0 && yy >= 0 && xx < img.w && yy < img.h) {
                        v = H[yy * img.w + xx];
                    }

                    // HW rejects if neighbor >= center.
                    if (v >= center) {
                        is_max = false;
                    }
                }
            }

            if (!is_max) continue;
            if (count >= max_kp_local) continue;

            int ox = x - 5;
            int oy = y - 5;

            if (ox < 0 || oy < 0 || ox >= img.w || oy >= img.h) continue;

            int ox_ref = (int)(((long long)ox * scale_num + scale_den / 2) / scale_den);
            int oy_ref = (int)(((long long)oy * scale_num + scale_den / 2) / scale_den);

            if (ox_ref < 0) ox_ref = 0;
            if (oy_ref < 0) oy_ref = 0;
            if (ox_ref >= W_ref) ox_ref = W_ref - 1;
            if (oy_ref >= H_ref) oy_ref = H_ref - 1;

            KP kp;
            kp.x = (uint16_t)ox_ref;
            kp.y = (uint16_t)oy_ref;
            kp.s = center;
            out.push_back(kp);

            count++;
        }
    }

    return out;
}

// Collect thresholded Harris candidates before the 5x5 local-NMS test.
// Coordinates are mapped to the common Level-0 image frame using
// the same coordinate convention as the FPGA NMS output.
static vector<KP> collect_pre_nms_level_fixed(
    const Image& img,
    const vector<uint32_t>& H,
    uint32_t thresh,
    int scale_num,
    int scale_den,
    int W_ref,
    int H_ref)
{
    vector<KP> out;

    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {

            // Same valid stream positions examined by the NMS stage.
            if (y < 8 || x < 8) continue;

            int cx = x - 2;
            int cy = y - 2;

            if (cx < 0 || cy < 0 ||
                cx >= img.w || cy >= img.h)
                continue;

            uint32_t center = H[cy * img.w + cx];

            // Candidate entering NMS: threshold passed,
            // but no local-maximum test has yet been applied.
            if (center <= thresh) continue;

            // Spatial coordinate associated with this Harris response.
            int ox = x - 5;
            int oy = y - 5;

            if (ox < 0 || oy < 0 ||
                ox >= img.w || oy >= img.h)
                continue;

            // Same Level-0 coordinate rescaling used by the NMS output.
            int ox_ref =
                (int)(((long long)ox * scale_num +
                       scale_den / 2) / scale_den);

            int oy_ref =
                (int)(((long long)oy * scale_num +
                       scale_den / 2) / scale_den);

            if (ox_ref < 0) ox_ref = 0;
            if (oy_ref < 0) oy_ref = 0;
            if (ox_ref >= W_ref) ox_ref = W_ref - 1;
            if (oy_ref >= H_ref) oy_ref = H_ref - 1;

            KP kp;
            kp.x = (uint16_t)ox_ref;
            kp.y = (uint16_t)oy_ref;
            kp.s = center;

            out.push_back(kp);
        }
    }

    return out;
}

static bool score_desc(const KP& a, const KP& b) {
    return a.s > b.s;
}

// Same idea as HW Top-K: keep strongest K by uint32 score.
// Tie behavior is not critical for metric evaluation, but we keep stable order when equal.
static vector<KP> select_topK_cpu(const vector<KP>& in, int K_top) {
    vector<KP> v = in;

    stable_sort(v.begin(), v.end(), [](const KP& a, const KP& b) {
        if (a.s != b.s) return a.s > b.s;
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
    });

    if ((int)v.size() > K_top) v.resize(K_top);
    return v;
}

// HW-equivalent ANMS:
// 1) sort by score descending
// 2) stronger test: ss[j] << 12 > c_q12 * ss[i]
// 3) keep N largest radius^2
static vector<KP> anms_fixed_q12_cpu(vector<KP> in, int K_top, int N_final) {
    vector<KP> cand = select_topK_cpu(in, K_top);

    stable_sort(cand.begin(), cand.end(), [](const KP& a, const KP& b) {
        if (a.s != b.s) return a.s > b.s;
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
    });

    int K = (int)cand.size();
    int N = std::min(K, N_final);

    if (K == 0) return {};

    const uint32_t INF = 0xFFFFFFFFu;
    vector<uint32_t> r2(K, INF);

    for (int i = 0; i < K; ++i) {
        uint32_t best = INF;
        uint64_t rhs = (uint64_t)C_Q12 * (uint64_t)cand[i].s;

        for (int j = 0; j < i; ++j) {
            uint64_t lhs = ((uint64_t)cand[j].s) << 12;

            if (lhs > rhs) {
                int dx = (int)cand[i].x - (int)cand[j].x;
                int dy = (int)cand[i].y - (int)cand[j].y;
                uint32_t d2 = (uint32_t)(dx * dx + dy * dy);

                if (d2 < best) best = d2;
            }
        }

        r2[i] = best;
    }

    vector<KP> out;
    vector<uint32_t> out_r2;

    out.reserve(N);
    out_r2.reserve(N);

    int filled = N;

    for (int i = 0; i < filled; ++i) {
        out.push_back(cand[i]);
        out_r2.push_back(r2[i]);
    }

    if (K > N) {
        int min_i = 0;
        uint32_t minv = out_r2[0];

        for (int t = 1; t < N; ++t) {
            if (out_r2[t] < minv) {
                minv = out_r2[t];
                min_i = t;
            }
        }

        for (int i = N; i < K; ++i) {
            uint32_t val = r2[i];

            if (val > minv) {
                out[min_i] = cand[i];
                out_r2[min_i] = val;

                min_i = 0;
                minv = out_r2[0];

                for (int t = 1; t < N; ++t) {
                    if (out_r2[t] < minv) {
                        minv = out_r2[t];
                        min_i = t;
                    }
                }
            }
        }
    }

    return out;
}

static void write_kps_txt(const string& path, const vector<KP>& kps) {
    ofstream ofs(path);
    if (!ofs) {
        cerr << "[ERR] Cannot write: " << path << "\n";
        return;
    }

    for (const auto& kp : kps) {
        ofs << kp.x << " " << kp.y << " " << kp.s << "\n";
    }
}

static void ensure_dir(const string& path) {
    mkdir(path.c_str(), 0755);
}

static vector<KP> run_fixed_detector_one_image( const Image& img, vector<KP>* pre_nms_out, vector<KP>* local_nms_out)
{
    vector<Image> pyr = build_pyramid_fixed(img);

    // Exact scale factors used by HW for mapping each level back to level 0.
    const int scale_num[LEVELS] = {
        1, 6, 36, 216, 1296, 7776, 46656, 279936
    };
    const int scale_den[LEVELS] = {
        1, 5, 25, 125, 625, 3125, 15625, 78125
    };

    vector<KP> pre_nms_merged;
    pre_nms_merged.reserve(100000);

    vector<KP> merged;
    merged.reserve(24000);

    for (int l = 0; l < LEVELS; ++l) {
        if (pyr[l].w <= 8 || pyr[l].h <= 8) continue;

        vector<uint32_t> H = compute_harris_map_fixed(pyr[l]);

        vector<KP> lvl_pre_nms = collect_pre_nms_level_fixed(
            pyr[l],
            H,
            100,                 // same threshold as FPGA run
            scale_num[l],
            scale_den[l],
            img.w,
            img.h
        );
        
        pre_nms_merged.insert(
            pre_nms_merged.end(),
            lvl_pre_nms.begin(),
            lvl_pre_nms.end()
        );

        vector<KP> lvl_kps = nms5x5_level_fixed(
            pyr[l],
            H,
            100,                 // same threshold as HW run
            scale_num[l],
            scale_den[l],
            img.w,
            img.h,
            12000                // local safety cap
        );

        cout << "  [L" << l << "] "
             << pyr[l].w << "x" << pyr[l].h
             << " nms=" << lvl_kps.size() << "\n";

        // For Table 5 quality, append all levels.
        // The HW merge is round-robin, but ANMS first selects Top-K by score,
        // so exact merge order only matters for rare equal-score ties.
        merged.insert(merged.end(), lvl_kps.begin(), lvl_kps.end());
    }

    cout << "  [PRE-NMS] " << pre_nms_merged.size() << "\n";
    cout << "  [MERGED] " << merged.size() << "\n";

    if (pre_nms_out != nullptr) {
    *pre_nms_out = pre_nms_merged;
}

if (local_nms_out != nullptr) {
    *local_nms_out = merged;
}

vector<KP> final_kps =
    anms_fixed_q12_cpu(merged, K_TOP, N_FINAL);

cout << "  [ANMS] " << final_kps.size() << "\n";

return final_kps;
}

int main(int argc, char** argv) {
    string dataset_root = "./data/VGG";
    string out_dir = "software_reference_outputs";

    if (argc >= 2) dataset_root = argv[1];
    if (argc >= 3) out_dir = argv[2];

    ensure_dir(out_dir);

    cout << "[CONFIG] dataset_root=" << dataset_root << "\n";
    cout << "[CONFIG] out_dir=" << out_dir << "\n";
    cout << "[CONFIG] levels=8 scale=1.2 threshold=100 K_top=2048 N_final=1000 c_q12="
         << C_Q12 << "\n";

    int total_files = 0;

    for (int seq = 1; seq <= 8; ++seq) {
        for (int img_id = 1; img_id <= 6; ++img_id) {
            stringstream base;
            base << dataset_root << "/seq" << seq << "/img" << img_id;

            Image img;
            if (!read_image_auto(base.str(), img)) {
                cerr << "[WARN] Missing image: " << base.str() << "\n";
                continue;
            }

            cout << "[RUN] seq" << seq << " img" << img_id << "\n";

            vector<KP> pre_nms_kps;
            vector<KP> local_nms_kps;
            vector<KP> final_kps = run_fixed_detector_one_image(img, &pre_nms_kps, &local_nms_kps);

            // 1) Software candidates before local NMS
            stringstream out_pre;
            out_pre << out_dir
            << "/seq" << seq
            << "_img" << img_id
            << "_thr100_L8_sw_fixed_pre_nms_kps.txt";

            write_kps_txt(out_pre.str(), pre_nms_kps);


            // 2) Software output after local NMS and multiscale merge
            stringstream out_nms;
            out_nms << out_dir
            << "/seq" << seq
            << "_img" << img_id
            << "_thr100_L8_sw_fixed_local_nms_kps.txt";
            write_kps_txt(out_nms.str(), local_nms_kps);

            // 3) Software output after ANMS
            stringstream out_anms;
            out_anms << out_dir
            << "/seq" << seq
            << "_img" << img_id
            << "_thr100_L8_K2048_N1000_c0.90_sw_fixed_anms_kps.txt";
            write_kps_txt(out_anms.str(), final_kps);

            cout << "[WRITE]"
            << " pre_nms=" << pre_nms_kps.size()
            << " local_nms=" << local_nms_kps.size()
            << " anms=" << final_kps.size()
            << "\n";

            total_files++;
        }
    }

    cout << "[DONE] files=" << total_files << "\n";

    return 0;
}
