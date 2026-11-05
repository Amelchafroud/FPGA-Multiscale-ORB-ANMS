#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include <array>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <string>
#include <cstdint>
#include <algorithm>
#include <exception>
#include <sstream>
#include <cstdlib>
#include <utility>
#include <chrono>
#include <cmath>
#include <random>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>

#include "read_image.h"


typedef unsigned char u8;
typedef uint64_t      u64;

// Helpers
struct PreloadedImage {
    int seq;
    int img;
    int W;
    int H;
    size_t IMG_N;
    std::string base;
    std::vector<u8> pixels;
};

// Pipeline slot: one complete set of device buffers and host-side result storage
// for one frame. With two slots per accelerator lane, the host can submit a new frame while a previous frame is still moving through the FPGA pipeline or being consumed by the CPU descriptor.
struct PipeSlot {
    cl::Buffer bo_src;
    cl::Buffer bo_nms_kp;
    cl::Buffer bo_nms_cnt;
    cl::Buffer bo_anms_kp;
    cl::Buffer bo_anms_cnt;
    cl::Buffer bo_stats;

    cl::Event evt_h2d;
    cl::Event evt_ms_hn;
    cl::Event evt_anms;
    cl::Event evt_cnt_anms;
    cl::Event evt_kp_anms;
    cl::Event evt_nms_cnt;
    cl::Event evt_stats;

    const PreloadedImage* input = nullptr;

    int seq = 0;
    int img = 0;
    int W = 0;
    int H = 0;
    size_t IMG_N = 0;
    int max_kp_merge_img = 0;
    int anms_kp_count = 0;
    int merged_nms_count_from_buf = 0;

    std::array<int, 8> pix = {0,0,0,0,0,0,0,0};
    std::array<int, 8> max_kp_img = {0,0,0,0,0,0,0,0};
    std::array<int, 17> stats = {0};

    std::vector<u64> anms_kp_out;

    bool valid = false;
    int slot_id = -1;
};

// Descriptor job passed from FPGA/main thread to CPU descriptor worker: We copy final ANMS keypoints.
struct DescJob {
    const PreloadedImage* input = nullptr;

    int seq = 0;
    int img = 0;
    int W = 0;
    int H = 0;
    int anms_kp_count = 0;
    int slot = -1;

    bool measure = true;

    std::vector<u64> packed_kps;
};

// Statistics produced by the CPU descriptor.
struct DescWorkerStats {
    double sum_cpu_orient = 0.0;
    double sum_cpu_brief  = 0.0;
    double sum_cpu_desc   = 0.0;

    long long sum_desc_kp = 0;
    long long jobs = 0;

    uint64_t checksum = 0;
};
// descriptor-side: orientation + oriented BRIEF.

static constexpr int DESC_BITS  = 256;
static constexpr int DESC_WORDS = DESC_BITS / 64;
using DescWord = uint64_t;

struct CpuKeyPoint {
    int x = 0;
    int y = 0;
    uint32_t score = 0;
    float angle = 0.0f;
};

struct BriefPair {
    int x1, y1;
    int x2, y2;
};

static constexpr double CPU_PI = 3.14159265358979323846;
static constexpr int ANGLE_BINS = 36;

static float g_cosLUT[ANGLE_BINS];
static float g_sinLUT[ANGLE_BINS];

static BriefPair g_brief_pairs[DESC_BITS];
static int16_t g_rotDX[ANGLE_BINS][DESC_BITS * 2];
static int16_t g_rotDY[ANGLE_BINS][DESC_BITS * 2];

static int g_umax[64];
static int g_umax_R = -1;

static inline double now_ms()
{
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(
        clock::now().time_since_epoch()).count();
}

static inline void initAngleLUT()
{
    for (int i = 0; i < ANGLE_BINS; ++i) {
        float a = float((2.0 * CPU_PI) * i / ANGLE_BINS);
        g_cosLUT[i] = std::cos(a);
        g_sinLUT[i] = std::sin(a);
    }
}

static inline int angleBin(float ang)
{
    if (ang < 0.0f) ang += float(2.0 * CPU_PI);
    float b = ang * (ANGLE_BINS / float(2.0 * CPU_PI));
    return int(std::lround(b)) % ANGLE_BINS;
}

static inline void initCircleRows(int R)
{
    if (g_umax_R == R) return;

    for (int v = 0; v <= R; ++v) {
        g_umax[v] = (int)std::floor(
            std::sqrt(double(R) * double(R) - double(v) * double(v))
        );
    }

    g_umax_R = R;
}

static inline bool inBoundsVec(int x, int y, int W, int H)
{
    return ((unsigned)x < (unsigned)W) && ((unsigned)y < (unsigned)H);
}

static void initBriefPattern()
{
    std::default_random_engine gen(12345);
    std::normal_distribution<float> dist(0.0f, 31.0f / 5.0f);

    for (int i = 0; i < DESC_BITS; ++i) {
        g_brief_pairs[i].x1 = (int)std::lround(dist(gen));
        g_brief_pairs[i].y1 = (int)std::lround(dist(gen));
        g_brief_pairs[i].x2 = (int)std::lround(dist(gen));
        g_brief_pairs[i].y2 = (int)std::lround(dist(gen));
    }
}

static void initRotatedBriefPattern()
{
    for (int b = 0; b < ANGLE_BINS; ++b) {
        float cs = g_cosLUT[b];
        float sn = g_sinLUT[b];

        for (int j = 0; j < DESC_BITS; ++j) {
            int x1 = g_brief_pairs[j].x1;
            int y1 = g_brief_pairs[j].y1;
            int x2 = g_brief_pairs[j].x2;
            int y2 = g_brief_pairs[j].y2;

            g_rotDX[b][2 * j + 0] = (int16_t)std::lround(cs * x1 - sn * y1);
            g_rotDY[b][2 * j + 0] = (int16_t)std::lround(sn * x1 + cs * y1);

            g_rotDX[b][2 * j + 1] = (int16_t)std::lround(cs * x2 - sn * y2);
            g_rotDY[b][2 * j + 1] = (int16_t)std::lround(sn * x2 + cs * y2);
        }
    }
}

static void initCpuDescriptor()
{
    initAngleLUT();
    initBriefPattern();
    initRotatedBriefPattern();
}

static float computeOrientationAtVec(
    const std::vector<u8>& gray,
    int W,
    int H,
    int cx,
    int cy,
    int R = 12)
{
    initCircleRows(R);

    long long m10 = 0;
    long long m01 = 0;

    for (int v = -R; v <= R; ++v) {
        int yy = cy + v;
        if ((unsigned)yy >= (unsigned)H) continue;

        int umax = g_umax[std::abs(v)];
        int xl = cx - umax;
        int xr = cx + umax;

        if (xr < 0 || xl >= W) continue;
        if (xl < 0) xl = 0;
        if (xr >= W) xr = W - 1;

        for (int u = xl; u <= xr; ++u) {
            int I = (int)gray[(size_t)yy * (size_t)W + (size_t)u];
            m10 += (long long)(u - cx) * I;
            m01 += (long long)v * I;
        }
    }

    if (m10 == 0 && m01 == 0) return 0.0f;
    return std::atan2((float)m01, (float)m10);
}

static void assignOrientationForKeypointsVec(
    const std::vector<u8>& gray,
    int W,
    int H,
    std::vector<CpuKeyPoint>& kps,
    int R = 12)
{
    for (size_t i = 0; i < kps.size(); ++i) {
        int x = kps[i].x;
        int y = kps[i].y;

        if (x <= R || x >= W - 1 - R || y <= R || y >= H - 1 - R) {
            kps[i].angle = 0.0f;
        } else {
            kps[i].angle = computeOrientationAtVec(gray, W, H, x, y, R);
        }
    }
}

static void computeBriefDescriptorsOrientedVec(
    const std::vector<u8>& gray,
    int W,
    int H,
    const std::vector<CpuKeyPoint>& kps,
    std::vector<std::array<DescWord, DESC_WORDS>>& desc)
{
    const size_t n = kps.size();

    for (size_t i = 0; i < n; ++i) {
        for (int widx = 0; widx < DESC_WORDS; ++widx) {
            desc[i][widx] = 0;
        }

        int b = angleBin(kps[i].angle);
        int word = 0;
        int shift = 0;

        for (int j = 0; j < DESC_BITS; ++j) {
            int dx1 = g_rotDX[b][2 * j + 0];
            int dy1 = g_rotDY[b][2 * j + 0];
            int dx2 = g_rotDX[b][2 * j + 1];
            int dy2 = g_rotDY[b][2 * j + 1];

            int p1x = kps[i].x + dx1;
            int p1y = kps[i].y + dy1;
            int p2x = kps[i].x + dx2;
            int p2y = kps[i].y + dy2;

            bool bit = false;

            if (inBoundsVec(p1x, p1y, W, H) &&
                inBoundsVec(p2x, p2y, W, H)) {
                bit = gray[(size_t)p1y * (size_t)W + (size_t)p1x] <
                      gray[(size_t)p2y * (size_t)W + (size_t)p2x];
            }

            if (bit) {
                desc[i][word] |= (DescWord(1) << shift);
            }

            shift++;
            if (shift == 64) {
                shift = 0;
                word++;
            }
        }
    }
}

static std::vector<CpuKeyPoint> decodeAnmsKeypoints(
    const std::vector<u64>& packed,
    int n,
    int W,
    int H)
{
    std::vector<CpuKeyPoint> out;
    out.reserve((size_t)n);

    for (int i = 0; i < n; ++i) {
        u64 w64 = packed[(size_t)i];

        uint16_t x = (uint16_t)(w64 & 0xFFFFull);
        uint16_t y = (uint16_t)((w64 >> 16) & 0xFFFFull);
        uint32_t s = (uint32_t)((w64 >> 32) & 0xFFFFFFFFull);

        if ((int)x >= 0 && (int)x < W && (int)y >= 0 && (int)y < H) {
            CpuKeyPoint kp;
            kp.x = (int)x;
            kp.y = (int)y;
            kp.score = s;
            kp.angle = 0.0f;
            out.push_back(kp);
        }
    }

    return out;
}

static std::vector<unsigned char> read_binary(const std::string& fn)
{
    std::ifstream f(fn, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "[ERR] open " << fn << "\n";
        std::exit(1);
    }
    auto n = f.tellg();
    if (n <= 0) {
        std::cerr << "[ERR] size " << fn << "\n";
        std::exit(1);
    }
    std::vector<unsigned char> b((size_t)n);
    f.seekg(0, std::ios::beg);
    if (!f.read(reinterpret_cast<char*>(b.data()), n)) {
        std::cerr << "[ERR] read " << fn << "\n";
        std::exit(1);
    }
    return b;
}

static float parse_float_or(const char* s, float fallback)
{
    if (!s) return fallback;
    try {
        std::string t(s);
        if (t.empty()) return fallback;
        return std::stof(t);
    } catch (...) {
        std::cerr << "[WARN] Could not parse float from '" << s
                  << "'. Using fallback " << fallback << "\n";
        return fallback;
    }
}

static int parse_int_or(const char* s, int fallback)
{
    if (!s) return fallback;
    try {
        std::string t(s);
        if (t.empty()) return fallback;
        return std::stoi(t);
    } catch (...) {
        std::cerr << "[WARN] Could not parse int from '" << s
                  << "'. Using fallback " << fallback << "\n";
        return fallback;
    }
}

static inline double evt_ms(const cl::Event& e)
{
    if (!e()) return 0.0;
    cl_ulong t0 = 0, t1 = 0;
    e.getProfilingInfo(CL_PROFILING_COMMAND_START, &t0);
    e.getProfilingInfo(CL_PROFILING_COMMAND_END,   &t1);
    return (t1 > t0) ? (double)(t1 - t0) / 1e6 : 0.0;
}

static inline std::string strip_ext_if_any(const std::string& s)
{
    if (s.size() >= 4) {
        std::string ext = s.substr(s.size() - 4);
        if (ext == ".ppm" || ext == ".pgm")
            return s.substr(0, s.size() - 4);
    }
    return s;
}

static inline int clamp_nonneg(int v, int hi)
{
    if (v < 0) return 0;
    if (v > hi) return hi;
    return v;
}

static inline int cap_from_req(int req, int max_allowed)
{
    if (max_allowed <= 0) return 0;
    if (req <= 0) return max_allowed;
    return std::max(0, std::min(req, max_allowed));
}

static void mkdir_p(const std::string& dir)
{
    std::string cmd = "mkdir -p " + dir;
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::cerr << "[WARN] mkdir failed (rc=" << rc << "): " << cmd << "\n";
    }
}

static void save_kps_txt(const std::string& path,
                         const std::vector<u64>& kps,
                         int n_kp,
                         int W, int H)
{
    std::ofstream ofs(path);
    if (!ofs) {
        std::cerr << "[WARN] Cannot write " << path << "\n";
        return;
    }

    ofs << "# x y score_u32\n";
    ofs << "# packed format: x[15:0], y[31:16], score_u32[63:32]\n";

    for (int i = 0; i < n_kp; ++i) {
        u64 w64 = kps[(size_t)i];
        uint16_t x = (uint16_t)(w64 & 0xFFFFull);
        uint16_t y = (uint16_t)((w64 >> 16) & 0xFFFFull);
        uint32_t s = (uint32_t)((w64 >> 32) & 0xFFFFFFFFull);

        if ((int)x >= 0 && (int)x < W && (int)y >= 0 && (int)y < H) {
            ofs << x << " " << y << " " << s << "\n";
        }
    }
}

static void save_u64_bin(const std::string& path,
                         const std::vector<u64>& v,
                         int n_valid)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        std::cerr << "[WARN] Cannot write " << path << "\n";
        return;
    }
    if (n_valid > 0) {
        ofs.write(reinterpret_cast<const char*>(v.data()),
                  (std::streamsize)n_valid * (std::streamsize)sizeof(u64));
    }
}

static void save_i32_bin(const std::string& path, int value)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        std::cerr << "[WARN] Cannot write " << path << "\n";
        return;
    }
    ofs.write(reinterpret_cast<const char*>(&value), sizeof(int));
}

static void save_txt_scalar(const std::string& path, int value)
{
    std::ofstream ofs(path);
    if (!ofs) {
        std::cerr << "[WARN] Cannot write " << path << "\n";
        return;
    }
    ofs << value << "\n";
}

static void save_stats_txt(const std::string& path,
                           const std::array<int, 17>& stats,
                           int active_levels)
{
    std::ofstream ofs(path);
    if (!ofs) {
        std::cerr << "[WARN] Cannot write " << path << "\n";
        return;
    }

    ofs << "active_levels " << active_levels << "\n";

    for (int l = 0; l < 8; ++l) {
        ofs << "harris_candidate_count_l" << l << " " << stats[(size_t)l] << "\n";
    }

    for (int l = 0; l < 8; ++l) {
        ofs << "merged_written_nms_count_l" << l << " " << stats[(size_t)(8 + l)] << "\n";
    }

    ofs << "merged_written_nms_count_total " << stats[16] << "\n";
}

static cl::Kernel make_kernel_with_fallback(
    const cl::Program& prog,
    const std::vector<std::string>& names)
{
    cl_int err = CL_SUCCESS;
    for (const auto& name : names) {
        cl::Kernel k(prog, name.c_str(), &err);
        if (err == CL_SUCCESS) {
            std::cout << "[HOST] Using kernel: " << name << "\n";
            return k;
        }
    }

    std::cerr << "[ERR] Could not create kernel. Tried names:\n";
    for (const auto& name : names) {
        std::cerr << "  - " << name << "\n";
    }
    std::exit(1);
}

// Main

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr
            << "Usage: " << argv[0]
            << " <xclbin>"
            << " [vgg_root=./data/VGG]"
            << " [thresh=100]"
            << " [max_kp_merge_req=0(auto)]"
            << " [dev=0]"
            << " [save_outputs=0]"
            << " [K_top=2048]"
            << " [N_final=1000]"
            << " [c=0.9]"
			<< " [active_levels=8]"
			<< " [max_kp_l0_req=0(auto)]"
			<< " [max_kp_l1_req=0(auto)]"
			<< " [max_kp_l2_req=0(auto)]"
			<< " [max_kp_l3_req=0(auto)]"
			<< " [max_kp_l4_req=0(auto)]"
			<< " [max_kp_l5_req=0(auto)]"
			<< " [max_kp_l6_req=0(auto)]"
			<< " [max_kp_l7_req=0(auto)]"
            << " [clean_benchmark=1]"
            << " [num_slots=2]"
            << " [flush_each_frame=0]"
            << " [trace_host=0]"
            << " [warmup_frames=1]"
            << " [num_lanes=1]"
            << " [force_single_cu=0]"
            << " [power_repeats=1]\n";
        return 1;
    }

    const std::string xclbin        = argv[1];
    const std::string vgg_root      = (argc > 2)  ? argv[2] : "./data/VGG";
    const int thresh                = (argc > 3)  ? parse_int_or(argv[3], 100) : 100;
    const int max_kp_merge_req      = (argc > 4)  ? parse_int_or(argv[4], 0) : 0;
    const int dev_index0            = (argc > 5)  ? parse_int_or(argv[5], 0) : 0;
    const int save_outputs          = (argc > 6)  ? parse_int_or(argv[6], 0) : 0;
    const int K_top                 = (argc > 7)  ? parse_int_or(argv[7], 2048) : 2048;
    const int N_final               = (argc > 8)  ? parse_int_or(argv[8], 1000) : 1000;
    const float c_anms              = (argc > 9)  ? parse_float_or(argv[9], 0.9f) : 0.9f;
    int active_levels               = (argc > 10) ? parse_int_or(argv[10], 8) : 8;
    const int max_kp_l0_req         = (argc > 11) ? parse_int_or(argv[11], 0) : 0;
    const int max_kp_l1_req         = (argc > 12) ? parse_int_or(argv[12], 0) : 0;
    const int max_kp_l2_req         = (argc > 13) ? parse_int_or(argv[13], 0) : 0;
    const int max_kp_l3_req         = (argc > 14) ? parse_int_or(argv[14], 0) : 0;
    const int max_kp_l4_req         = (argc > 15) ? parse_int_or(argv[15], 0) : 0;
    const int max_kp_l5_req         = (argc > 16) ? parse_int_or(argv[16], 0) : 0;
    const int max_kp_l6_req         = (argc > 17) ? parse_int_or(argv[17], 0) : 0;
    const int max_kp_l7_req         = (argc > 18) ? parse_int_or(argv[18], 0) : 0;
    const int clean_benchmark       = (argc > 19) ? parse_int_or(argv[19], 1) : 1;
    int num_slots                   = (argc > 20) ? parse_int_or(argv[20], 2) : 2;
    int flush_each_frame            = (argc > 21) ? parse_int_or(argv[21], 0) : 0;
    int trace_host                  = (argc > 22) ? parse_int_or(argv[22], 0) : 0;
    int warmup_frames               = (argc > 23) ? parse_int_or(argv[23], 1) : 1;
    int num_lanes                   = (argc > 24) ? parse_int_or(argv[24], 1) : 1;
    int force_single_cu             = (argc > 25) ? parse_int_or(argv[25], 0) : 0;
    int power_repeats               = (argc > 26) ? parse_int_or(argv[26], 1) : 1;

    if (force_single_cu < 0 || force_single_cu > 6) {
        std::cerr << "[WARN] force_single_cu must be 0, 1, 2, 3, 4, 5, or 6. Using 0.\n";
        force_single_cu = 0;
    }
    if (force_single_cu > 0) {
        num_lanes = 1;
    }

    if (num_lanes < 1) num_lanes = 1;
    if (num_lanes > 6) {
        std::cerr << "[WARN] This host version supports up to 6 accelerator lanes. Using num_lanes=6.\n";
        num_lanes = 6;
    }

    if (num_slots < 2) num_slots = 2;
    if (num_slots > 12) num_slots = 12;
    if (num_slots < num_lanes) num_slots = num_lanes;
    if (flush_each_frame != 0) flush_each_frame = 1;
    if (trace_host != 0) trace_host = 1;
    if (warmup_frames < 0) warmup_frames = 0;
    if (warmup_frames > 8) warmup_frames = 8;
    if (warmup_frames > 0 && warmup_frames < num_lanes) {
        // Warm all accelerator lanes before measured timing.
        warmup_frames = num_lanes;
    }

    if (power_repeats < 1) power_repeats = 1;
    if (power_repeats > 10000) {
        std::cerr << "[WARN] power_repeats too large. Using 10000.\n";
        power_repeats = 10000;
    }

    if (active_levels < 1) active_levels = 1;
    if (active_levels > 8) active_levels = 8;

    const std::array<int, 8> max_kp_req = {
        max_kp_l0_req, max_kp_l1_req, max_kp_l2_req, max_kp_l3_req,
        max_kp_l4_req, max_kp_l5_req, max_kp_l6_req, max_kp_l7_req
    };


    const int MAX_W0 = 1000;
    const int MAX_H0 = 700;

    std::array<int, 8> maxW = {0,0,0,0,0,0,0,0};
    std::array<int, 8> maxH = {0,0,0,0,0,0,0,0};
    std::array<int, 8> maxPixels = {0,0,0,0,0,0,0,0};

    maxW[0] = MAX_W0;
    maxH[0] = MAX_H0;
    maxPixels[0] = maxW[0] * maxH[0];

    for (int l = 1; l < 8; ++l) {
        maxW[(size_t)l] = (maxW[(size_t)(l - 1)] * 5) / 6;
        maxH[(size_t)l] = (maxH[(size_t)(l - 1)] * 5) / 6;
        maxPixels[(size_t)l] = maxW[(size_t)l] * maxH[(size_t)l];
    }

    std::array<int, 8> max_kp_global = {0,0,0,0,0,0,0,0};

    int max_kp_merge_auto_global = 0;
    for (int l = 0; l < 8; ++l) {
        if (active_levels >= l + 1) {
            max_kp_global[(size_t)l] =
                cap_from_req(max_kp_req[(size_t)l], maxPixels[(size_t)l]);
        } else {
            max_kp_global[(size_t)l] = 0;
        }
        max_kp_merge_auto_global += max_kp_global[(size_t)l];
    }

    const int max_kp_merge_global =
        (max_kp_merge_req <= 0)
        ? max_kp_merge_auto_global
        : std::max(1, std::min(max_kp_merge_req, max_kp_merge_auto_global));

    std::cout << "[HOST] Dataset root        : " << vgg_root << "\n";
    std::cout << "[HOST] thresh              : " << thresh << "\n";
    std::cout << "[HOST] active_levels       : " << active_levels << "\n";
    for (int l = 0; l < 8; ++l) {
        std::cout << "[HOST] max_kp_l" << l << "(global)   : "
                  << max_kp_global[(size_t)l] << "\n";
    }
    std::cout << "[HOST] max_kp_merge(global): " << max_kp_merge_global << "\n";
    std::cout << "[HOST] save_outputs        : " << save_outputs << "\n";
    std::cout << "[HOST] ANMS K_top          : " << K_top << "\n";
    std::cout << "[HOST] ANMS N_final        : " << N_final << "\n";
    std::cout << "[HOST] ANMS c              : " << c_anms << "\n";
    std::cout << "[HOST] clean_benchmark     : " << clean_benchmark << "\n";
    std::cout << "[HOST] num_slots           : " << num_slots << "\n";
    std::cout << "[HOST] num_lanes           : " << num_lanes << "\n";
    std::cout << "[HOST] force_single_cu     : " << force_single_cu << "\n";
    std::cout << "[HOST] flush_each_frame    : " << flush_each_frame << "\n";
    std::cout << "[HOST] trace_host          : " << trace_host << "\n";
    std::cout << "[HOST] warmup_frames       : " << warmup_frames << "\n";
    std::cout << "[HOST] power_repeats       : " << power_repeats << "\n\n";

    const std::string out_dir = "hw_out_multiscale";
    if (save_outputs) mkdir_p(out_dir);

    try {
        // Platform / Device
        std::vector<cl::Platform> plats;
        cl::Platform::get(&plats);

        cl::Platform plat;
        for (auto &p : plats) {
            std::string vendor = p.getInfo<CL_PLATFORM_VENDOR>();
            if (vendor.find("Xilinx") != std::string::npos) {
                plat = p;
                break;
            }
        }
        if (!plat()) {
            std::cerr << "[ERR] No Xilinx OpenCL platform found.\n";
            return 1;
        }

        std::vector<cl::Device> devs;
        plat.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devs);
        if (devs.empty()) {
            std::cerr << "[ERR] No accelerator device found.\n";
            return 1;
        }

        int dev_index = dev_index0;
        if (dev_index < 0 || dev_index >= (int)devs.size()) dev_index = 0;

        cl::Device dev = devs[dev_index];
        cl::Context ctx(dev);

        std::vector<cl::CommandQueue> q_h2d_lanes;
        std::vector<cl::CommandQueue> q_ms_lanes;
        std::vector<cl::CommandQueue> q_anms_lanes;
        std::vector<cl::CommandQueue> q_d2h_lanes;

        q_h2d_lanes.reserve((size_t)num_lanes);
        q_ms_lanes.reserve((size_t)num_lanes);
        q_anms_lanes.reserve((size_t)num_lanes);
        q_d2h_lanes.reserve((size_t)num_lanes);

        for (int l = 0; l < num_lanes; ++l) {
            q_h2d_lanes.emplace_back(ctx, dev, CL_QUEUE_PROFILING_ENABLE);
            q_ms_lanes.emplace_back(ctx, dev, CL_QUEUE_PROFILING_ENABLE);
            q_anms_lanes.emplace_back(ctx, dev, CL_QUEUE_PROFILING_ENABLE);
            q_d2h_lanes.emplace_back(ctx, dev, CL_QUEUE_PROFILING_ENABLE);
        }

        // Program / Kernels
        auto blob = read_binary(xclbin);
        cl::Program::Binaries bins;
        bins.push_back(blob);

        cl_int err = CL_SUCCESS;
        cl::Program prog(ctx, {dev}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cerr << "[ERR] Program load failed err=" << err << "\n";
            return 1;
        }

        std::vector<cl::Kernel> krn_ms_lanes;
        std::vector<cl::Kernel> krn_anms_lanes;
        krn_ms_lanes.reserve((size_t)num_lanes);
        krn_anms_lanes.reserve((size_t)num_lanes);

        auto make_ms_instance = [&](int cu_id) {
            return make_kernel_with_fallback(
                prog, {"ms_harris_nms_merge:{ms_harris_nms_merge_" + std::to_string(cu_id) + "}",
                       "ms_harris_nms_merge_" + std::to_string(cu_id)});
        };

        auto make_anms_instance = [&](int cu_id) {
            return make_kernel_with_fallback(
                prog, {"anms_kernel:{anms_kernel_" + std::to_string(cu_id) + "}",
                       "anms_kernel_" + std::to_string(cu_id)});
        };

        if (num_lanes == 1 && force_single_cu == 0) {
            krn_ms_lanes.push_back(make_kernel_with_fallback(
                prog, {"ms_harris_nms_merge", "multiscale_harris_nms_merge_kernel"}));

            krn_anms_lanes.push_back(make_kernel_with_fallback(
                prog, {"anms_kernel"}));
        } else {
            const int first_cu = (force_single_cu > 0) ? force_single_cu : 1;
            const int lanes_to_create = (force_single_cu > 0) ? 1 : num_lanes;

            for (int li = 0; li < lanes_to_create; ++li) {
                const int cu_id = first_cu + li;
                krn_ms_lanes.push_back(make_ms_instance(cu_id));
                krn_anms_lanes.push_back(make_anms_instance(cu_id));
            }
        }

        // Initialize CPU-side ORB descriptor pattern
        initCpuDescriptor();


        // Multi-buffered ping-pong pipeline.
        // Each slot owns independent input, intermediate NMS, ANMS output, and stats buffers.
        // This allows different frames to occupy different FPGA/host stages concurrently, while completed ANMS keypoints are handed to the CPU descriptor.
        const int max_anms_out = std::max(1, N_final);

        std::vector<PipeSlot> slots((size_t)num_slots);

        for (int s = 0; s < num_slots; ++s) {
            slots[(size_t)s].slot_id = s;
            slots[(size_t)s].bo_src =
                cl::Buffer(ctx, CL_MEM_READ_ONLY,  (size_t)maxPixels[0] * sizeof(u8));

            slots[(size_t)s].bo_nms_kp =
                cl::Buffer(ctx, CL_MEM_READ_WRITE, (size_t)max_kp_merge_global * sizeof(u64));

            slots[(size_t)s].bo_nms_cnt =
                cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(int));

            slots[(size_t)s].bo_anms_kp =
                cl::Buffer(ctx, CL_MEM_READ_WRITE, (size_t)max_anms_out * sizeof(u64));

            slots[(size_t)s].bo_anms_cnt =
                cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(int));

            slots[(size_t)s].bo_stats =
                cl::Buffer(ctx, CL_MEM_READ_WRITE, (size_t)17 * sizeof(int));

            slots[(size_t)s].anms_kp_out.assign((size_t)max_anms_out, 0);
        }

        // Preload all VGG images into host RAM before the timed loop: This removes disk I/O and image loading from the per-image FPGA loop.
        std::vector<PreloadedImage> images;
        int skipped = 0;

        for (int seq = 1; seq <= 8; ++seq) {
            for (int img = 1; img <= 6; ++img) {

                std::string base = vgg_root + "/seq" + std::to_string(seq)
                                 + "/img" + std::to_string(img);
                base = strip_ext_if_any(base);

                GrayscaleImage* g_in = readImage(base);
                if (!g_in) {
                    std::cerr << "[WARN] Cannot read image base: "
                              << base << " (skipping)\n";
                    skipped++;
                    continue;
                }

                const int W = g_in->m_w;
                const int H = g_in->m_h;
                const size_t IMG_N = (size_t)W * (size_t)H;

                if (W <= 0 || H <= 0 || IMG_N > (size_t)maxPixels[0]) {
                    std::cerr << "[WARN] Invalid/too large image: "
                              << base << " got " << W << "x" << H
                              << " (skipping)\n";
                    delete g_in;
                    skipped++;
                    continue;
                }

                PreloadedImage item;
                item.seq = seq;
                item.img = img;
                item.W = W;
                item.H = H;
                item.IMG_N = IMG_N;
                item.base = base;
                item.pixels.resize(IMG_N);

                std::memcpy(item.pixels.data(), g_in->m_data, IMG_N);
                delete g_in;

                images.push_back(std::move(item));
            }
        }

        std::cout << "[HOST] Preloaded images     : " << images.size() << "\n";
        if (skipped > 0) {
            std::cout << "[HOST] Skipped images       : " << skipped << "\n";
        }

        // results
        std::cout << "\n VGG Multiscale Harris+NMS + ANMS benchmark (FPGA, FIXED-POINT) \n";
        std::cout << " Timed path includes only required transfers: input H2D + multiscale HN + ANMS + final ANMS D2H.\n";
        std::cout << " Auxiliary reads (stats_out, merged NMS count, merged NMS tuples, exports) are excluded from Total.\n";

        if (!clean_benchmark) {
            std::cout << std::left
                      << std::setw(12) << "Image"
                      << std::setw(10) << "WxH"
                      << std::setw(4)  << "L";

            for (int l = 0; l < 8; ++l) {
                std::ostringstream hname;
                hname << "H" << l;
                std::cout << std::setw(8) << hname.str();
            }

            for (int l = 0; l < 8; ++l) {
                std::ostringstream nname;
                nname << "N" << l;
                std::cout << std::setw(8) << nname.str();
            }

            std::cout << std::setw(10) << "NMS_TOT"
                      << std::setw(10) << "ANMS_KP"
                      << std::setw(10) << "H2D"
                      << std::setw(12) << "MS_HN"
                      << std::setw(12) << "ANMS"
                      << std::setw(12) << "D2H"
                      << std::setw(12) << "Total"
                      << "\n";

            std::cout << std::string(12 + 10 + 4 + 8*8 + 8*8 + 10 + 10 + 10 + 12 + 12 + 12 + 12, '-') << "\n";
        }
        // Timed-path accumulators
        double sum_h2d = 0.0, sum_ms_hn = 0.0, sum_anms = 0.0, sum_d2h = 0.0, sum_tot = 0.0;

        DescWorkerStats desc_stats;

        // Auxiliary accumulators
        std::array<long long, 8> sum_hcand = {0,0,0,0,0,0,0,0};
        std::array<long long, 8> sum_nms   = {0,0,0,0,0,0,0,0};
        long long sum_nms_tot = 0;
        long long sum_anms_kp = 0;

        std::vector<long long> lane_images((size_t)num_lanes, 0);
        std::vector<uint64_t> lane_kp_checksum((size_t)num_lanes, 0);

        int ok_images = 0;

        //host-side trace for CPU/FPGA overlap visualization: disabled during normal timing unless trace_host=1.

        struct HostTraceEvent {
            double t0_ms = 0.0;
            double t1_ms = 0.0;
            int seq = 0;
            int img = 0;
            int slot = -1;
            std::string stage;
        };

        std::vector<HostTraceEvent> host_trace;
        std::mutex host_trace_mtx;
        double host_trace_t0_ms = 0.0;

        auto trace_add = [&](const std::string& stage,
                             int seq, int img, int slot,
                             double t0_ms, double t1_ms) {
            if (!trace_host) return;

            HostTraceEvent ev;
            ev.t0_ms = t0_ms - host_trace_t0_ms;
            ev.t1_ms = t1_ms - host_trace_t0_ms;
            ev.seq = seq;
            ev.img = img;
            ev.slot = slot;
            ev.stage = stage;

            std::lock_guard<std::mutex> lock(host_trace_mtx);
            host_trace.push_back(std::move(ev));
        };

        auto mix64 = [](uint64_t x) {
            x ^= x >> 30;
            x *= 0xbf58476d1ce4e5b9ULL;
            x ^= x >> 27;
            x *= 0x94d049bb133111ebULL;
            x ^= x >> 31;
            return x;
        };


        // CPU descriptor: The main thread pushes completed FPGA ANMS keypoints into a queue.
        // The worker thread : computes orientation + BRIEF in parallel with FPGA acc
        std::deque<DescJob> desc_queue;
        std::mutex desc_mtx;
        std::condition_variable desc_cv;
        std::condition_variable desc_done_cv;
        int desc_pending = 0;
        bool desc_stop = false;

        auto process_desc_job = [&](const DescJob& job) {
            double t_cpu_orient_ms = 0.0;
            double t_cpu_brief_ms  = 0.0;
            double t_cpu_desc_ms   = 0.0;
            int desc_kp_count      = 0;
            uint64_t desc_checksum = 0;

            if (job.input && job.anms_kp_count > 0) {
                const bool do_trace = trace_host && job.measure;
                const double t_desc0 = do_trace ? now_ms() : 0.0;

                const std::vector<u8>& img_in = job.input->pixels;

                std::vector<CpuKeyPoint> cpu_kps =
                    decodeAnmsKeypoints(job.packed_kps, job.anms_kp_count, job.W, job.H);

                desc_kp_count = (int)cpu_kps.size();

                std::vector<std::array<DescWord, DESC_WORDS>> cpu_desc(
                    (size_t)desc_kp_count
                );

                const double t_or0 = now_ms();
                assignOrientationForKeypointsVec(img_in, job.W, job.H, cpu_kps, 12);
                const double t_or1 = now_ms();

                const double t_br0 = now_ms();
                computeBriefDescriptorsOrientedVec(img_in, job.W, job.H, cpu_kps, cpu_desc);
                const double t_br1 = now_ms();

                // Consume descriptors 
                // This checksum is not included in BRIEF timing.
                for (const auto& d : cpu_desc) {
                    for (int w = 0; w < DESC_WORDS; ++w) {
                        desc_checksum ^= d[w];
                    }
                }

                const double t_desc1 = do_trace ? now_ms() : 0.0;
                if (do_trace) {
                    trace_add("CPU_DESC", job.seq, job.img, job.slot, t_desc0, t_desc1);
                }

                t_cpu_orient_ms = t_or1 - t_or0;
                t_cpu_brief_ms  = t_br1 - t_br0;
                t_cpu_desc_ms   = t_cpu_orient_ms + t_cpu_brief_ms;
            }

            // Only measured jobs update final descriptor statistics.
            if (job.measure) {
                desc_stats.sum_cpu_orient += t_cpu_orient_ms;
                desc_stats.sum_cpu_brief  += t_cpu_brief_ms;
                desc_stats.sum_cpu_desc   += t_cpu_desc_ms;
                desc_stats.sum_desc_kp    += desc_kp_count;
                desc_stats.checksum       ^= desc_checksum;
                desc_stats.jobs++;
            }
        };

        std::thread desc_worker([&]() {
            while (true) {
                DescJob job;

                {
                    std::unique_lock<std::mutex> lock(desc_mtx);
                    desc_cv.wait(lock, [&]() {
                        return desc_stop || !desc_queue.empty();
                    });

                    if (desc_stop && desc_queue.empty()) {
                        break;
                    }

                    job = std::move(desc_queue.front());
                    desc_queue.pop_front();
                }

                process_desc_job(job);

                {
                    std::lock_guard<std::mutex> lock(desc_mtx);
                    desc_pending--;
                    if (desc_pending == 0 && desc_queue.empty()) {
                        desc_done_cv.notify_all();
                    }
                }
            }
        });

        auto enqueue_desc_job = [&](const PipeSlot& s, bool measure) {
            DescJob job;
            job.input = s.input;
            job.seq = s.seq;
            job.img = s.img;
            job.W = s.W;
            job.H = s.H;
            job.anms_kp_count = clamp_nonneg(s.anms_kp_count, N_final);
            job.slot = s.slot_id;
            job.measure = measure;

            if (job.anms_kp_count > 0) {
                job.packed_kps.assign(
                    s.anms_kp_out.begin(),
                    s.anms_kp_out.begin() + job.anms_kp_count
                );
            }

            {
                std::lock_guard<std::mutex> lock(desc_mtx);
                desc_queue.push_back(std::move(job));
                desc_pending++;
            }

            desc_cv.notify_one();
        };

        auto wait_desc_idle = [&]() {
            std::unique_lock<std::mutex> lock(desc_mtx);
            desc_done_cv.wait(lock, [&]() {
                return desc_pending == 0 && desc_queue.empty();
            });
        };

        // Asynchronous multi-buffered host pipeline.
        // Each frame is submitted through the FPGA stages using non-blocking OpenCL commands and event dependencies: H2D -> MS-HN -> ANMS -> final D2H.
        // Across different slots, new frames can be submitted while previous frames are still executing on the FPGA or being consumed by the CPU descriptor.
        // The ANMS keypoint buffer is read with a fixed size N_final instead of first blocking on the keypoint count and then launching a second read. This avoids introducing a host-side blocking dependency inside the submission loop.


        if (save_outputs) {
            std::cerr << "[WARN] save_outputs=1 is disabled in this ping-pong benchmark path.\n";
            std::cerr << "       Run the old sequential host path if you need PNG/TXT/BIN exports.\n";
        }

        auto submit_frame = [&](PipeSlot& s, const PreloadedImage& input, bool measure) {
            const int lane_id = s.slot_id % num_lanes;
            cl::CommandQueue& q_h2d_lane  = q_h2d_lanes[(size_t)lane_id];
            cl::CommandQueue& q_ms_lane   = q_ms_lanes[(size_t)lane_id];
            cl::CommandQueue& q_anms_lane = q_anms_lanes[(size_t)lane_id];
            cl::CommandQueue& q_d2h_lane  = q_d2h_lanes[(size_t)lane_id];

            cl::Kernel& krn_ms = krn_ms_lanes[(size_t)lane_id];
            cl::Kernel& krn_anms = krn_anms_lanes[(size_t)lane_id];

            const bool do_trace = trace_host && measure;
            const double t_submit0 = do_trace ? now_ms() : 0.0;

            s.input = &input;
            s.seq = input.seq;
            s.img = input.img;
            s.W = input.W;
            s.H = input.H;
            s.IMG_N = input.IMG_N;

            const int W = s.W;
            const int H = s.H;

            std::array<int, 8> Wl = {0,0,0,0,0,0,0,0};
            std::array<int, 8> Hl = {0,0,0,0,0,0,0,0};

            Wl[0] = W;
            Hl[0] = H;
            s.pix[0] = Wl[0] * Hl[0];

            for (int l = 1; l < 8; ++l) {
                Wl[(size_t)l] = (Wl[(size_t)(l - 1)] * 5) / 6;
                Hl[(size_t)l] = (Hl[(size_t)(l - 1)] * 5) / 6;
                s.pix[(size_t)l] = Wl[(size_t)l] * Hl[(size_t)l];
            }

            int max_kp_merge_auto_img = 0;
            for (int l = 0; l < 8; ++l) {
                if (active_levels >= l + 1) {
                    s.max_kp_img[(size_t)l] =
                        cap_from_req(max_kp_req[(size_t)l], s.pix[(size_t)l]);
                } else {
                    s.max_kp_img[(size_t)l] = 0;
                }
                max_kp_merge_auto_img += s.max_kp_img[(size_t)l];
            }

            s.max_kp_merge_img =
                (max_kp_merge_req <= 0)
                ? max_kp_merge_auto_img
                : std::max(1, std::min(max_kp_merge_req, max_kp_merge_auto_img));

            s.anms_kp_count = 0;
            s.merged_nms_count_from_buf = 0;
            s.stats.fill(0);
            std::fill(s.anms_kp_out.begin(), s.anms_kp_out.end(), 0);

            // Kernel args
            krn_ms.setArg(0,  s.bo_src);
            krn_ms.setArg(1,  s.bo_nms_kp);
            krn_ms.setArg(2,  s.bo_nms_cnt);
            krn_ms.setArg(3,  s.bo_stats);
            krn_ms.setArg(4,  thresh);
            krn_ms.setArg(5,  W);
            krn_ms.setArg(6,  H);
            krn_ms.setArg(7,  active_levels);
            krn_ms.setArg(8,  s.max_kp_merge_img);
            krn_ms.setArg(9,  s.max_kp_img[0]);
            krn_ms.setArg(10, s.max_kp_img[1]);
            krn_ms.setArg(11, s.max_kp_img[2]);
            krn_ms.setArg(12, s.max_kp_img[3]);
            krn_ms.setArg(13, s.max_kp_img[4]);
            krn_ms.setArg(14, s.max_kp_img[5]);
            krn_ms.setArg(15, s.max_kp_img[6]);
            krn_ms.setArg(16, s.max_kp_img[7]);

            krn_anms.setArg(0, s.bo_nms_kp);
            krn_anms.setArg(1, s.bo_nms_cnt);
            krn_anms.setArg(2, s.bo_anms_kp);
            krn_anms.setArg(3, s.bo_anms_cnt);
            krn_anms.setArg(4, K_top);
            krn_anms.setArg(5, N_final);
            krn_anms.setArg(6, c_anms);
            krn_anms.setArg(7, s.max_kp_merge_img);

            q_h2d_lane.enqueueWriteBuffer(
                s.bo_src, false, 0, s.IMG_N * sizeof(u8),
                input.pixels.data(), nullptr, &s.evt_h2d
            );

            std::vector<cl::Event> wait_h2d = { s.evt_h2d };
            q_ms_lane.enqueueTask(krn_ms, &wait_h2d, &s.evt_ms_hn);

            std::vector<cl::Event> wait_ms = { s.evt_ms_hn };
            q_anms_lane.enqueueTask(krn_anms, &wait_ms, &s.evt_anms);


            std::vector<cl::Event> wait_anms_done = { s.evt_anms };

            // Schedule the final ANMS count read as a non-blocking D2H transfer.
            // The transfer depends on evt_anms, so it starts only after ANMS finishes.
            // The host does not wait here; consume_slot() waits later only when the slot output is actually needed or the slot is reused.
            q_d2h_lane.enqueueReadBuffer(
                s.bo_anms_cnt, false, 0, sizeof(int),
                &s.anms_kp_count, &wait_anms_done, &s.evt_cnt_anms
            );

            // Schedule the final ANMS keypoint read as a non-blocking fixed-size transfer.
            // We read the full final keypoint buffer, normally N_final tuples, instead of first blocking on the count and then launching a variable-size read.
            // this avoids a host-side synchronization point inside the frame-submission loop.
            q_d2h_lane.enqueueReadBuffer(
                s.bo_anms_kp, false, 0,
                (size_t)max_anms_out * sizeof(u64),
                s.anms_kp_out.data(), &wait_anms_done, &s.evt_kp_anms
            );

            if (!clean_benchmark) {
                q_d2h_lane.enqueueReadBuffer(
                    s.bo_nms_cnt, false, 0, sizeof(int),
                    &s.merged_nms_count_from_buf, &wait_anms_done, &s.evt_nms_cnt
                );

                q_d2h_lane.enqueueReadBuffer(
                    s.bo_stats, false, 0, sizeof(int) * s.stats.size(),
                    s.stats.data(), &wait_anms_done, &s.evt_stats
                );
            }

            s.valid = true;

            // Optional queue flush: pushes queued commands to the device without waiting for completion.
            // We keep it optional because avoiding per-frame flushes can reduce host overhead in the asynchronous multi-buffered pipeline.
            if (flush_each_frame) {
                q_h2d_lane.flush();
                q_ms_lane.flush();
                q_anms_lane.flush();
                q_d2h_lane.flush();
            }

            const double t_submit1 = do_trace ? now_ms() : 0.0;
            if (do_trace) {
                trace_add("FPGA_SUBMIT", s.seq, s.img, s.slot_id, t_submit0, t_submit1);
            }
        };

        auto consume_slot = [&](PipeSlot& s, bool measure) {
            if (!s.valid) return;

            const bool do_trace = trace_host && measure;
            const double t_wait0 = do_trace ? now_ms() : 0.0;

            // wait for the current slot's final result
            s.evt_cnt_anms.wait();
            s.evt_kp_anms.wait();

            if (!clean_benchmark) {
                s.evt_nms_cnt.wait();
                s.evt_stats.wait();
            }

            const double t_wait1 = do_trace ? now_ms() : 0.0;
            if (do_trace) {
                trace_add("FPGA_WAIT_OUTPUT", s.seq, s.img, s.slot_id, t_wait0, t_wait1);
            }

            const int seq = s.seq;
            const int img = s.img;
            const int W = s.W;
            const int H = s.H;

            s.anms_kp_count = clamp_nonneg(s.anms_kp_count, N_final);


            // Event-based FPGA timing

            const double t_h2d_ms       = evt_ms(s.evt_h2d);
            const double t_ms_hn_ms     = evt_ms(s.evt_ms_hn);
            const double t_anms_ms      = evt_ms(s.evt_anms);
            const double t_d2h_final_ms = evt_ms(s.evt_cnt_anms) + evt_ms(s.evt_kp_anms);
            const double t_tot_final_ms = t_h2d_ms + t_ms_hn_ms + t_anms_ms + t_d2h_final_ms;



            // Debug/statistics path: in clean benchmark mode, we skip stats processing completely.
            std::array<int, 8> hcand = {0,0,0,0,0,0,0,0};
            std::array<int, 8> nms   = {0,0,0,0,0,0,0,0};
            int nmsTot = 0;

            if (!clean_benchmark) {
                s.merged_nms_count_from_buf =
                    clamp_nonneg(s.merged_nms_count_from_buf, s.max_kp_merge_img);

                for (int l = 0; l < 8; ++l) {
                    const int level_enabled = (active_levels >= l + 1) ? 1 : 0;

                    hcand[(size_t)l] = level_enabled
                        ? clamp_nonneg(s.stats[(size_t)l], s.pix[(size_t)l])
                        : 0;

                    nms[(size_t)l] = level_enabled
                        ? clamp_nonneg(s.stats[(size_t)(8 + l)], s.max_kp_img[(size_t)l])
                        : 0;
                }

                nmsTot = clamp_nonneg(s.stats[16], s.max_kp_merge_img);

                if (s.merged_nms_count_from_buf != nmsTot) {
                    std::cerr << "[WARN] Count mismatch at seq" << seq << "/img" << img
                              << " : bo_nms_cnt=" << s.merged_nms_count_from_buf
                              << " stats_out[16]=" << nmsTot << "\n";
                    nmsTot = s.merged_nms_count_from_buf;
                }
            }

            // Per-image row printing

            if (measure && !clean_benchmark) {
                std::ostringstream img_id; img_id << "seq" << seq << "/img" << img;
                std::ostringstream wh;     wh << W << "x" << H;

                std::cout << std::left
                          << std::setw(12) << img_id.str()
                          << std::setw(10) << wh.str()
                          << std::setw(4)  << active_levels;

                for (int l = 0; l < 8; ++l) {
                    std::cout << std::setw(8) << hcand[(size_t)l];
                }

                for (int l = 0; l < 8; ++l) {
                    std::cout << std::setw(8) << nms[(size_t)l];
                }

                std::cout << std::setw(10) << nmsTot
                          << std::setw(10) << s.anms_kp_count
                          << std::setw(10) << std::fixed << std::setprecision(3) << t_h2d_ms
                          << std::setw(12) << std::fixed << std::setprecision(3) << t_ms_hn_ms
                          << std::setw(12) << std::fixed << std::setprecision(3) << t_anms_ms
                          << std::setw(12) << std::fixed << std::setprecision(3) << t_d2h_final_ms
                          << std::setw(12) << std::fixed << std::setprecision(3) << t_tot_final_ms
                          << "\n";
            }

            // Accumulation

            if (measure) {
                sum_h2d    += t_h2d_ms;
                sum_ms_hn  += t_ms_hn_ms;
                sum_anms   += t_anms_ms;
                sum_d2h    += t_d2h_final_ms;
                sum_tot    += t_tot_final_ms;

                if (!clean_benchmark) {
                    for (int l = 0; l < 8; ++l) {
                        sum_hcand[(size_t)l] += hcand[(size_t)l];
                        sum_nms[(size_t)l]   += nms[(size_t)l];
                    }

                    sum_nms_tot += nmsTot;
                }

                const int lane_id_for_stats = s.slot_id % num_lanes;
                if (lane_id_for_stats >= 0 && (size_t)lane_id_for_stats < lane_images.size()) {
                    lane_images[(size_t)lane_id_for_stats]++;
                    uint64_t local_kp_checksum = mix64(
                        ((uint64_t)(unsigned)seq << 32) ^
                        ((uint64_t)(unsigned)img << 16) ^
                        (uint64_t)(unsigned)s.anms_kp_count
                    );

                    const int n_chk = clamp_nonneg(s.anms_kp_count, N_final);
                    for (int kk = 0; kk < n_chk; ++kk) {
                        local_kp_checksum ^= mix64(
                            s.anms_kp_out[(size_t)kk] ^
                            ((uint64_t)(unsigned)kk * 0x9e3779b97f4a7c15ULL)
                        );
                    }

                    lane_kp_checksum[(size_t)lane_id_for_stats] ^= local_kp_checksum;
                }

                sum_anms_kp += s.anms_kp_count;
            }

            // Hand off the completed FPGA ANMS keypoints to the CPU descriptor worker.
            // N_final=1000 keypoints × 8 bytes = 8 KB/image.
            enqueue_desc_job(s, measure);

            if (measure) {
                ok_images++;
            }
            s.valid = false;
        };

        const int warmup_count =
            std::min(warmup_frames, (int)images.size());

        if (warmup_count > 0) {
            std::cout << "[HOST] Running warm-up frames : "
                      << warmup_count << " (excluded from timing/statistics/trace)\n";

            for (int i = 0; i < warmup_count; ++i) {
                PipeSlot& cur = slots[(size_t)i % (size_t)num_slots];

                if (cur.valid) {
                    consume_slot(cur, false);
                }

                submit_frame(cur, images[(size_t)i], false);
            }

            // Drain warm-up FPGA frames.
            for (int s = 0; s < num_slots; ++s) {
                consume_slot(slots[(size_t)s], false);
            }

            // Wait until warm-up descriptor jobs finished.
            wait_desc_idle();

            std::cout << "[HOST] Warm-up completed. Starting measured loop.\n";
        }

        if (power_repeats > 1) {
            std::cout << "[HOST] Power-repeat mode   : repeating measured dataset "
                      << power_repeats << " times inside one process.\n";
        }

        const double wall_pipe_start_ms = now_ms();
        host_trace_t0_ms = wall_pipe_start_ms;

        size_t global_frame_idx = 0;
        for (int rep = 0; rep < power_repeats; ++rep) {
            for (size_t i = 0; i < images.size(); ++i, ++global_frame_idx) {
                PipeSlot& cur = slots[global_frame_idx % (size_t)num_slots];

                // If this slot is still holding an old frame, consume it before reuse.
                if (cur.valid) {
                    consume_slot(cur, true);
                }

                // Submit current frame without blocking.
                submit_frame(cur, images[i], true);
            }
        }

        // Drain all remaining in-flight measured FPGA frames.
        for (int s = 0; s < num_slots; ++s) {
            consume_slot(slots[(size_t)s], true);
        }

        // Stop descriptor after all queued descriptor jobs are processed.
        {
            std::lock_guard<std::mutex> lock(desc_mtx);
            desc_stop = true;
        }
        desc_cv.notify_one();
        desc_worker.join();

        if (trace_host) {
            std::ofstream ofs("host_cpu_fpga_overlap_trace.csv");
            ofs << "stage,seq,img,slot,t0_ms,t1_ms,duration_ms\n";

            {
                std::lock_guard<std::mutex> lock(host_trace_mtx);

                std::sort(host_trace.begin(), host_trace.end(),
                          [](const HostTraceEvent& a, const HostTraceEvent& b) {
                              return a.t0_ms < b.t0_ms;
                          });

                for (const auto& ev : host_trace) {
                    ofs << ev.stage << ","
                        << ev.seq << ","
                        << ev.img << ","
                        << ev.slot << ","
                        << std::fixed << std::setprecision(3) << ev.t0_ms << ","
                        << std::fixed << std::setprecision(3) << ev.t1_ms << ","
                        << std::fixed << std::setprecision(3) << (ev.t1_ms - ev.t0_ms)
                        << "\n";
                }
            }

            std::cout << "[HOST] Wrote host trace: host_cpu_fpga_overlap_trace.csv\n";
        }

        // Full pipeline wall time: FPGA work + CPU descriptor completion.
        const double wall_pipe_total_ms = now_ms() - wall_pipe_start_ms;

        // Summary
        std::cout << "\n Summary \n";
        std::cout << "Processed: " << ok_images << " images";
        if (skipped > 0) std::cout << " (skipped " << skipped << ")";
        std::cout << "\n";
        std::cout << "Pipeline slots        : " << num_slots << "\n";
        std::cout << "Accelerator lanes     : " << num_lanes << "\n";


        if (ok_images > 0) {
            std::cout << "Avg H2D(ms)           : " << std::fixed << std::setprecision(3) << (sum_h2d   / ok_images) << "\n";
            std::cout << "Avg MS_HN(ms)         : " << std::fixed << std::setprecision(3) << (sum_ms_hn / ok_images) << "\n";
            std::cout << "Avg ANMS(ms)          : " << std::fixed << std::setprecision(3) << (sum_anms  / ok_images) << "\n";
            std::cout << "Avg D2H_final(ms)     : " << std::fixed << std::setprecision(3) << (sum_d2h   / ok_images) << "\n";
            std::cout << "Avg Total_final(ms)   : " << std::fixed << std::setprecision(3) << (sum_tot   / ok_images) << "\n";

            const double avg_h2d_ms       = sum_h2d   / ok_images;
            const double avg_ms_hn_ms     = sum_ms_hn / ok_images;
            const double avg_anms_ms      = sum_anms  / ok_images;
            const double avg_d2h_ms       = sum_d2h   / ok_images;
            const double avg_fpga_detector_ms = sum_tot / ok_images;

            const double avg_fpga_stage_bottleneck_ms =
                std::max(std::max(avg_h2d_ms, avg_ms_hn_ms),
                         std::max(avg_anms_ms, avg_d2h_ms));

            const double avg_fpga_throughput_ms_print =
                (num_lanes > 0) ? (avg_fpga_stage_bottleneck_ms / (double)num_lanes)
                                : avg_fpga_stage_bottleneck_ms;

            std::cout << "Avg FPGA stage bottleneck(ms): " << std::fixed << std::setprecision(3)
                      << avg_fpga_stage_bottleneck_ms << "\n";
            std::cout << "Avg FPGA throughput(ms/img): " << std::fixed << std::setprecision(3)
                      << avg_fpga_throughput_ms_print << "\n";

            const long long desc_jobs = desc_stats.jobs;
            const double avg_cpu_orient_ms =
                (desc_jobs > 0) ? (desc_stats.sum_cpu_orient / desc_jobs) : 0.0;
            const double avg_cpu_brief_ms =
                (desc_jobs > 0) ? (desc_stats.sum_cpu_brief / desc_jobs) : 0.0;
            const double avg_cpu_desc_ms =
                (desc_jobs > 0) ? (desc_stats.sum_cpu_desc / desc_jobs) : 0.0;

            const double avg_seq_orb_ms  = avg_fpga_detector_ms + avg_cpu_desc_ms;

            // For multi-CU stage-pipelined execution, event profiling gives the latency of one image through one accelerator lane. 
            //Steady-state throughput is approximated by the slowest FPGA stage divided by the number of lanes.
            const double avg_fpga_throughput_ms =
                (num_lanes > 0) ? (avg_fpga_stage_bottleneck_ms / (double)num_lanes)
                                : avg_fpga_stage_bottleneck_ms;

            const double avg_pipe_est_ms = std::max(avg_fpga_throughput_ms, avg_cpu_desc_ms);

            std::cout << "Avg CPU orient(ms)    : " << std::fixed << std::setprecision(3)
                      << avg_cpu_orient_ms << "\n";
            std::cout << "Avg CPU BRIEF(ms)     : " << std::fixed << std::setprecision(3)
                      << avg_cpu_brief_ms << "\n";
            std::cout << "Avg CPU desc(ms)      : " << std::fixed << std::setprecision(3)
                      << avg_cpu_desc_ms << "\n";
            std::cout << "Descriptor jobs       : " << desc_jobs << "\n";
            std::cout << "Avg desc KP           : "
                      << (desc_jobs > 0 ? (desc_stats.sum_desc_kp / desc_jobs) : 0)
                      << "\n";
            std::cout << "CPU desc checksum     : " << desc_stats.checksum << "\n";

            std::cout << "Avg ORB sequential(ms): " << std::fixed << std::setprecision(3)
                      << avg_seq_orb_ms << "\n";
            std::cout << "Avg ORB pipeline est(ms): " << std::fixed << std::setprecision(3)
                      << avg_pipe_est_ms << "\n";
            std::cout << "Avg ORB pipeline est FPS: " << std::fixed << std::setprecision(3)
                      << (avg_pipe_est_ms > 0.0 ? 1000.0 / avg_pipe_est_ms : 0.0)
                      << "\n";
            const double measured_pipe_wall_ms =
                (ok_images > 0) ? (wall_pipe_total_ms / ok_images) : 0.0;

            const double measured_pipe_fps =
                (measured_pipe_wall_ms > 0.0) ? (1000.0 / measured_pipe_wall_ms) : 0.0;

            const double speedup_vs_seq =
                (measured_pipe_wall_ms > 0.0) ? (avg_seq_orb_ms / measured_pipe_wall_ms) : 0.0;

            const double gap_to_ideal_ms =
                measured_pipe_wall_ms - avg_pipe_est_ms;

            std::cout << "Measured pipeline wall(ms/img): " << std::fixed << std::setprecision(3)
                      << measured_pipe_wall_ms << "\n";

            std::cout << "Measured pipeline FPS     : " << std::fixed << std::setprecision(3)
                      << measured_pipe_fps << "\n";

            std::cout << "Speedup vs sequential     : " << std::fixed << std::setprecision(3)
                      << speedup_vs_seq << "x\n";

            std::cout << "Gap vs stage-pipeline ideal(ms): " << std::fixed << std::setprecision(3)
                      << gap_to_ideal_ms << "\n";

            if (!clean_benchmark) {
                for (int l = 0; l < 8; ++l) {
                    std::cout << "Avg HCand L" << l << "          : "
                              << (sum_hcand[(size_t)l] / ok_images) << "\n";
                }

                for (int l = 0; l < 8; ++l) {
                    std::cout << "Avg merged NMS L" << l << "     : "
                              << (sum_nms[(size_t)l] / ok_images) << "\n";
                }

                std::cout << "Avg merged NMS total  : " << (sum_nms_tot / ok_images) << "\n";
            }

            for (int l = 0; l < num_lanes; ++l) {
                std::cout << "FPGA lane" << l << " images     : "
                          << lane_images[(size_t)l] << "\n";
            }

            for (int l = 0; l < num_lanes; ++l) {
                std::cout << "FPGA lane" << l << " kp checksum: "
                          << lane_kp_checksum[(size_t)l] << "\n";
            }

            std::cout << "Avg ANMS_KP           : " << (sum_anms_kp / ok_images) << "\n";
            std::cout << "Avg lane-latency FPS  : " << std::fixed << std::setprecision(3)
                      << (sum_tot > 0.0 ? (1000.0 / (sum_tot / ok_images)) : 0.0) << "\n";
        }

        return 0;

    } catch (const cl::Error& e) {
        std::cerr << "[OpenCL] " << e.what() << " err=" << e.err() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[STD] " << e.what() << "\n";
        return 1;
    }
}
