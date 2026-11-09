#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

static bool load_vgg_image(const fs::path& dataset_root,
                           int seq,
                           int img_id,
                           cv::Mat& gray,
                           std::string& loaded_path)
{
    const fs::path base =
        dataset_root / ("seq" + std::to_string(seq)) /
        ("img" + std::to_string(img_id));

    const fs::path ppm = base.string() + ".ppm";
    const fs::path pgm = base.string() + ".pgm";

    cv::Mat color = cv::imread(ppm.string(), cv::IMREAD_COLOR);
    if (!color.empty()) {
        loaded_path = ppm.string();
    } else {
        color = cv::imread(pgm.string(), cv::IMREAD_COLOR);
        if (color.empty()) {
            return false;
        }
        loaded_path = pgm.string();
    }

    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
    return true;
}

static bool write_keypoints(const fs::path& path,
                            const std::vector<cv::KeyPoint>& keypoints)
{
    std::ofstream ofs(path);
    if (!ofs) {
        return false;
    }

    ofs << "# x y response octave angle size\n";
    ofs << std::fixed << std::setprecision(6);

    for (const auto& kp : keypoints) {
        ofs << kp.pt.x << ' '
            << kp.pt.y << ' '
            << kp.response << ' '
            << kp.octave << ' '
            << kp.angle << ' '
            << kp.size << '\n';
    }

    return true;
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr
            << "Usage: " << argv[0]
            << " <dataset_root> <output_dir>\n";
        return 1;
    }

    const fs::path dataset_root = argv[1];
    const fs::path output_dir = argv[2];

    // Final OpenCV ORB baseline configuration used for evaluation:
    // nfeatures   = 1000
    // scaleFactor = 1.2
    // nlevels     = 8
    //
    // All remaining ORB parameters use OpenCV defaults.
    cv::Ptr<cv::ORB> orb = cv::ORB::create(
        1000,
        1.2f,
        8
    );

    constexpr int NUM_SEQUENCES = 8;
    constexpr int NUM_IMAGES = 6;

    fs::create_directories(output_dir);

    double total_time_ms = 0.0;
    int processed_images = 0;
    std::size_t total_keypoints = 0;

    std::cout << "[CONFIG] dataset_root=" << dataset_root.string() << '\n';
    std::cout << "[CONFIG] output_dir=" << output_dir.string() << '\n';
    std::cout << "[CONFIG] nfeatures=1000 scaleFactor=1.2 nlevels=8\n";

    for (int seq = 1; seq <= NUM_SEQUENCES; ++seq) {
        const fs::path seq_out =
            output_dir / ("seq" + std::to_string(seq));
        fs::create_directories(seq_out);

        for (int img_id = 1; img_id <= NUM_IMAGES; ++img_id) {
            cv::Mat gray;
            std::string loaded_path;

            if (!load_vgg_image(dataset_root, seq, img_id,
                                gray, loaded_path)) {
                std::cerr
                    << "[WARN] Missing image for seq" << seq
                    << " img" << img_id
                    << " (.ppm/.pgm)\n";
                continue;
            }

            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;

            const auto t0 =
                std::chrono::high_resolution_clock::now();

            orb->detectAndCompute(
                gray,
                cv::noArray(),
                keypoints,
                descriptors
            );

            const auto t1 =
                std::chrono::high_resolution_clock::now();

            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    t1 - t0
                ).count();

            const fs::path out_file =
                seq_out /
                ("img" + std::to_string(img_id) +
                 "_opencv_orb_kps.txt");

            if (!write_keypoints(out_file, keypoints)) {
                std::cerr
                    << "[ERROR] Cannot write "
                    << out_file.string() << '\n';
                return 1;
            }

            total_time_ms += elapsed_ms;
            total_keypoints += keypoints.size();
            ++processed_images;

            std::cout
                << "[RUN] seq" << seq
                << " img" << img_id
                << " input=" << loaded_path
                << " keypoints=" << keypoints.size()
                << " time_ms="
                << std::fixed << std::setprecision(3)
                << elapsed_ms
                << '\n';
        }
    }

    if (processed_images == 0) {
        std::cerr << "[ERROR] No images were processed.\n";
        return 1;
    }

    const double mean_time_ms =
        total_time_ms / static_cast<double>(processed_images);

    const double fps =
        (mean_time_ms > 0.0)
            ? 1000.0 / mean_time_ms
            : 0.0;

    const double mean_keypoints =
        static_cast<double>(total_keypoints) /
        static_cast<double>(processed_images);

    const fs::path summary_path =
        output_dir / "benchmark_summary.txt";

    {
        std::ofstream summary(summary_path);
        if (!summary) {
            std::cerr
                << "[ERROR] Cannot write "
                << summary_path.string() << '\n';
            return 1;
        }

        summary << std::fixed << std::setprecision(6);
        summary << "processed_images " << processed_images << '\n';
        summary << "mean_time_ms " << mean_time_ms << '\n';
        summary << "fps " << fps << '\n';
        summary << "mean_keypoints " << mean_keypoints << '\n';
        summary << "nfeatures 1000\n";
        summary << "scaleFactor 1.2\n";
        summary << "nlevels 8\n";
    }

    std::cout << "\n========== OpenCV ORB Summary ==========\n";
    std::cout << "Processed images : " << processed_images << '\n';
    std::cout << "Mean time (ms)   : "
              << std::fixed << std::setprecision(3)
              << mean_time_ms << '\n';
    std::cout << "FPS              : "
              << std::fixed << std::setprecision(3)
              << fps << '\n';
    std::cout << "Mean keypoints   : "
              << std::fixed << std::setprecision(2)
              << mean_keypoints << '\n';
    std::cout << "Summary file     : "
              << summary_path.string() << '\n';

    return 0;
}
