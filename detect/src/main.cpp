#include "detector.h"
#include "target_selector.h"
#include "visualizer.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

// ============================================================
// 解析命令行参数
// ============================================================
struct Args {
    std::string model_path = "../models/best.onnx";
    std::string input_path = "";
    float conf_threshold = 0.3f;
    float nms_threshold = 0.5f;
    std::string result_dir = "../results";
    bool quiet = false;
};

static Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) a.model_path = argv[++i];
        else if (arg == "--input" && i + 1 < argc) a.input_path = argv[++i];
        else if (arg == "--conf" && i + 1 < argc) a.conf_threshold = std::stof(argv[++i]);
        else if (arg == "--nms" && i + 1 < argc) a.nms_threshold = std::stof(argv[++i]);
        else if (arg == "--out" && i + 1 < argc) a.result_dir = argv[++i];
        else if (arg == "--quiet") a.quiet = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout << R"(
RoboMaster Armor Detection - Based on ZJUT-Deus/Assess_Vision_27

Usage: armor_detect [options]

Options:
  --model <path>   ONNX model path (default: ../models/best.onnx)
  --input <path>   Image/video/directory path (empty = try camera)
  --conf <float>   Confidence threshold (default: 0.3)
  --nms <float>    NMS IoU threshold (default: 0.5)
  --out <dir>      Output directory (default: ../results)
  --quiet          Minimize console output
  --help           Show this help
)" << std::endl;
            exit(0);
        }
    }
    return a;
}

// ============================================================
// 处理单张图片
// ============================================================
static void processImage(const cv::Mat& image, ArmorDetector& detector,
                         TargetSelector& selector, Visualizer& vis,
                         const Args& args, const std::string& save_name)
{
    // 检测
    std::vector<ArmorObject> detections;
    detector.detect(image, detections);

    // 提取中心点
    FrameResult result = selector.update(detections);

    // 可视化
    cv::Mat display = image.clone();
    vis.drawDetections(display, detections);
    vis.drawCenters(display, result);
    vis.drawHUD(display, 0, result.detected_count);

    // 保存
    fs::create_directories(args.result_dir);
    std::string out_path = args.result_dir + "/" + save_name;
    cv::imwrite(out_path, display);

    // 输出
    if (!args.quiet) {
        std::cout << "[" << save_name << "] "
                  << image.cols << "x" << image.rows
                  << " | detections: " << detections.size()
                  << " | centers: " << result.detected_count;
        for (int i = 0; i < result.detected_count; ++i)
            std::cout << " (" << (int)result.centers[i].x
                      << "," << (int)result.centers[i].y << ")";
        std::cout << std::endl;
    }
}

// ============================================================
// 处理视频
// ============================================================
static void processVideo(ArmorDetector& detector, TargetSelector& selector,
                         Visualizer& vis, const Args& args,
                         const std::string& video_path)
{
    cv::VideoCapture cap;
    bool is_camera = video_path.empty();

    if (is_camera) {
        cap.open(0);
        if (!cap.isOpened()) { std::cerr << "Failed to open camera" << std::endl; return; }
    } else {
        cap.open(video_path);
        if (!cap.isOpened()) { std::cerr << "Failed to open: " << video_path << std::endl; return; }
    }

    fs::create_directories(args.result_dir);

    cv::Mat frame;
    int total_frames = 0, detected_frames = 0;
    float total_time_ms = 0;

    while (cap.read(frame)) {
        if (frame.empty()) break;
        total_frames++;

        auto t0 = std::chrono::high_resolution_clock::now();

        // 检测
        std::vector<ArmorObject> detections;
        detector.detect(frame, detections);

        // 提取中心点
        FrameResult result = selector.update(detections);

        auto t1 = std::chrono::high_resolution_clock::now();
        total_time_ms += std::chrono::duration<float, std::milli>(t1 - t0).count();

        if (result.detected_count > 0) detected_frames++;

        // 可视化
        vis.drawDetections(frame, detections);
        vis.drawCenters(frame, result);
        float fps = (total_time_ms > 0 && total_frames > 0)
                        ? (1000.0f * total_frames / total_time_ms) : 0;
        vis.drawHUD(frame, fps, result.detected_count);

        // 保存帧截图
        if (total_frames % 30 == 1) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "frame_%04d.jpg", total_frames);
            cv::imwrite(args.result_dir + "/" + buf, frame);
        }
    }

    cap.release();

    // 统计结果
    float detection_rate = (total_frames > 0)
        ? (float)detected_frames / (float)total_frames : 0;
    float avg_fps = (total_time_ms > 0)
        ? (1000.0f * total_frames / total_time_ms) : 0;

    if (!args.quiet) {
        std::cout << "\n====== Result ======" << std::endl;
        std::cout << "Total frames:    " << total_frames << std::endl;
        std::cout << "Detected frames: " << detected_frames << std::endl;
        std::cout << "Detection rate:  " << std::fixed << std::setprecision(4)
                  << detection_rate << " (" << detected_frames << "/" << total_frames << ")" << std::endl;
        std::cout << "Average FPS:     " << std::fixed << std::setprecision(2) << avg_fps << std::endl;
        std::cout << "====================" << std::endl;
    }
}

// ============================================================
// 批量处理目录
// ============================================================
static void processDirectory(const std::string& dir_path, ArmorDetector& detector,
                             TargetSelector& selector, Visualizer& vis, const Args& args)
{
    if (!args.quiet) std::cout << "Batch processing: " << dir_path << std::endl;
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;

        cv::Mat img = cv::imread(entry.path().string());
        if (img.empty()) continue;

        std::string name = "result_" + entry.path().filename().string();
        processImage(img, detector, selector, vis, args, name);
    }
}

// ============================================================
// 主函数
// ============================================================
int main(int argc, char** argv)
{
    Args args = parseArgs(argc, argv);

    // 初始化检测器
    ArmorDetector detector;
    if (!detector.init(args.model_path, args.conf_threshold, args.nms_threshold)) {
        std::cerr << "Failed to initialize detector" << std::endl;
        return 1;
    }

    TargetSelector selector;
    Visualizer vis;

    // 判断输入类型
    if (!args.input_path.empty()) {
        if (fs::is_directory(args.input_path)) {
            processDirectory(args.input_path, detector, selector, vis, args);
        } else {
            // 检查是图片还是视频
            std::string ext = fs::path(args.input_path).extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            bool is_img = (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
            if (is_img) {
                cv::Mat img = cv::imread(args.input_path);
                if (!img.empty()) {
                    selector.init(img.size());
                    std::string name = "result_" + fs::path(args.input_path).filename().string();
                    processImage(img, detector, selector, vis, args, name);
                }
            } else {
                processVideo(detector, selector, vis, args, args.input_path);
            }
        }
    } else {
        // 默认处理 assets 目录
        std::string assets = "../assets";
        if (fs::exists(assets)) {
            processDirectory(assets, detector, selector, vis, args);
        } else {
            std::cerr << "No input specified and assets/ not found" << std::endl;
            return 1;
        }
    }

    if (!args.quiet)
        std::cout << "Done. Results saved to: " << args.result_dir << "/" << std::endl;

    return 0;
}
