#include "detector.h"
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <cstring>

namespace fs = std::filesystem;

// ======================================================================
// 配置
// ======================================================================
struct Config {
    std::string modelPath     = "../models/best.onnx";
    std::string inputPath     = "";
    float confThreshold       = 0.3f;
    float nmsThreshold        = 0.5f;
    int inputSize             = 640;
    std::string resultsDir    = "../results";
    std::string classNamesPath = "";   // 类别文件路径（可选）
    bool quiet                = false; // P3: 安静模式
    bool saveVideo            = true;
};

// ======================================================================
// 辅助工具
// ======================================================================
namespace Utils {

// ---- P1 修复: 跨平台时间戳 ----
std::string timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[32];
#ifdef _WIN32
    struct tm local;
    localtime_s(&local, &t);
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &local);
#else
    struct tm* local = std::localtime(&t);
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", local);
#endif
    return buf;
}

bool isImageExt(const std::string& path) {
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
           ext == ".bmp" || ext == ".tiff";
}

bool isVideoExt(const std::string& path) {
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".mp4" || ext == ".avi" || ext == ".mov" ||
           ext == ".mkv" || ext == ".wmv";
}

/** ---- P3 修复: 从文件加载类别名称 ---- */
std::vector<std::string> loadClassNames(const std::string& path) {
    std::vector<std::string> names;
    if (path.empty()) return names;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[警告] 无法读取类别文件: " << path << std::endl;
        return names;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty())
            names.push_back(line);
    }
    std::cout << "[信息] 已加载 " << names.size() << " 个类别名称" << std::endl;
    return names;
}

/** ---- P2 修复: 输入路径验证 ---- */
bool validateInput(const std::string& path, std::string& errMsg) {
    if (path.empty()) return true; // 摄像头
    if (!fs::exists(path)) {
        errMsg = "路径不存在: " + path;
        return false;
    }
    if (isImageExt(path) || isVideoExt(path)) {
        return true;
    }
    if (fs::is_directory(path)) {
        // 检查目录下是否有图片
        bool hasImg = false;
        for (const auto& e : fs::directory_iterator(path)) {
            if (e.is_regular_file() && isImageExt(e.path().string())) {
                hasImg = true;
                break;
            }
        }
        if (!hasImg) {
            errMsg = "目录中没有图片文件: " + path;
            return false;
        }
        return true;
    }
    // 尝试作为视频
    return true;
}

/** 计时器辅助 */
struct ScopedTimer {
    std::string label;
    std::chrono::high_resolution_clock::time_point start;
    ScopedTimer(const std::string& lbl) : label(lbl) {
        start = std::chrono::high_resolution_clock::now();
    }
    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        // 输出在析构时自动完成，暂时不用
    }
    double elapsedMs() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

} // namespace Utils

// ======================================================================
// 模块1-4: 主处理流程
// ======================================================================

/** 处理单张图片 */
void processImage(const cv::Mat& image, Detector& detector,
                  const Config& config, const std::string& saveName)
{
    // ---- 模块1: 模型推理 ----
    std::vector<Detection> rawDetections = detector.detect(image);
    double inferMs = detector.getLastInferenceMs();

    // ---- 模块2+3: 结果处理 + 中心点提取 + NMS ----
    std::vector<Detection> results = CenterExtractor::processResults(
        rawDetections, config.confThreshold, config.nmsThreshold);

    // ---- P3 修复: 安静模式 ----
    if (!config.quiet) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "[" << saveName << "] " << image.cols << "x" << image.rows << std::endl;
        std::cout << "  推理: " << std::fixed << std::setprecision(1) << inferMs << " ms"
                  << " | 原始: " << rawDetections.size()
                  << " | NMS后: " << results.size() << std::endl;
        for (size_t i = 0; i < results.size(); ++i) {
            std::cout << "  #" << (i+1)
                      << " 中心(" << (int)results[i].center.x
                      << "," << (int)results[i].center.y << ")"
                      << " conf=" << std::fixed << std::setprecision(3)
                      << results[i].confidence;
            if (!detector.getClassNames().empty() &&
                results[i].classId < (int)detector.getClassNames().size())
                std::cout << " " << detector.getClassNames()[results[i].classId];
            std::cout << std::endl;
        }
    }

    // ---- 模块4: 结果展示 ----
    std::string extra = "Raw: " + std::to_string(rawDetections.size()) +
                        " | NMS: " + std::to_string(results.size());
    cv::Mat display = Visualizer::visualize(image, results, detector.getClassNames(),
                                            0.0, inferMs, extra);

    // 保存
    fs::create_directories(config.resultsDir);
    std::string path = config.resultsDir + "/" + saveName;
    cv::imwrite(path, display);
    if (!config.quiet)
        std::cout << "  [保存] " << path << std::endl;
}

/** 处理视频/摄像头 */
void processVideo(const std::string& videoPath, Detector& detector,
                  const Config& config)
{
    cv::VideoCapture cap;
    bool isCamera = videoPath.empty();

    if (isCamera) {
        cap.open(0);
        if (!cap.isOpened()) {
            std::cerr << "[错误] 无法打开摄像头!" << std::endl;
            return;
        }
        if (!config.quiet) std::cout << "\n[摄像头] 已打开" << std::endl;
    } else {
        cap.open(videoPath);
        if (!cap.isOpened()) {
            std::cerr << "[错误] 无法打开视频: " << videoPath << std::endl;
            return;
        }
        if (!config.quiet) std::cout << "\n[视频] " << videoPath << std::endl;
    }

    // ---- P2 修复: 多种编码尝试 ----
    fs::create_directories(config.resultsDir);
    cv::VideoWriter writer;
    std::string outVideoPath;

    if (!isCamera && config.saveVideo) {
        int fw = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int fh = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 30;

        outVideoPath = config.resultsDir + "/output_" + Utils::timestamp() + ".mp4";

        // 尝试多种编码
        struct CodecTry {
            int fourcc;
            const char* ext;
        };
        CodecTry tries[] = {
            {cv::VideoWriter::fourcc('m', 'p', '4', 'v'), ".mp4"},
            {cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), ".avi"},
            {cv::VideoWriter::fourcc('X', '2', '6', '4'), ".mp4"},
        };

        for (const auto& t : tries) {
            std::string tryPath = config.resultsDir + "/output_" + Utils::timestamp() + t.ext;
            writer.open(tryPath, t.fourcc, fps, cv::Size(fw, fh));
            if (writer.isOpened()) {
                outVideoPath = tryPath;
                break;
            }
        }

        if (writer.isOpened()) {
            if (!config.quiet)
                std::cout << "[输出] 保存视频到: " << outVideoPath << std::endl;
        } else {
            std::cerr << "[警告] 无法创建输出视频（将只保存帧截图）" << std::endl;
        }
    }

    // ---- 视频循环 ----
    cv::Mat frame;
    int frameIdx = 0, totalFrames = 0;
    double totalInferMs = 0, totalProcessMs = 0;
    auto startTime = std::chrono::high_resolution_clock::now();

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        frameIdx++;

        auto loopStart = std::chrono::high_resolution_clock::now();

        // 模块1: 推理
        std::vector<Detection> rawDetections = detector.detect(frame);
        double inferMs = detector.getLastInferenceMs();

        // 模块2+3: 处理 + NMS + 中心点
        std::vector<Detection> results = CenterExtractor::processResults(
            rawDetections, config.confThreshold, config.nmsThreshold);

        // 模块4: 可视化
        auto now = std::chrono::high_resolution_clock::now();
        double totalMs = std::chrono::duration<double, std::milli>(now - loopStart).count();
        double avgFps = (totalMs > 0) ? 1000.0 / totalMs : 0;

        std::string extra = "F:" + std::to_string(frameIdx);
        cv::Mat display = Visualizer::visualize(frame, results, detector.getClassNames(),
                                                avgFps, inferMs, extra);

        // 保存视频帧
        if (writer.isOpened()) writer.write(display);

        // 保存关键帧截图
        if (frameIdx % 30 == 1) {
            char buf[64];
            snprintf(buf, sizeof(buf), "frame_%04d.jpg", frameIdx);
            cv::imwrite(config.resultsDir + "/" + buf, display);
        }

        // ---- P3 修复: 减少控制台输出 ----
        totalInferMs += inferMs;
        totalProcessMs += totalMs;
        totalFrames++;

        if (!config.quiet && frameIdx % 30 == 0) {
            double avgInfer = totalInferMs / totalFrames;
            double avgProc = totalProcessMs / totalFrames;
            std::cout << "\r[帧" << frameIdx << "] "
                      << "FPS:" << std::fixed << std::setprecision(1) << avgFps
                      << " 推理:" << std::setprecision(1) << inferMs << "ms"
                      << " 目标:" << results.size() << "     " << std::flush;
        }
    }

    cap.release();
    if (writer.isOpened()) writer.release();

    auto endTime = std::chrono::high_resolution_clock::now();
    double totalWallMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    // ---- P3 修复: 最终统计 ----
    if (!config.quiet) {
        std::cout << "\n\n===== 视频处理统计 =====" << std::endl;
        std::cout << "总帧数: " << totalFrames << std::endl;
        std::cout << "总耗时: " << std::fixed << std::setprecision(1) << totalWallMs / 1000.0 << " s" << std::endl;
        if (totalFrames > 0) {
            std::cout << "平均FPS: " << std::setprecision(1) << (totalFrames / (totalWallMs / 1000.0)) << std::endl;
            std::cout << "平均推理: " << std::setprecision(1) << (totalInferMs / totalFrames) << " ms/帧" << std::endl;
        }
        if (writer.isOpened()) std::cout << "视频已保存: " << outVideoPath << std::endl;
    }
}

/** 批量处理目录 */
void processDirectory(const std::string& dirPath, Detector& detector,
                      const Config& config)
{
    int totalImages = 0;
    double totalInferMs = 0;

    if (!config.quiet)
        std::cout << "\n===== 批量处理: " << dirPath << " =====" << std::endl;

    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (!entry.is_regular_file()) continue;
        if (!Utils::isImageExt(entry.path().string())) continue;

        cv::Mat img = cv::imread(entry.path().string());
        if (img.empty()) {
            std::cerr << "[警告] 无法读取: " << entry.path().filename() << std::endl;
            continue;
        }

        std::string saveName = "result_" + entry.path().filename().string();
        processImage(img, detector, config, saveName);
        totalImages++;
    }

    if (!config.quiet)
        std::cout << "\n===== 批量完成: " << totalImages << " 张 =====" << std::endl;
}

// ======================================================================
// P3: 帮助信息
// ======================================================================
void printHelp() {
    std::cout << R"(
===== RoboMaster 装甲板检测系统 =====
模块: ①模型推理 → ②结果处理(NMS) → ③中心点提取 → ④结果展示

用法: detect [选项]

选项:
  --model <路径>      ONNX 模型路径 (默认: ../models/best.onnx)
  --input <路径>      输入图片/视频/目录 (空=摄像头尝试)
  --conf <阈值>       置信度阈值 (默认: 0.3)
  --nms <阈值>        NMS IoU 阈值 (默认: 0.5, 0=禁用NMS)
  --classes <路径>    类别名称文件 (每行一个名称)
  --out <目录>        结果输出目录 (默认: ../results)
  --quiet             安静模式 (减少控制台输出)
  --help, -h          显示此帮助

示例:
  detect --input test.jpg                # 单张图片
  detect --input video.mp4               # 视频
  detect --input ./images --quiet        # 批量 + 安静
  detect --conf 0.5 --nms 0.6            # 调参
  detect --classes classes.txt           # 自定义类别名
)" << std::endl;
}

// ======================================================================
// 主函数
// ======================================================================
int main(int argc, char** argv)
{
    Config config;

    // ---- 解析参数 ----
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            config.modelPath = argv[++i];
        } else if (arg == "--input" && i + 1 < argc) {
            config.inputPath = argv[++i];
        } else if (arg == "--conf" && i + 1 < argc) {
            config.confThreshold = std::stof(argv[++i]);
        } else if (arg == "--nms" && i + 1 < argc) {
            config.nmsThreshold = std::stof(argv[++i]);
        } else if (arg == "--out" && i + 1 < argc) {
            config.resultsDir = argv[++i];
        } else if (arg == "--classes" && i + 1 < argc) {
            config.classNamesPath = argv[++i];
        } else if (arg == "--quiet") {
            config.quiet = true;
        } else if (arg == "--help" || arg == "-h") {
            printHelp();
            return 0;
        }
    }

    // ---- P2 修复: 输入验证 ----
    {
        std::string err;
        if (!Utils::validateInput(config.inputPath, err)) {
            std::cerr << "[错误] " << err << std::endl;
            return 1;
        }
    }

    // ---- 打印配置 ----
    if (!config.quiet) {
        std::cout << "\n===== RoboMaster 装甲板检测系统 =====" << std::endl;
        std::cout << "模块: 模型推理 → NMS去重 → 中心点提取 → 可视化" << std::endl;
        std::cout << "模型: " << config.modelPath << std::endl;
        std::cout << "阈值: conf=" << config.confThreshold
                  << " nms=" << config.nmsThreshold << std::endl;
        std::cout << "输出: " << config.resultsDir << std::endl;
        std::cout << "=====================================\n" << std::endl;
    }

    try {
        // ---- 初始化检测器 ----
        Detector detector(config.modelPath, config.inputSize, config.confThreshold);

        // ---- P3 修复: 从文件加载类别名 ----
        auto customClasses = Utils::loadClassNames(config.classNamesPath);
        if (!customClasses.empty()) {
            detector.setClassNames(customClasses);
        }

        // ---- 决定输入模式 ----
        if (!config.inputPath.empty()) {
            if (Utils::isVideoExt(config.inputPath)) {
                processVideo(config.inputPath, detector, config);
            } else if (Utils::isImageExt(config.inputPath)) {
                cv::Mat img = cv::imread(config.inputPath);
                if (img.empty()) {
                    std::cerr << "[错误] 无法读取: " << config.inputPath << std::endl;
                    return 1;
                }
                std::string name = "result_" + fs::path(config.inputPath).filename().string();
                processImage(img, detector, config, name);
            } else if (fs::is_directory(config.inputPath)) {
                processDirectory(config.inputPath, detector, config);
            } else {
                processVideo(config.inputPath, detector, config);
            }
        } else {
            // 尝试摄像头，失败则处理 assets
            cv::VideoCapture testCam(0);
            if (testCam.isOpened()) {
                testCam.release();
                processVideo("", detector, config);
            } else {
                std::string assetsDir = "../assets";
                if (fs::exists(assetsDir)) {
                    processDirectory(assetsDir, detector, config);
                } else {
                    std::cerr << "[错误] 未指定输入，assets 不存在" << std::endl;
                    return 1;
                }
            }
        }

        if (!config.quiet)
            std::cout << "\n处理完成！结果保存在: " << config.resultsDir << "/" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
