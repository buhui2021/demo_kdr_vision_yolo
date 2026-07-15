#ifndef DETECTOR_H
#define DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <cstdio>
#include <fstream>

// ======================================================================
// 模块1: 数据结构定义
// ======================================================================

/// 单个装甲板检测结果
struct Detection {
    cv::Rect bbox;          // 边界框 (x, y, w, h)
    float confidence;       // 置信度 (0~1)
    int classId;            // 类别ID
    cv::Point2f center;     // 装甲板中心点像素坐标
    cv::Point2f corners[4]; // 四个角点（矩形框角点，用于 PnP）
};

// ======================================================================
// 模块2: 模型推理 (Model Inference)
// ======================================================================

class Detector {
public:
    Detector(const std::string& modelPath,
             int inputSize = 640,
             float confThresh = 0.3f);

    ~Detector() = default;

    /** 对单帧图像执行推理 */
    std::vector<Detection> detect(const cv::Mat& image);

    // 访问器
    const std::vector<std::string>& getClassNames() const { return classNames_; }
    void setClassNames(const std::vector<std::string>& names) { classNames_ = names; }
    int getInputSize() const { return inputSize_; }
    float getConfThreshold() const { return confThreshold_; }
    void setConfThreshold(float t) { confThreshold_ = t; }
    double getLastInferenceMs() const { return lastInferenceMs_; }

    /** 打印模型信息 */
    void printModelInfo() const;

    /** 检查模型文件是否存在 */
    static bool modelExists(const std::string& path) {
        std::ifstream f(path);
        return f.good();
    }

private:
    cv::dnn::Net net_;
    int inputSize_;
    float confThreshold_;
    double lastInferenceMs_ = 0.0;
    std::vector<std::string> classNames_;

    // 前处理：图像 -> blob (letterbox 居中填充)
    void preprocess(const cv::Mat& image, cv::Mat& blob,
                    float& scaleFactor, cv::Size& pad);

    // 后处理：模型输出 -> Detection 列表
    std::vector<Detection> postprocess(const std::vector<cv::Mat>& outputs,
                                       float scaleFactor,
                                       const cv::Size& pad,
                                       const cv::Size& imgSize);
};

// ======================================================================
// 模块3: 结果处理 + 中心点提取 + NMS
// ======================================================================

namespace CenterExtractor {

/**
 * 提取并精化中心点，同时做 NMS 去重
 * @param raw          原始 Detection 列表
 * @param confThresh   置信度阈值
 * @param nmsThresh    NMS IoU 阈值 (0~1), 0=不做NMS
 * @return 处理后的检测列表
 */
inline std::vector<Detection> processResults(const std::vector<Detection>& raw,
                                              float confThreshold,
                                              float nmsThresh = 0.5f) {
    std::vector<Detection> valid;
    for (const auto& d : raw) {
        if (d.confidence < confThreshold) continue;
        Detection det = d;
        // 精化中心点
        det.center.x = det.bbox.x + det.bbox.width / 2.0f;
        det.center.y = det.bbox.y + det.bbox.height / 2.0f;
        // 计算矩形角点
        det.corners[0] = cv::Point2f(det.bbox.x, det.bbox.y);
        det.corners[1] = cv::Point2f(det.bbox.x + det.bbox.width, det.bbox.y);
        det.corners[2] = cv::Point2f(det.bbox.x + det.bbox.width,
                                     det.bbox.y + det.bbox.height);
        det.corners[3] = cv::Point2f(det.bbox.x, det.bbox.y + det.bbox.height);
        valid.push_back(det);
    }

    // 按置信度降序排列
    std::sort(valid.begin(), valid.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });

    // ---- NMS 去重 ----
    if (nmsThresh > 0.0f && !valid.empty()) {
        std::vector<cv::Rect> boxes;
        std::vector<float> confs;
        for (const auto& d : valid) {
            boxes.push_back(d.bbox);
            confs.push_back(d.confidence);
        }
        std::vector<int> keep;
        cv::dnn::NMSBoxes(boxes, confs, 0.0f, nmsThresh, keep);

        std::vector<Detection> nmsResult;
        for (int idx : keep) {
            nmsResult.push_back(valid[idx]);
        }
        return nmsResult;
    }

    return valid;
}

} // namespace CenterExtractor

// ======================================================================
// 模块4: 结果展示 (Result Visualization)
// ======================================================================

namespace Visualizer {

// 颜色常量
const cv::Scalar COLOR_BOX(0, 255, 0);
const cv::Scalar COLOR_CENTER(0, 0, 255);
const cv::Scalar COLOR_RING(0, 255, 255);
const cv::Scalar COLOR_INFO(0, 255, 0);
const cv::Scalar COLOR_WARN(0, 0, 255);
const cv::Scalar COLOR_COORD(0, 255, 255);

/**
 * 可视化检测结果
 * @param image       原图
 * @param detections  处理后检测列表
 * @param classNames  类别名称
 * @param fps         FPS
 * @param inferMs     推理耗时
 * @param extraInfo   额外信息（如 NMS 前后数量）
 * @return 标注图像
 */
inline cv::Mat visualize(const cv::Mat& image,
                         const std::vector<Detection>& detections,
                         const std::vector<std::string>& classNames,
                         double fps = 0.0,
                         double inferMs = 0.0,
                         const std::string& extraInfo = "") {
    cv::Mat display = image.clone();
    int lineY = 25;
    const int step = 22;

    // ---- 状态信息（左上角） ----
    if (fps > 0) {
        char buf[64];
        snprintf(buf, sizeof(buf), "FPS: %.1f", fps);
        cv::putText(display, buf, cv::Point(10, lineY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, COLOR_INFO, 2);
        lineY += step;
    }
    if (inferMs > 0) {
        char buf[64];
        snprintf(buf, sizeof(buf), "Infer: %.1f ms", inferMs);
        cv::putText(display, buf, cv::Point(10, lineY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, COLOR_INFO, 2);
        lineY += step;
    }
    if (!extraInfo.empty()) {
        cv::putText(display, extraInfo, cv::Point(10, lineY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, COLOR_INFO, 1);
        lineY += step;
    }

    if (!detections.empty()) {
        char buf[64];
        snprintf(buf, sizeof(buf), "Targets: %zu", detections.size());
        cv::putText(display, buf, cv::Point(10, lineY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, COLOR_INFO, 2);
    } else {
        cv::putText(display, "NO RELIABLE TARGET", cv::Point(10, lineY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WARN, 2);
    }

    // ---- 逐个绘制检测目标 ----
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        const auto& box = det.bbox;
        const auto& center = det.center;

        // 边界框
        cv::rectangle(display, box, COLOR_BOX, 2);

        // 中心点准星：红点 + 黄圈 + 十字线
        cv::circle(display, center, 5, COLOR_CENTER, -1);
        cv::circle(display, center, 10, COLOR_RING, 2);
        cv::line(display, cv::Point(center.x - 12, center.y),
                 cv::Point(center.x + 12, center.y), COLOR_RING, 1);
        cv::line(display, cv::Point(center.x, center.y - 12),
                 cv::Point(center.x, center.y + 12), COLOR_RING, 1);

        // 类别 + 置信度标签
        std::string label;
        if (det.classId >= 0 && det.classId < (int)classNames.size())
            label = classNames[det.classId];
        else
            label = "cls_" + std::to_string(det.classId);

        char buf[128];
        snprintf(buf, sizeof(buf), "%s %.2f", label.c_str(), det.confidence);
        std::string text = buf;

        int baseline = 0;
        cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(display,
                      cv::Point(box.x, box.y - ts.height - 5),
                      cv::Point(box.x + ts.width, box.y),
                      COLOR_BOX, cv::FILLED);
        cv::putText(display, text, cv::Point(box.x, box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        // 中心坐标（交错偏移避免重叠）
        int ox = (i % 2 == 0) ? 15 : -80;
        int oy = (i < 2) ? -15 : 15;
        cv::putText(display,
                    "(" + std::to_string((int)center.x) + "," +
                    std::to_string((int)center.y) + ")",
                    cv::Point(center.x + ox, center.y + oy),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, COLOR_COORD, 1);

        // 目标编号
        cv::putText(display, "#" + std::to_string(i + 1),
                    cv::Point(center.x - 20, center.y - 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
    }

    return display;
}

} // namespace Visualizer

#endif // DETECTOR_H
