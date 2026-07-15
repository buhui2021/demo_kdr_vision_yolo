#include "visualizer.h"
#include <cstdio>

void Visualizer::drawDetections(cv::Mat& frame,
                                const std::vector<ArmorObject>& detections)
{
    for (const auto& det : detections) {
        const auto& b = det.bbox;

        // 绿色粗边界框
        cv::rectangle(frame, b, cv::Scalar(0, 255, 0), 3);

        // 标签背景
        char tag[64];
        std::snprintf(tag, sizeof(tag), "%.2f", det.confidence);
        std::string text = tag;
        if (det.class_id == 0) text = "Armor " + text;
        else text = "C" + std::to_string(det.class_id) + " " + text;

        int base = 0;
        cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                      0.6, 2, &base);
        cv::rectangle(frame,
                      cv::Point(b.x, b.y - ts.height - 8),
                      cv::Point(b.x + ts.width + 8, b.y),
                      cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(frame, text, cv::Point(b.x + 4, b.y - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    }
}

void Visualizer::drawCenters(cv::Mat& frame, const FrameResult& result)
{
    for (int i = 0; i < result.detected_count; ++i) {
        const cv::Point2f& c = result.centers[i];
        if (c.x < 1 && c.y < 1) continue;

        // 大红点 + 黄色大外圈 + 粗十字准星
        cv::circle(frame, c, 8, cv::Scalar(0, 0, 255), -1);       // 实心红点
        cv::circle(frame, c, 16, cv::Scalar(0, 255, 255), 3);     // 黄色外圈
        cv::line(frame, cv::Point(c.x - 18, c.y), cv::Point(c.x + 18, c.y),
                 cv::Scalar(0, 255, 255), 2);                      // 水平线
        cv::line(frame, cv::Point(c.x, c.y - 18), cv::Point(c.x, c.y + 18),
                 cv::Scalar(0, 255, 255), 2);                      // 垂直线

        // 编号（大号，半透明背景）
        char id_str[8];
        std::snprintf(id_str, sizeof(id_str), "#%d", i + 1);
        cv::putText(frame, id_str, cv::Point(c.x - 22, c.y - 22),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

        // 坐标（更大字号）
        char coord[32];
        std::snprintf(coord, sizeof(coord), "(%d,%d)", (int)c.x, (int)c.y);
        int ox = (i % 2 == 0) ? 22 : -100;
        int oy = (i < 2) ? -22 : 22;
        cv::putText(frame, coord, cv::Point(c.x + ox, c.y + oy),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 2);
    }
}

void Visualizer::drawHUD(cv::Mat& frame, float fps, int detected_count, float infer_ms)
{
    char line[128];

    // 左上角信息背景
    cv::rectangle(frame, cv::Point(5, 5), cv::Point(320, 115),
                  cv::Scalar(0, 0, 0), cv::FILLED);

    int y = 30;
    if (infer_ms > 0) {
        std::snprintf(line, sizeof(line), "Infer: %.0f ms", infer_ms);
        cv::putText(frame, line, cv::Point(15, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        y += 28;
    }

    if (fps > 0) {
        std::snprintf(line, sizeof(line), "FPS: %.1f", fps);
        cv::putText(frame, line, cv::Point(15, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        y += 28;
    }

    std::snprintf(line, sizeof(line), "Targets: %d", detected_count);
    cv::putText(frame, line, cv::Point(15, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                detected_count > 0 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);

    if (detected_count == 0) {
        y += 28;
        cv::putText(frame, "NO RELIABLE TARGET",
                    cv::Point(15, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 3);
    }
}
