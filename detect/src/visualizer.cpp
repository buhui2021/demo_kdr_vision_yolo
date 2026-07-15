#include "visualizer.h"
#include <cstdio>

void Visualizer::drawDetections(cv::Mat& frame,
                                const std::vector<ArmorObject>& detections)
{
    for (const auto& det : detections) {
        // 边界框
        cv::rectangle(frame, det.bbox, cv::Scalar(0, 255, 0), 2);

        // 类别 + 置信度
        char buf[128];
        snprintf(buf, sizeof(buf), "%.2f", det.confidence);
        std::string text = buf;
        if (!det.class_id) {
            text = "armor " + text;
        } else {
            text = "cls_" + std::to_string(det.class_id) + " " + text;
        }

        int baseline = 0;
        cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                      0.5, 1, &baseline);
        cv::rectangle(frame,
                      cv::Point(det.bbox.x, det.bbox.y - ts.height - 5),
                      cv::Point(det.bbox.x + ts.width, det.bbox.y),
                      cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(frame, text, cv::Point(det.bbox.x, det.bbox.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

void Visualizer::drawCenters(cv::Mat& frame, const FrameResult& result)
{
    for (int i = 0; i < result.detected_count; ++i) {
        const cv::Point2f& c = result.centers[i];
        if (c.x == 0 && c.y == 0) continue;

        // 红点 + 黄圈 + 十字准星
        cv::circle(frame, c, 5, cv::Scalar(0, 0, 255), -1);
        cv::circle(frame, c, 10, cv::Scalar(0, 255, 255), 2);
        cv::line(frame, cv::Point(c.x - 12, c.y), cv::Point(c.x + 12, c.y),
                 cv::Scalar(0, 255, 255), 1);
        cv::line(frame, cv::Point(c.x, c.y - 12), cv::Point(c.x, c.y + 12),
                 cv::Scalar(0, 255, 255), 1);

        // 中心坐标
        std::string text = "(" + std::to_string((int)c.x) + "," +
                           std::to_string((int)c.y) + ")";
        int ox = (i % 2 == 0) ? 15 : -80;
        int oy = (i < 2) ? -15 : 15;
        cv::putText(frame, text, cv::Point(c.x + ox, c.y + oy),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1);

        // 编号
        cv::putText(frame, "#" + std::to_string(i + 1),
                    cv::Point(c.x - 20, c.y - 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
    }
}

void Visualizer::drawHUD(cv::Mat& frame, float fps, int detected_count)
{
    char buf[128];
    int y = 25;

    if (fps > 0) {
        snprintf(buf, sizeof(buf), "FPS: %.1f", fps);
        cv::putText(frame, buf, cv::Point(10, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2);
        y += 22;
    }

    snprintf(buf, sizeof(buf), "Targets: %d", detected_count);
    cv::putText(frame, buf, cv::Point(10, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2);
    y += 22;

    if (detected_count == 0) {
        cv::putText(frame, "NO RELIABLE TARGET", cv::Point(10, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    }
}
