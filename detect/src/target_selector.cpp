#include "target_selector.h"
#include <algorithm>
#include <cmath>

void TargetSelector::init(const cv::Size& frame_size)
{
    frame_size_ = frame_size;
}

FrameResult TargetSelector::update(const std::vector<ArmorObject>& detections)
{
    FrameResult result;

    // 从检测结果中提取中心点，最多 MAX_ARMOR_COUNT 个
    int count = std::min((int)detections.size(), MAX_ARMOR_COUNT);
    result.detected_count = count;

    for (int i = 0; i < count; ++i) {
        result.centers[i] = detections[i].center;
    }

    // 不足 4 个的填 (0,0)，已在构造函数中初始化

    return result;
}
