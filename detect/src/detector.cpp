#include "detector.h"
#include <algorithm>

// ==================== 默认类别名称 ====================
// 可通过 setClassNames() 在运行前修改
static const std::vector<std::string> DEFAULT_CLASSES = {
    "bule1","bule3","bule","red","red1",
};

// ==================== 模块2: 模型推理实现 ====================

Detector::Detector(const std::string& modelPath,
                   int inputSize,
                   float confThresh)
    : inputSize_(inputSize)
    , confThreshold_(confThresh)
    , classNames_(DEFAULT_CLASSES)
{
    // ---- P2 修复: 输入验证 ----
    if (!modelExists(modelPath)) {
        std::cerr << "[Detector] 模型文件不存在: " << modelPath << std::endl;
        throw std::runtime_error("Model file not found: " + modelPath);
    }

    std::cout << "[Detector] 加载模型: " << modelPath << std::endl;
    try {
        net_ = cv::dnn::readNetFromONNX(modelPath);
    } catch (const cv::Exception& e) {
        std::cerr << "[Detector] 模型加载失败: " << e.what() << std::endl;
        throw;
    }

    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        std::cout << "[Detector] 使用 CUDA 推理" << std::endl;
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else {
        std::cout << "[Detector] 使用 CPU 推理" << std::endl;
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    printModelInfo();
}

void Detector::printModelInfo() const
{
    std::cout << "\n========== 模型信息 ==========" << std::endl;
    std::vector<cv::String> outNames = net_.getUnconnectedOutLayersNames();
    std::cout << "输出层数: " << outNames.size() << std::endl;
    for (size_t i = 0; i < outNames.size(); ++i)
        std::cout << "  输出层 " << i << ": " << outNames[i] << std::endl;
    std::cout << "类别数: " << classNames_.size() << std::endl;
    std::cout << "==============================\n" << std::endl;
}

// ---- P0 修复: 居中 Letterbox ----
void Detector::preprocess(const cv::Mat& image, cv::Mat& blob,
                           float& scaleFactor, cv::Size& pad)
{
    int h = image.rows, w = image.cols;
    scaleFactor = std::min((float)inputSize_ / h, (float)inputSize_ / w);

    int newW = (int)(w * scaleFactor);
    int newH = (int)(h * scaleFactor);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(newW, newH));

    // 居中填充（标准 letterbox）
    int padW = inputSize_ - newW;
    int padH = inputSize_ - newH;
    int top = padH / 2;
    int bottom = padH - top;
    int left = padW / 2;
    int right = padW - left;
    pad = cv::Size(left, top);  // 记录左上填充量

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::dnn::blobFromImage(padded, blob, 1.0 / 255.0,
                           cv::Size(inputSize_, inputSize_),
                           cv::Scalar(), true, false);
}

// ---- P0 修复: 后处理（去填充偏移）+ P3 修复: 多格式支持 ----
std::vector<Detection> Detector::postprocess(const std::vector<cv::Mat>& outputs,
                                              float scaleFactor,
                                              const cv::Size& pad,
                                              const cv::Size& imgSize)
{
    std::vector<Detection> detections;
    if (outputs.empty()) return detections;

    const cv::Mat& out = outputs[0];
    int imgW = imgSize.width, imgH = imgSize.height;
    int padX = pad.width, padY = pad.height;

    // ---- P3 修复: 多种输出格式兼容 ----
    if (out.dims == 3 && out.size[0] == 1 && out.size[2] == 6) {
        // 格式 (1, N, 6): [x1, y1, x2, y2, conf, cls]
        int maxDet = out.size[1];
        const float* data = out.ptr<float>();

        for (int i = 0; i < maxDet; ++i) {
            const float* row = data + i * 6;
            float conf = row[4];
            if (conf < confThreshold_) continue;

            float x1v = row[0], y1v = row[1];
            float x2v = row[2], y2v = row[3];
            int clsId = (int)row[5];

            float x1 = std::min(x1v, x2v);
            float y1 = std::min(y1v, y2v);
            float x2 = std::max(x1v, x2v);
            float y2 = std::max(y1v, y2v);

            // ---- P0 修复: 减去填充偏移，再缩放到原图 ----
            float invScale = 1.0f / scaleFactor;
            int bx = (int)((x1 - padX) * invScale);
            int by = (int)((y1 - padY) * invScale);
            int bw = (int)((x2 - x1) * invScale);
            int bh = (int)((y2 - y1) * invScale);

            // 裁剪到图像边界
            bx = std::max(0, std::min(bx, imgW - 1));
            by = std::max(0, std::min(by, imgH - 1));
            bw = std::min(bw, imgW - bx);
            bh = std::min(bh, imgH - by);

            if (bw <= 0 || bh <= 0) continue;

            Detection det;
            det.bbox = cv::Rect(bx, by, bw, bh);
            det.confidence = conf;
            det.classId = clsId;
            det.center = cv::Point2f(bx + bw / 2.0f, by + bh / 2.0f);
            detections.push_back(det);
        }
    }
    else if (out.dims == 3 && out.size[1] == 6) {
        // 格式 (1, 6, N): 转置后的版本
        int numDet = out.size[2];
        const float* data = out.ptr<float>();
        for (int i = 0; i < numDet; ++i) {
            float conf = data[4 * numDet + i];
            if (conf < confThreshold_) continue;
            float x1v = data[0 * numDet + i];
            float y1v = data[1 * numDet + i];
            float x2v = data[2 * numDet + i];
            float y2v = data[3 * numDet + i];
            int clsId = (int)data[5 * numDet + i];

            float invScale = 1.0f / scaleFactor;
            int bx = (int)((std::min(x1v, x2v) - padX) * invScale);
            int by = (int)((std::min(y1v, y2v) - padY) * invScale);
            int bw = (int)(std::abs(x2v - x1v) * invScale);
            int bh = (int)(std::abs(y2v - y1v) * invScale);
            bx = std::max(0, std::min(bx, imgW - 1));
            by = std::max(0, std::min(by, imgH - 1));
            bw = std::min(bw, imgW - bx);
            bh = std::min(bh, imgH - by);
            if (bw <= 0 || bh <= 0) continue;
            Detection det;
            det.bbox = cv::Rect(bx, by, bw, bh);
            det.confidence = conf;
            det.classId = clsId;
            det.center = cv::Point2f(bx + bw / 2.0f, by + bh / 2.0f);
            detections.push_back(det);
        }
    }
    else {
        std::cerr << "[Detector] 未支持的输出格式! dims=" << out.dims;
        if (out.dims > 0) {
            std::cerr << " [";
            for (int d = 0; d < out.dims; ++d)
                std::cerr << (d ? "," : "") << out.size[d];
            std::cerr << "]";
        }
        std::cerr << std::endl;
    }

    return detections;
}

std::vector<Detection> Detector::detect(const cv::Mat& image)
{
    cv::Mat blob;
    float scaleFactor;
    cv::Size pad;

    // ---- 前处理 ----
    preprocess(image, blob, scaleFactor, pad);

    // ---- 推理 ----
    net_.setInput(blob);
    std::vector<cv::String> outNames = net_.getUnconnectedOutLayersNames();
    std::vector<cv::Mat> outputs;

    auto start = cv::getTickCount();
    net_.forward(outputs, outNames);
    auto end = cv::getTickCount();
    lastInferenceMs_ = (end - start) / cv::getTickFrequency() * 1000.0;

    // ---- 后处理 ----
    return postprocess(outputs, scaleFactor, pad, image.size());
}
