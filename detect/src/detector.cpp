#include "detector.h"
#include <algorithm>
#include <iostream>
#include <fstream>

static const std::vector<std::string> DEFAULT_CLASSES = {"armor"};

bool ArmorDetector::init(const std::string& model_path,
                         float conf_threshold,
                         float nms_threshold,
                         const cv::Size& input_size)
{
    conf_threshold_ = conf_threshold;
    nms_threshold_ = nms_threshold;
    input_size_ = input_size;
    class_names_ = DEFAULT_CLASSES;

    std::ifstream f(model_path);
    if (!f.good()) {
        std::cerr << "[ArmorDetector] 模型文件不存在: " << model_path << std::endl;
        return false;
    }

    std::cout << "[ArmorDetector] 加载模型: " << model_path << std::endl;
    try {
        net_ = cv::dnn::readNetFromONNX(model_path);
    } catch (const cv::Exception& e) {
        std::cerr << "[ArmorDetector] 模型加载失败: " << e.what() << std::endl;
        return false;
    }

    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        std::cout << "[ArmorDetector] 使用 CUDA 推理" << std::endl;
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else {
        std::cout << "[ArmorDetector] 使用 CPU 推理" << std::endl;
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    std::cout << "[ArmorDetector] 初始化完成, 类别数: " << class_names_.size() << std::endl;
    return true;
}

bool ArmorDetector::detect(const cv::Mat& frame, std::vector<ArmorObject>& results)
{
    results.clear();
    if (frame.empty()) return false;

    cv::Mat blob = preprocess(frame);

    net_.setInput(blob);
    std::vector<cv::String> outNames = net_.getUnconnectedOutLayersNames();
    std::vector<cv::Mat> outputs;

    auto start = cv::getTickCount();
    net_.forward(outputs, outNames);
    auto end = cv::getTickCount();
    last_inference_ms_ = (end - start) / cv::getTickFrequency() * 1000.0;

    if (!outputs.empty()) {
        postprocess(outputs[0], frame.size(), results);
    }

    return !results.empty();
}

// ★ 前处理：底右填充（与 YOLO 训练时的 letterbox 一致）
cv::Mat ArmorDetector::preprocess(const cv::Mat& frame)
{
    int h = frame.rows, w = frame.cols;
    int target = input_size_.width;  // 正方形，如 640

    float scale = std::min((float)target / h, (float)target / w);
    int newW = (int)(w * scale);
    int newH = (int)(h * scale);

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(newW, newH));

    // 底右填充（YOLO 标准 letterbox，图像在左上角）
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded,
                       0, target - newH,   // top=0, bottom=padH
                       0, target - newW,   // left=0, right=padW
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::Mat blob;
    cv::dnn::blobFromImage(padded, blob, 1.0 / 255.0,
                           cv::Size(target, target),
                           cv::Scalar(), true, false);
    return blob;
}

// ★ 后处理：直接缩放（底右填充不需要偏移修正）
void ArmorDetector::postprocess(const cv::Mat& output,
                                const cv::Size& frame_size,
                                std::vector<ArmorObject>& results)
{
    int imgW = frame_size.width, imgH = frame_size.height;
    int target = input_size_.width;

    if (output.dims != 3 || output.size[0] != 1 || output.size[2] != 6) {
        std::cerr << "[ArmorDetector] 不支持的输出格式" << std::endl;
        return;
    }

    int maxDet = output.size[1];
    const float* data = output.ptr<float>();

    // 缩放因子（与 preprocess 一致）
    float scale = std::min((float)target / imgH, (float)target / imgW);

    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    for (int i = 0; i < maxDet; ++i) {
        const float* row = data + i * 6;
        float conf = row[4];
        if (conf < conf_threshold_) continue;

        float x1v = row[0], y1v = row[1];
        float x2v = row[2], y2v = row[3];
        int clsId = (int)row[5];

        float x1 = std::min(x1v, x2v);
        float y1 = std::min(y1v, y2v);
        float x2 = std::max(x1v, x2v);
        float y2 = std::max(y1v, y2v);

        // ★ 直接缩放到原图（底右填充，内容在左上角）
        float invScale = 1.0f / scale;
        int bx = (int)(x1 * invScale);
        int by = (int)(y1 * invScale);
        int bw = (int)((x2 - x1) * invScale);
        int bh = (int)((y2 - y1) * invScale);

        // 边界裁剪
        bx = std::max(0, std::min(bx, imgW - 1));
        by = std::max(0, std::min(by, imgH - 1));
        bw = std::min(bw, imgW - bx);
        bh = std::min(bh, imgH - by);
        if (bw <= 0 || bh <= 0) continue;

        boxes.push_back(cv::Rect(bx, by, bw, bh));
        confs.push_back(conf);
        classIds.push_back(clsId);
    }

    // NMS 去重
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, confs, 0.0f, nms_threshold_, keep);

    for (int idx : keep) {
        ArmorObject obj;
        obj.bbox = boxes[idx];
        obj.confidence = confs[idx];
        obj.class_id = classIds[idx];
        obj.center = cv::Point2f(boxes[idx].x + boxes[idx].width / 2.0f,
                                  boxes[idx].y + boxes[idx].height / 2.0f);
        results.push_back(obj);
    }
}
