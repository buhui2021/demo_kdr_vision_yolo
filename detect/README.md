# RoboMaster 装甲板检测系统 — detect

## 项目功能

本工程是 RoboMaster 2027 视觉考核阶段二的核心部分。基于阶段一训练的 YOLO ONNX 模型，
使用 **C++ + OpenCV DNN** 完成 RoboMaster 机器人装甲板的目标检测、中心点提取与可视化。

### 功能特性

- ✅ 加载阶段一导出的 `.onnx` 模型进行推理
- ✅ 支持图片、视频、摄像头三种输入方式
- ✅ 装甲板边界框检测与置信度输出
- ✅ **所有检测目标的中心点提取与可视化**
- ✅ 多目标同时检测与独立中心点标注
- ✅ 无目标时的明确状态提示
- ✅ FPS / 推理耗时实时显示
- ✅ 结果保存到 `results/` 文件夹
- ✅ 命令行参数灵活配置

---

## 环境依赖

| 依赖 | 版本 | 说明 |
|------|------|------|
| CMake | ≥ 3.10 | 构建工具 |
| MinGW-w64 | ≥ 13 | C++ 编译器（GCC） |
| OpenCV | ≥ 4.8 | 计算机视觉库（需含 dnn 模块） |

### Windows 环境配置（使用 MSYS2）

1. **安装 MSYS2**
   ```
   winget install MSYS2.MSYS2
   ```

2. **安装 OpenCV（MinGW 版）**
   打开 MSYS2 MinGW64 终端，执行：
   ```bash
   pacman -Syu
   pacman -S mingw-w64-x86_64-opencv
   ```

3. **安装 CMake**
   ```
   winget install Kitware.CMake
   ```

4. **添加环境变量**
   将以下路径加入系统 PATH：
   - `C:\msys64\mingw64\bin`（OpenCV DLL、MinGW 编译器）
   - `C:\Program Files\CMake\bin`（CMake）

---

## 编译方式

```bash
# 进入 detect 目录
cd demo_kdr_vision_yolo/detect

# 创建构建目录
mkdir build && cd build

# 配置 CMake（指定 OpenCV 路径）
cmake .. -G "MinGW Makefiles" \
    -DOpenCV_DIR="C:/msys64/mingw64/lib/cmake/opencv4" \
    -DCMAKE_CXX_COMPILER="C:/msys64/mingw64/bin/g++.exe"

# 编译
mingw32-make -j$(nproc)
```

编译完成后，`build/` 目录下会生成 `detect.exe`。

---

## 运行方式

### 基本用法

```bash
# 摄像头实时检测（默认）
./detect.exe

# 检测单张图片
./detect.exe --input ../assets/test.jpg

# 检测视频文件
./detect.exe --input ../assets/test.mp4

# 指定模型路径和参数
./detect.exe --model ../models/best.onnx --input test.jpg --conf 0.5 --nms 0.45
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model <路径>` | ONNX 模型路径 | `../models/best.onnx` |
| `--input <路径>` | 输入图片/视频路径（空=摄像头） | 空 |
| `--conf <阈值>` | 置信度阈值 | `0.5` |
| `--nms <阈值>` | NMS 阈值 | `0.45` |
| `--size <像素>` | 模型输入尺寸 | `640` |
| `--no-display` | 不显示画面 | 显示 |
| `--no-save` | 不保存结果 | 保存 |
| `--help` | 显示帮助信息 | - |

---

## 模型来源

模型文件位于 `../models/best.onnx`，由阶段一训练得到：
- 训练脚本：`../yolo_train.py`
- 转换脚本：`../yolo_convert.py`
- 训练日志：`../runs/`

---

## 项目结构

```
demo_kdr_vision_yolo/
├── yolo_train.py               # 阶段一：训练脚本
├── yolo_convert.py             # 阶段一：模型转换脚本
├── models/                     # 模型文件
│   ├── best.pt                 # PyTorch 权重
│   └── best.onnx               # ONNX 导出模型（阶段二使用）
├── runs/                       # 训练日志
└── detect/                     # 阶段二：C++ 推理工程
    ├── CMakeLists.txt           # CMake 构建配置
    ├── README.md                # 本文件
    ├── include/
    │   └── detector.h           # 检测器头文件
    ├── src/
    │   ├── main.cpp             # 主程序（输入解析、可视化）
    │   └── detector.cpp         # 检测器实现（前/后处理）
    ├── assets/                  # 测试图片/视频（可选）
    └── results/                 # 运行结果截图
```

---

## 输入输出示例

### 输入
- 单张图片（jpg/png/bmp）
- 视频文件（mp4/avi）
- 摄像头实时画面

### 输出
- 检测结果图像/视频帧，包含：
  - 🟩 绿色边界框（检测到的装甲板）
  - 🔴 红色中心点
  - 🟡 黄色中心点标注圆圈
  - 类别名称 + 置信度
  - 中心点坐标
  - 实时 FPS / 推理耗时
  - 检测目标数量统计

---

## 已知问题

1. **模型类别名称**：`detector.cpp` 中的 `DEFAULT_CLASSES` 需要根据阶段一训练时的类别顺序手动修改。
2. **推理后端**：默认使用 CPU 推理；如需 CUDA 加速，需编译支持 CUDA 的 OpenCV。
3. **输入尺寸**：需与阶段一训练时的 `imgsz` 参数一致（默认 640）。
4. **测试图片**：`assets/` 目录为空，请自行放入测试图片或使用 `--input` 指定路径。

---

## 后续优化方向

- [ ] 添加装甲板灯条检测（RGB 阈值分割）
- [ ] 基于几何关系过滤误检框
- [ ] 添加 EKF 目标跟踪与预测（大二扩展）
- [ ] 添加 PnP 解算与云台角度输出
- [ ] 使用 TensorRT 加速推理
