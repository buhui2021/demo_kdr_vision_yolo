cmake_minimum_required(VERSION 3.10)
project(RoboMasterDetect LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# OpenCV 路径
set(OpenCV_DIR "C:/msys64/mingw64/lib/cmake/opencv4" CACHE PATH "Path to OpenCV CMake config")

find_package(OpenCV REQUIRED COMPONENTS core dnn imgproc imgcodecs videoio)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(detect
    src/main.cpp
    src/detector.cpp
)

target_include_directories(detect PRIVATE include)

target_link_libraries(detect ${OpenCV_LIBS})

# 复制模型和测试资源
file(COPY ${CMAKE_SOURCE_DIR}/../models DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/assets DESTINATION ${CMAKE_BINARY_DIR})
