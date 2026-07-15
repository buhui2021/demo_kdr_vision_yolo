cmake_minimum_required(VERSION 3.15)
project(armor_detect VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================
# OpenCV 依赖
# ============================================================
find_package(OpenCV 4.5 REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# ============================================================
# 源文件
# ============================================================
set(SOURCES
    src/main.cpp
    src/detector.cpp
    src/target_selector.cpp
    src/visualizer.cpp
)

# ============================================================
# 构建目标
# ============================================================
add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -O2)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
