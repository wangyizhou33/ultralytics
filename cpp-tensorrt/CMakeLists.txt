cmake_minimum_required(VERSION 3.5)

project(Yolov8CPPInference VERSION 0.1)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
set(CUDA_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so")
include_directories(${CUDA_INCLUDE_DIRS})
find_package(CUDA 11 REQUIRED)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# !CUDA

# OpenCV
set(OpenCV_DIR /nas_adas_data_platform/mnt/ofs-meta-data2/test/liwanjun/ultralytics/opencv_lib/lib/cmake/opencv4)
set(OpenCV_INCLUDE_DIRS /nas_adas_data_platform/mnt/ofs-meta-data2/test/liwanjun/ultralytics/opencv_lib/include)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# !OpenCV

set(PROJECT_SOURCES
    main.cpp

    inference.h
    inference.cpp
)

add_executable(Yolov8CPPInference ${PROJECT_SOURCES})
target_link_libraries(Yolov8CPPInference ${OpenCV_LIBS})

# ONNXRUNTIME
set(ONNXRUNTIME_VERSION 1.15.1)
set(ONNXRUNTIME_ROOT /nas_adas_data_platform/mnt/ofs-meta-data2/test/liwanjun/ultralytics/onnxruntime-linux-x64-1.22.0)
include_directories(${PROJECT_NAME} ${ONNXRUNTIME_ROOT}/include)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
if (USE_CUDA)
    target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
endif ()

# CUDNN
set(CUDNN_INCLUDE_DIR /usr/include)
set(CUDNN_LIBRARY_DIR /usr/lib/x86_64-linux-gnu)
include_directories(${CUDNN_INCLUDE_DIR})
link_directories(${CUDNN_LIBRARY_DIR})
target_link_libraries(${PROJECT_NAME} ${CUDNN_LIBRARY_DIR}/libcudnn.so.8)

# TENSORRT
set(TENSORRT_INCLUDE_DIR /nas_adas_data_platform/mnt/ofs-meta-data2/test/liwanjun/ultralytics/x86_64-linux-gnu/include)
set(TENSORRT_LIBRARY_DIR /nas_adas_data_platform/mnt/ofs-meta-data2/test/liwanjun/ultralytics/x86_64-linux-gnu/lib)
include_directories(${TENSORRT_INCLUDE_DIR})
link_directories(${TENSORRT_LIBRARY_DIR})
target_link_libraries(${PROJECT_NAME} ${TENSORRT_LIBRARY_DIR}/libnvinfer.so.8 ${TENSORRT_LIBRARY_DIR}/libnvinfer_plugin.so.8 cudart)