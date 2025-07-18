cmake_minimum_required(VERSION 3.5)

project(Yolov8CPPInference VERSION 0.1)

cmake_policy(SET CMP0146 OLD)  # to find cuda correctly
cmake_policy(SET CMP0074 NEW)  # to find trt::nvinfer correctly

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(CUDA 12 REQUIRED)

# OpenCV
set(OpenCV_DIR /usr/lib/x86_64-linux-gnu/cmake/opencv4)
find_package(OpenCV REQUIRED)


set(TensorRT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/TensorRT-10.12.0.36/")
find_package(TensorRT REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TensorRT_INCLUDE_DIRS})
# !OpenCV

set(PROJECT_SOURCES
    main.cpp

    inference.h
    inference.cpp
)

include_directories(${CUDA_TOOLKIT_INCLUDE})

add_executable(Yolov8CPPInference ${PROJECT_SOURCES})
target_link_libraries(Yolov8CPPInference ${OpenCV_LIBS} ${CUDA_LIBRARIES} trt::nvinfer)

# # ONNXRUNTIME
# set(ONNXRUNTIME_VERSION 1.15.1)
# set(ONNXRUNTIME_ROOT /nas_adas_data_platform/mnt/ofs-meta-data2/test/liwanjun/ultralytics/onnxruntime-linux-x64-1.22.0)
# include_directories(${PROJECT_NAME} ${ONNXRUNTIME_ROOT}/include)
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
# if (USE_CUDA)
#     target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
# endif ()

# # CUDNN
# set(CUDNN_INCLUDE_DIR /usr/include)
# set(CUDNN_LIBRARY_DIR /usr/lib/x86_64-linux-gnu)
# include_directories(${CUDNN_INCLUDE_DIR})
# link_directories(${CUDNN_LIBRARY_DIR})
# target_link_libraries(${PROJECT_NAME} ${CUDNN_LIBRARY_DIR}/libcudnn.so.8)

# # TENSORRT
# set(TENSORRT_INCLUDE_DIR /nas_adas_data_platform/mnt/ofs-meta-data2/test/liwanjun/ultralytics/x86_64-linux-gnu/include)
# set(TENSORRT_LIBRARY_DIR /nas_adas_data_platform/mnt/ofs-meta-data2/test/liwanjun/ultralytics/x86_64-linux-gnu/lib)
# include_directories(${TENSORRT_INCLUDE_DIR})
# link_directories(${TENSORRT_LIBRARY_DIR})
# target_link_libraries(${PROJECT_NAME} ${TENSORRT_LIBRARY_DIR}/libnvinfer.so.8 ${TENSORRT_LIBRARY_DIR}/libnvinfer_plugin.so.8 cudart)
