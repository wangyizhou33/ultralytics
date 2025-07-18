## Download
Download tensorrt runtime prebuilt binary  
look for tensorrt github
https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.12.0/tars/TensorRT-10.12.0.36.Linux.x86_64-gnu.cuda-12.9.tar.gz


## Build docker image
```
cd cpp/
docker build -t tensorrt ./cpp-tensorrt/dockfile
```

## Enter the container
cd to `cpp-tensorrt/` in the host env

```
docker run -it -v ${PWD}:${PWD} --gpus all tensorrt bash
```

### Build
cd to `cpp-tensorrt/` inside the container

```
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Run
cp the tensorrt model to the `cpp-tensorrt/`. cp the image to run inference on to `cpp/build/images`
```
./Yolov8OnnxRuntimeCPPInference
```

## Note
1. host machine nvidia-driver 575 (NVIDIA-SMI 575.64.03  Driver Version: 575.64.03   CUDA Version: 12.9 )