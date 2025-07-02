## Build docker image
```
cd cpp/
docker build -t ultralytics ./cpp/dockfile
```

## Enter the container
```
docker run -it -v ${PWD}:${PWD} ultralytics bash
```

### Install dependencies using conan
cd to `cpp/` inside the container
```
conan profile detect

conan install . --output-folder=build --build=missing -c tools.system.package_manager:mode=install
```


### Build
```
cd build
source conanbuild.sh
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF -DLINUX=TRUE

cmake --build . --config Release
```

## Run
cp the onnx model to the `cpp/`. cp the image to run inference on to `cpp/build/images`
```
./Yolov8OnnxRuntimeCPPInference
```