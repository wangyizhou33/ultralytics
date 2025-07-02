```
conan install . --output-folder=build --build=missing -c tools.system.package_manager:mode=install
```

```
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF -DLINUX=TRUE
```