# Build C++ library and Python wheel for Apple Silicon 
```shell
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --skip_tests --build_wheel --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=arm64 && 
pip install  --force-reinstall build/MacOS/RelWithDebInfo/dist/*.whl
```

# Install Python wheel
```shell
pip install  --force-reinstall build/MacOS/RelWithDebInfo/dist/*.whl
```
