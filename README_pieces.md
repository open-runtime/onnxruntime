[submodule "cmake/external/onnx"]
path = cmake/external/onnx
url = https://github.com/onnx/onnx.:git

# Build

```shell
./build.sh --config Release \
--build_shared_lib \
--parallel \
--compile_no_warning_as_error \
--skip_submodule_sync \
--skip_tests \
--cmake_extra_defines CMAKE_OSX_ARCHITECTURES=arm64 \
FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER
```