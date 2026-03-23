add_library(af_native_tensor STATIC
  Context.cpp
  Device.cpp
  core/TensorImpl.cpp
  accelerator/Accelerator.cpp
  native/native_stub.cpp
)

target_include_directories(af_native_tensor PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

target_compile_definitions(af_native_tensor PUBLIC
  ASTERFLOW_NATIVE_BUILDING=1
)
