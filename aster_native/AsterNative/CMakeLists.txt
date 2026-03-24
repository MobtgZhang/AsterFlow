set(AF_NATIVE_SOURCES
  Context.cpp
  Device.cpp
  core/Storage.cpp
  core/TensorImpl.cpp
  accelerator/Accelerator.cpp
  native/cpu/f32_kernels.cpp
)

add_library(af_native_tensor STATIC ${AF_NATIVE_SOURCES})

target_include_directories(af_native_tensor PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  "${CMAKE_CURRENT_SOURCE_DIR}/../include"
)

target_compile_definitions(af_native_tensor PUBLIC
  ASTERFLOW_NATIVE_BUILDING=1
)

add_library(asterflow_native SHARED
  c_api/native_c_api.cpp
)

target_include_directories(asterflow_native PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  "${CMAKE_CURRENT_SOURCE_DIR}/../include"
)

target_link_libraries(asterflow_native PRIVATE af_native_tensor)

set_target_properties(asterflow_native PROPERTIES OUTPUT_NAME "asterflow_native")

if(MSVC)
  set_target_properties(asterflow_native PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
