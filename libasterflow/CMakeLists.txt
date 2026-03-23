cmake_minimum_required(VERSION 3.18)
project(asterflow_c LANGUAGES C)

set(CMAKE_C_STANDARD 11)

add_library(asterflow SHARED asterflow.c)
set_target_properties(asterflow PROPERTIES OUTPUT_NAME "asterflow")
target_compile_definitions(asterflow PRIVATE)

if(UNIX AND NOT APPLE)
  set_target_properties(asterflow PROPERTIES SUFFIX ".so")
endif()
