cmake_minimum_required(VERSION 3.26)
project(cupiqp_native VERSION 0.1.0 LANGUAGES CXX CUDA)

# ---------------------------------------------------------------- options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4070 SUPER
endif()

option(CUPIQP_BUILD_TESTS "Build C++ module tests (GoogleTest)" ON)
option(CUPIQP_BUILD_PYTHON "Build the nanobind module cupiqp_native._ext" ON)
option(CUPIQP_BUILD_BENCH "Build native benchmarks" ON)
set(CUPIQP_CUDSS_ROOT "$ENV{HOME}/Projects/.deps/cudss-0.8.0.10" CACHE PATH "cuDSS install root (redist layout)")

# ---------------------------------------------------------------- dependencies
find_package(CUDAToolkit 13.0 REQUIRED)
find_package(cudss 0.8 REQUIRED CONFIG PATHS "${CUPIQP_CUDSS_ROOT}/lib/cmake/cudss" NO_DEFAULT_PATH)
message(STATUS "cuDSS: ${cudss_LIBRARY_DIR}")

# ---------------------------------------------------------------- library
# Every module lists its sources in src/<group>/<module>/sources.cmake (or
# src/common/sources.cmake) by appending to CUPIQP_SOURCES (paths relative to
# native/), so modules never edit this file.
set(CUPIQP_SOURCES "")
file(GLOB_RECURSE CUPIQP_MODULE_SOURCE_LISTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/sources.cmake")
list(SORT CUPIQP_MODULE_SOURCE_LISTS)
foreach(f ${CUPIQP_MODULE_SOURCE_LISTS})
  include(${f})
endforeach()
list(TRANSFORM CUPIQP_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

add_library(cupiqp_native STATIC ${CUPIQP_SOURCES})
target_include_directories(cupiqp_native
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(cupiqp_native PUBLIC CUDA::cudart CUDA::cusparse cudss)
target_compile_options(cupiqp_native PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo -Xcompiler=-Wall>
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
set_target_properties(cupiqp_native PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
  BUILD_RPATH "${cudss_LIBRARY_DIR}" INSTALL_RPATH "${cudss_LIBRARY_DIR}")
target_compile_definitions(cupiqp_native PUBLIC
  CUPIQP_CUDSS_MT_LIB="${cudss_LIBRARY_DIR}/libcudss_mtlayer_gomp.so.0")

# Internal headers for tests, the binding and benchmarks.
add_library(cupiqp_native_internal INTERFACE)
target_include_directories(cupiqp_native_internal INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(cupiqp_native_internal INTERFACE cupiqp_native)

# ---------------------------------------------------------------- tests
if(CUPIQP_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  include(GoogleTest)
  # Each module owns tests/cpp/<group>/<module>/CMakeLists.txt (mirroring src/)
  # and calls cupiqp_add_test.
  function(cupiqp_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE cupiqp_native_internal GTest::gtest_main)
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set_target_properties(${name} PROPERTIES BUILD_RPATH "${cudss_LIBRARY_DIR}")
    gtest_discover_tests(${name} DISCOVERY_TIMEOUT 120)
  endfunction()
  file(GLOB_RECURSE CUPIQP_TEST_LISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/CMakeLists.txt")
  list(SORT CUPIQP_TEST_LISTS)
  foreach(f ${CUPIQP_TEST_LISTS})
    get_filename_component(d ${f} DIRECTORY)
    file(RELATIVE_PATH rel "${CMAKE_CURRENT_SOURCE_DIR}" "${d}")
    add_subdirectory(${d} "${CMAKE_BINARY_DIR}/${rel}")
  endforeach()
endif()

# ---------------------------------------------------------------- python
if(CUPIQP_BUILD_PYTHON)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)
  file(GLOB CUPIQP_BIND_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/python/bindings/*.cpp"
                                                  "${CMAKE_CURRENT_SOURCE_DIR}/python/bindings/*.cu")
  nanobind_add_module(_ext NB_STATIC ${CUPIQP_BIND_SOURCES})
  target_link_libraries(_ext PRIVATE cupiqp_native_internal)
  set_target_properties(_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/cupiqp_native"
    BUILD_RPATH "${cudss_LIBRARY_DIR}")
  # Copy the pure-Python package next to the extension.
  file(GLOB CUPIQP_PY_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/python/cupiqp_native/*.py")
  add_custom_target(cupiqp_native_py ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/python/cupiqp_native"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CUPIQP_PY_FILES} "${CMAKE_BINARY_DIR}/python/cupiqp_native/"
    DEPENDS ${CUPIQP_PY_FILES})
  add_dependencies(_ext cupiqp_native_py)
endif()

# ---------------------------------------------------------------- benchmarks
if(CUPIQP_BUILD_BENCH AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/CMakeLists.txt")
  add_subdirectory(bench)
endif()
