cupiqp_add_test(test_cudss test_cudss.cu)

# Micro-benchmark (not a test): factor and solve timings, eager vs graph, on
# the MPC KKT shape. Run: <build>/tests/cpp/linalg/cudss/bench_cudss
add_executable(bench_cudss bench_cudss.cu)
target_link_libraries(bench_cudss PRIVATE cupiqp_native_internal)
set_target_properties(bench_cudss PROPERTIES BUILD_RPATH "${cudss_LIBRARY_DIR}")
