cupiqp_add_test(test_spmv test_spmv.cu)

# Benchmark (not a ctest): ./bench_spmv [reps] [batch]
add_executable(bench_spmv bench_spmv.cu)
target_link_libraries(bench_spmv PRIVATE cupiqp_native_internal)
target_compile_options(bench_spmv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(bench_spmv PROPERTIES BUILD_RPATH "${cudss_LIBRARY_DIR}")
