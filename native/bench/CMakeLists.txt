# Native benchmarks (public C++ API; no Python).
add_executable(cupiqp_native_replay mpc_replay.cpp)
target_link_libraries(cupiqp_native_replay PRIVATE cupiqp_native)

add_executable(cupiqp_native_module_bench module_bench.cu)
target_link_libraries(cupiqp_native_module_bench PRIVATE cupiqp_native_internal)
target_compile_options(cupiqp_native_module_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# Reduced KKT backend microbenchmark (docs/modules/reduced_kkt_backend.md).
add_executable(cupiqp_native_reduced_kkt_bench reduced_kkt_bench.cu)
target_link_libraries(cupiqp_native_reduced_kkt_bench PRIVATE cupiqp_native_internal)
target_compile_options(cupiqp_native_reduced_kkt_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# Reduced KKT backend generality sweep and real-state accuracy check (docs/modules/ncu_kernels.md).
add_executable(cupiqp_native_reduced_kkt_sweep reduced_kkt_sweep.cu)
target_link_libraries(cupiqp_native_reduced_kkt_sweep PRIVATE cupiqp_native_internal)
target_compile_options(cupiqp_native_reduced_kkt_sweep PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
