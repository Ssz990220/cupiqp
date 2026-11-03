# Native benchmarks (public C++ API; no Python).
add_executable(cupiqp_native_replay mpc_replay.cpp)
target_link_libraries(cupiqp_native_replay PRIVATE cupiqp_native)
set_target_properties(cupiqp_native_replay PROPERTIES BUILD_RPATH "${cudss_LIBRARY_DIR}")

add_executable(cupiqp_native_module_bench module_bench.cu)
target_link_libraries(cupiqp_native_module_bench PRIVATE cupiqp_native_internal)
target_compile_options(cupiqp_native_module_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(cupiqp_native_module_bench PROPERTIES BUILD_RPATH "${cudss_LIBRARY_DIR}")

# Reduced (Schur-complement) KKT feasibility probe (docs/briefs/reduced_kkt.md).
add_executable(cupiqp_native_reduced_kkt_probe reduced_kkt_probe.cu)
target_link_libraries(cupiqp_native_reduced_kkt_probe PRIVATE cupiqp_native_internal CUDA::cublas CUDA::cusolver)
set_target_properties(cupiqp_native_reduced_kkt_probe PROPERTIES BUILD_RPATH "${cudss_LIBRARY_DIR}")
