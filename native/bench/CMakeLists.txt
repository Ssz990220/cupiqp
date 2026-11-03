# Native benchmarks (public C++ API; no Python).
add_executable(cupiqp_native_replay mpc_replay.cpp)
target_link_libraries(cupiqp_native_replay PRIVATE cupiqp_native)
set_target_properties(cupiqp_native_replay PROPERTIES BUILD_RPATH "${cudss_LIBRARY_DIR}")
