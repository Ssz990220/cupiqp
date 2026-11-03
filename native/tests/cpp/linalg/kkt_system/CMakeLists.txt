# kkt_system tests. The kernel / capture_if tests link without the dependency
# modules; the class-level tests need the spmv, kkt_assembly and cudss
# implementations and are only added once those sources exist (after merge).
# The implementation files may be .cu or .cpp (cudss uses cudss_solver.cpp).
cupiqp_add_test(test_kkt_system_kernels test_kkt_system_kernels.cu)

file(GLOB _kkt_dep_cudss "${CMAKE_SOURCE_DIR}/src/linalg/cudss/cudss_solver.cu"
                         "${CMAKE_SOURCE_DIR}/src/linalg/cudss/cudss_solver.cpp")
file(GLOB _kkt_dep_spmv "${CMAKE_SOURCE_DIR}/src/linalg/spmv/spmv.cu"
                        "${CMAKE_SOURCE_DIR}/src/linalg/spmv/spmv.cpp")
file(GLOB _kkt_dep_assembly "${CMAKE_SOURCE_DIR}/src/linalg/kkt_assembly/kkt_assembly.cu"
                            "${CMAKE_SOURCE_DIR}/src/linalg/kkt_assembly/kkt_assembly.cpp")
if(_kkt_dep_cudss AND _kkt_dep_spmv AND _kkt_dep_assembly)
  cupiqp_add_test(test_kkt_system_class test_kkt_system_class.cu)
else()
  message(STATUS "kkt_system: class-level tests skipped (spmv / kkt_assembly / cudss sources not present)")
endif()
