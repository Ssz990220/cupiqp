# kkt_system tests: kernels / capture_if, and the class-level tests on top of
# spmv, kkt_assembly and reduced_kkt.
cupiqp_add_test(test_kkt_system_kernels test_kkt_system_kernels.cu)
cupiqp_add_test(test_kkt_system_class test_kkt_system_class.cu)
