# Sources of module kkt_system (owned by that module). Paths relative to native/.
# kkt_system.cu references the spmv / kkt_assembly / cudss implementations; it
# sits in the static library and is only pulled into an executable that uses
# KktSystem, so the kernel tests and the binding link without them.
list(APPEND CUPIQP_SOURCES
  src/linalg/kkt_system/kkt_system_kernels.cu
  src/linalg/kkt_system/kkt_system.cu)
