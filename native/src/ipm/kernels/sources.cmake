# Sources of module ipm_kernels (owned by that module). Paths relative to native/.
list(APPEND CUPIQP_SOURCES
  src/ipm/kernels/ipm_step_kernels.cu
  src/ipm/kernels/ipm_residual_kernels.cu
  src/ipm/kernels/ipm_control_kernels.cu)
