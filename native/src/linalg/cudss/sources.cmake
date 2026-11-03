# Sources of module cudss (owned by that module). Paths relative to native/.
list(APPEND CUPIQP_SOURCES
  src/linalg/cudss/cudss_memory.cpp
  src/linalg/cudss/cudss_solver.cpp
  src/linalg/cudss/pageable_memcpy_patch.cpp)
