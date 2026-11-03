// Feasibility PoC: cuDSS 0.8 native C API, CUDA graph capture and conditional
// (WHILE) node capture of each cuDSS phase. Mirrors the cuDSS configuration of
// the pinned cuPIQP sparse backend (cupiqp/sparse/sparse_direct_solver.py).
//
// Build: see native/poc/README.md. Prints one line per experiment.
#include <cuda_runtime.h>
#include <cudss.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  std::printf("CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1);} } while (0)
#define CS(x) do { cudssStatus_t s_ = (x); if (s_ != CUDSS_STATUS_SUCCESS) { \
  std::printf("cuDSS error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); std::exit(1);} } while (0)

static int pool_alloc(void* ctx, void** ptr, size_t size, cudaStream_t st) {
  return cudaMallocFromPoolAsync(ptr, size, (cudaMemPool_t)ctx, st) == cudaSuccess ? 0 : 1;
}
static int pool_free(void*, void* ptr, size_t, cudaStream_t st) {
  return cudaFreeAsync(ptr, st) == cudaSuccess ? 0 : 1;
}

// Quasi-definite KKT [P+I  ; A  -d I] lower triangle stored with full P block,
// like the reference (P full, A in the lower-left block, diagonal placeholders).
struct Kkt { int n, p, dim; std::vector<int> indptr, indices; std::vector<double> val; };

static Kkt make_kkt(int n, int p) {
  Kkt k; k.n = n; k.p = p; k.dim = n + p; k.indptr.push_back(0);
  for (int r = 0; r < n; ++r) {               // tridiagonal SPD P + I, full storage
    for (int c = r - 1; c <= r + 1; ++c) if (c >= 0 && c < n) {
      k.indices.push_back(c); k.val.push_back(c == r ? 4.0 : -1.0);
    }
    k.indptr.push_back((int)k.indices.size());
  }
  for (int r = 0; r < p; ++r) {               // A row r touches columns r, r+1; then -delta diag
    k.indices.push_back(r); k.val.push_back(1.0);
    k.indices.push_back(r + 1); k.val.push_back(0.5);
    k.indices.push_back(n + r); k.val.push_back(-1e-4);
    k.indptr.push_back((int)k.indices.size());
  }
  return k;
}

__global__ void set_counter(int* c, int v) { *c = v; }
__global__ void dec_counter(int* c, cudaGraphConditionalHandle h) {
  int v = --(*c); cudaGraphSetConditional(h, v > 0 ? 1u : 0u);
}

struct Ctx {
  cudssHandle_t h; cudssConfig_t cfg; cudssData_t data;
  cudssMatrix_t A, x, b; double *dval, *drhs, *dsol; int *dptr, *didx; Kkt k;
};

static void setup(Ctx& c, cudaStream_t st, bool mt) {
  c.k = make_kkt(400, 100);
  int dim = c.k.dim, nnz = (int)c.k.val.size();
  CK(cudaMalloc(&c.dptr, sizeof(int) * (dim + 1)));
  CK(cudaMalloc(&c.didx, sizeof(int) * nnz));
  CK(cudaMalloc(&c.dval, sizeof(double) * nnz));
  CK(cudaMalloc(&c.drhs, sizeof(double) * dim));
  CK(cudaMalloc(&c.dsol, sizeof(double) * dim));
  CK(cudaMemcpy(c.dptr, c.k.indptr.data(), sizeof(int) * (dim + 1), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(c.didx, c.k.indices.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(c.dval, c.k.val.data(), sizeof(double) * nnz, cudaMemcpyHostToDevice));
  std::vector<double> rhs(dim, 1.0);
  CK(cudaMemcpy(c.drhs, rhs.data(), sizeof(double) * dim, cudaMemcpyHostToDevice));
  CS(cudssCreate(&c.h));
  CS(cudssSetStream(c.h, st));
  if (mt) {
    const char* lib = std::getenv("CUDSS_THREADING_LIB");
    if (lib) CS(cudssSetThreadingLayer(c.h, lib));
  }
  cudaMemPool_t pool; CK(cudaDeviceGetDefaultMemPool(&pool, 0));
  cudssDeviceMemHandler_t mh{}; mh.ctx = pool; mh.device_alloc = pool_alloc; mh.device_free = pool_free;
  std::snprintf(mh.name, CUDSS_ALLOCATOR_NAME_LEN, "default_pool");
  CS(cudssSetDeviceMemHandler(c.h, &mh));
  CS(cudssConfigCreate(&c.cfg));
  int zero = 0, one = 1; cudssReorderingAlg_t alg = CUDSS_REORDERING_ALG_DEFAULT;
  CS(cudssConfigSet(c.cfg, CUDSS_CONFIG_HYBRID_MEMORY_MODE, &zero, sizeof(int)));
  CS(cudssConfigSet(c.cfg, CUDSS_CONFIG_USE_CUDA_REGISTER_MEMORY, &one, sizeof(int)));
  CS(cudssConfigSet(c.cfg, CUDSS_CONFIG_REORDERING_ALG, &alg, sizeof(alg)));
  CS(cudssConfigSet(c.cfg, CUDSS_CONFIG_USE_SUPERPANELS, &zero, sizeof(int)));
  CS(cudssConfigSet(c.cfg, CUDSS_CONFIG_IR_N_STEPS, &zero, sizeof(int)));
  CS(cudssDataCreate(c.h, &c.data));
  CS(cudssMatrixCreateCsr(&c.A, dim, dim, nnz, c.dptr, nullptr, c.didx, c.dval, CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
                          CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO));
  CS(cudssMatrixCreateDn(&c.b, dim, 1, dim, c.drhs, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
  CS(cudssMatrixCreateDn(&c.x, dim, 1, dim, c.dsol, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
  CS(cudssExecute(c.h, CUDSS_PHASE_ANALYSIS, c.cfg, c.data, c.A, c.x, c.b));
  CS(cudssExecute(c.h, CUDSS_PHASE_FACTORIZATION, c.cfg, c.data, c.A, c.x, c.b));
  CS(cudssExecute(c.h, CUDSS_PHASE_SOLVE, c.cfg, c.data, c.A, c.x, c.b));
  CK(cudaStreamSynchronize(st));
}

// Residual ||K x - b||_inf on the host (K symmetric, lower stored + full P block).
static double residual(Ctx& c) {
  int dim = c.k.dim; std::vector<double> x(dim), r(dim, -1.0);
  CK(cudaMemcpy(x.data(), c.dsol, sizeof(double) * dim, cudaMemcpyDeviceToHost));
  for (int row = 0; row < dim; ++row)
    for (int q = c.k.indptr[row]; q < c.k.indptr[row + 1]; ++q) {
      int col = c.k.indices[q]; double v = c.k.val[q];
      if (col > row) continue;                    // lower view: upper entries ignored
      r[row] += v * x[col]; if (col != row) r[col] += v * x[row];
    }
  double m = 0; for (double v : r) m = std::fmax(m, std::fabs(v)); return m;
}

// Capture `phase` once into a plain graph; return status text and residual.
static std::string try_plain(Ctx& c, cudaStream_t st, int phase) {
  cudaGraph_t g; cudaGraphExec_t ge;
  CK(cudaMemsetAsync(c.dsol, 0, sizeof(double) * c.k.dim, st));
  CK(cudaStreamSynchronize(st));
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  cudssStatus_t s = cudssExecute(c.h, phase, c.cfg, c.data, c.A, c.x, c.b);
  cudaError_t e = cudaStreamEndCapture(st, &g);
  if (s != CUDSS_STATUS_SUCCESS || e != cudaSuccess) {
    cudaGetLastError();
    return "FAIL capture (cudss=" + std::to_string((int)s) + ", cuda=" + cudaGetErrorName(e) + ")";
  }
  e = cudaGraphInstantiate(&ge, g, 0);
  if (e != cudaSuccess) { cudaGetLastError(); return std::string("FAIL instantiate ") + cudaGetErrorName(e); }
  for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(ge, st));
  CK(cudaStreamSynchronize(st));
  cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
  return "OK";
}

// Capture `phase` into the body of a conditional WHILE node that runs `iters` times.
static std::string try_while(Ctx& c, cudaStream_t st, int phase, int iters) {
  int* counter; CK(cudaMalloc(&counter, sizeof(int)));
  cudaGraph_t g; CK(cudaGraphCreate(&g, 0));
  cudaGraphConditionalHandle h;
  CK(cudaGraphConditionalHandleCreate(&h, g, 1, cudaGraphCondAssignDefault));
  cudaGraphNodeParams np{}; np.type = cudaGraphNodeTypeConditional;
  np.conditional.handle = h; np.conditional.type = cudaGraphCondTypeWhile; np.conditional.size = 1;
  cudaGraphNode_t node; CK(cudaGraphAddNode(&node, g, nullptr, nullptr, 0, &np));
  cudaGraph_t body = np.conditional.phGraph_out[0];
  CK(cudaStreamBeginCaptureToGraph(st, body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
  cudssStatus_t s = cudssExecute(c.h, phase, c.cfg, c.data, c.A, c.x, c.b);
  dec_counter<<<1, 1, 0, st>>>(counter, h);
  cudaGraph_t out; cudaError_t e = cudaStreamEndCapture(st, &out);
  std::string res;
  if (s != CUDSS_STATUS_SUCCESS || e != cudaSuccess) {
    cudaGetLastError();
    res = "FAIL capture (cudss=" + std::to_string((int)s) + ", cuda=" + cudaGetErrorName(e) + ")";
  } else {
    cudaGraphExec_t ge; e = cudaGraphInstantiate(&ge, g, 0);
    if (e != cudaSuccess) { cudaGetLastError(); res = std::string("FAIL instantiate ") + cudaGetErrorName(e); }
    else {
      set_counter<<<1, 1, 0, st>>>(counter, iters);
      CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));
      cudaGraphExecDestroy(ge); res = "OK";
    }
  }
  cudaGraphDestroy(g); cudaFree(counter);
  return res;
}

// Replace every pageable-source memcpy node of `g` (recursively not needed:
// captured bodies are flat) by a copy from a pinned snapshot of the same bytes.
// Returns the number of patched nodes; snapshots are appended to `keep`.
static int patch_pageable_memcpy(cudaGraph_t g, std::vector<void*>& keep, std::vector<std::vector<unsigned char>>* bytes) {
  size_t nn = 0; CK(cudaGraphGetNodes(g, nullptr, &nn));
  std::vector<cudaGraphNode_t> nodes(nn); CK(cudaGraphGetNodes(g, nodes.data(), &nn));
  int patched = 0;
  for (auto nd : nodes) {
    cudaGraphNodeType t; CK(cudaGraphNodeGetType(nd, &t));
    if (t != cudaGraphNodeTypeMemcpy) continue;
    cudaMemcpy3DParms mp{}; CK(cudaGraphMemcpyNodeGetParams(nd, &mp));
    cudaPointerAttributes at{}; cudaPointerGetAttributes(&at, mp.srcPtr.ptr); cudaGetLastError();
    if (at.type != cudaMemoryTypeUnregistered) continue;
    size_t sz = mp.extent.width * mp.extent.height * mp.extent.depth;
    void* pinned; CK(cudaMallocHost(&pinned, sz));
    std::memcpy(pinned, mp.srcPtr.ptr, sz);
    if (bytes) bytes->emplace_back((unsigned char*)pinned, (unsigned char*)pinned + sz);
    mp.srcPtr.ptr = pinned; mp.kind = cudaMemcpyHostToDevice;
    CK(cudaGraphMemcpyNodeSetParams(nd, &mp));
    keep.push_back(pinned); ++patched;
  }
  return patched;
}

// SOLVE inside a WHILE body with the pageable memcpy nodes patched to pinned
// snapshots. Compares the solution with an eager solve bit for bit.
static void try_patched_while(Ctx& c, cudaStream_t st) {
  int dim = c.k.dim;
  CS(cudssExecute(c.h, CUDSS_PHASE_FACTORIZATION, c.cfg, c.data, c.A, c.x, c.b));
  CS(cudssExecute(c.h, CUDSS_PHASE_SOLVE, c.cfg, c.data, c.A, c.x, c.b));
  std::vector<double> eager(dim), viagraph(dim);
  CK(cudaMemcpyAsync(eager.data(), c.dsol, sizeof(double) * dim, cudaMemcpyDeviceToHost, st));
  CK(cudaStreamSynchronize(st));
  int* counter; CK(cudaMalloc(&counter, sizeof(int)));
  std::vector<void*> keep; std::vector<std::vector<unsigned char>> snap1, snap2;
  for (int round = 0; round < 2; ++round) {
    cudaGraph_t g; CK(cudaGraphCreate(&g, 0));
    cudaGraphConditionalHandle h;
    CK(cudaGraphConditionalHandleCreate(&h, g, 1, cudaGraphCondAssignDefault));
    cudaGraphNodeParams np{}; np.type = cudaGraphNodeTypeConditional;
    np.conditional.handle = h; np.conditional.type = cudaGraphCondTypeWhile; np.conditional.size = 1;
    cudaGraphNode_t node; CK(cudaGraphAddNode(&node, g, nullptr, nullptr, 0, &np));
    cudaGraph_t body = np.conditional.phGraph_out[0];
    CK(cudaStreamBeginCaptureToGraph(st, body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
    CK(cudaMemsetAsync(c.dsol, 0, sizeof(double) * dim, st));
    CS(cudssExecute(c.h, CUDSS_PHASE_SOLVE, c.cfg, c.data, c.A, c.x, c.b));
    dec_counter<<<1, 1, 0, st>>>(counter, h);
    cudaGraph_t out; CK(cudaStreamEndCapture(st, &out));
    int np_ = patch_pageable_memcpy(body, keep, round == 0 ? &snap1 : &snap2);
    cudaGraphExec_t ge; cudaError_t e = cudaGraphInstantiate(&ge, g, 0);
    if (e != cudaSuccess) { std::printf("patched while: FAIL instantiate %s\n", cudaGetErrorName(e)); cudaGetLastError(); return; }
    set_counter<<<1, 1, 0, st>>>(counter, 3);
    CK(cudaGraphLaunch(ge, st));
    CK(cudaMemcpyAsync(viagraph.data(), c.dsol, sizeof(double) * dim, cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    bool same = std::memcmp(eager.data(), viagraph.data(), sizeof(double) * dim) == 0;
    std::printf("patched while round %d: patched %d nodes, instantiate OK, 3 body runs, bitwise equal to eager: %s\n",
                round, np_, same ? "yes" : "NO");
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    if (round == 0) {   // change matrix values and refactor before the second capture
      for (double& v : c.k.val) if (v == 5.0 || v == 4.0) v = 6.0;
      CK(cudaMemcpy(c.dval, c.k.val.data(), sizeof(double) * c.k.val.size(), cudaMemcpyHostToDevice));
      CS(cudssExecute(c.h, CUDSS_PHASE_FACTORIZATION, c.cfg, c.data, c.A, c.x, c.b));
      CS(cudssExecute(c.h, CUDSS_PHASE_SOLVE, c.cfg, c.data, c.A, c.x, c.b));
      CK(cudaMemcpyAsync(eager.data(), c.dsol, sizeof(double) * dim, cudaMemcpyDeviceToHost, st));
      CK(cudaStreamSynchronize(st));
    }
  }
  bool invariant = snap1.size() == snap2.size();
  for (size_t i = 0; invariant && i < snap1.size(); ++i) invariant = snap1[i] == snap2[i];
  std::printf("pageable snapshot bytes:");
  for (auto& s : snap1) { long long iv = 0; double dv = 0; std::memcpy(&iv, s.data(), 8); std::memcpy(&dv, s.data(), 8); std::printf(" %zuB(int64=%lld, f64=%g)", s.size(), iv, dv); }
  std::printf("; identical across value updates: %s\n", invariant ? "yes" : "NO");
  for (void* p : keep) cudaFreeHost(p);
  cudaFree(counter);
}

int main() {
  int ver = 0; cudaRuntimeGetVersion(&ver);
  int major, minor, patch; cudssGetProperty(MAJOR_VERSION, &major); cudssGetProperty(MINOR_VERSION, &minor);
  cudssGetProperty(PATCH_LEVEL, &patch);
  std::printf("cudart %d, cuDSS %d.%d.%d\n", ver, major, minor, patch);
  cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamDefault));
  Ctx c; setup(c, st, true);
  std::printf("eager factor+solve residual %.3e\n", residual(c));

  struct { const char* name; int phase; } phases[] = {
    {"FACTORIZATION", CUDSS_PHASE_FACTORIZATION},
    {"REFACTORIZATION", CUDSS_PHASE_REFACTORIZATION},
    {"SOLVE", CUDSS_PHASE_SOLVE},
    {"SOLVE_FWD_PERM", CUDSS_PHASE_SOLVE_FWD_PERM},
    {"SOLVE_FWD", CUDSS_PHASE_SOLVE_FWD},
    {"SOLVE_DIAG", CUDSS_PHASE_SOLVE_DIAG},
    {"SOLVE_BWD", CUDSS_PHASE_SOLVE_BWD},
    {"SOLVE_BWD_PERM", CUDSS_PHASE_SOLVE_BWD_PERM},
    {"SOLVE_REFINEMENT", CUDSS_PHASE_SOLVE_REFINEMENT},
    {"SOLVE minus REFINEMENT", CUDSS_PHASE_SOLVE & ~CUDSS_PHASE_SOLVE_REFINEMENT},
  };
  // Node census of a plain capture of FACTORIZATION and SOLVE: memcpy nodes
  // whose source is pageable host memory are illegal in conditional bodies.
  for (int phase : {(int)CUDSS_PHASE_FACTORIZATION, (int)CUDSS_PHASE_SOLVE}) {
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    CS(cudssExecute(c.h, phase, c.cfg, c.data, c.A, c.x, c.b));
    CK(cudaStreamEndCapture(st, &g));
    size_t nn = 0; CK(cudaGraphGetNodes(g, nullptr, &nn));
    std::vector<cudaGraphNode_t> nodes(nn); CK(cudaGraphGetNodes(g, nodes.data(), &nn));
    int kernels = 0, memsets = 0, other = 0, memcpy_pageable = 0, memcpy_other = 0;
    for (auto nd : nodes) {
      cudaGraphNodeType t; CK(cudaGraphNodeGetType(nd, &t));
      if (t == cudaGraphNodeTypeKernel) ++kernels;
      else if (t == cudaGraphNodeTypeMemset) ++memsets;
      else if (t == cudaGraphNodeTypeMemcpy) {
        cudaMemcpy3DParms mp{}; CK(cudaGraphMemcpyNodeGetParams(nd, &mp));
        cudaPointerAttributes at{}; cudaPointerGetAttributes(&at, mp.srcPtr.ptr); cudaGetLastError();
        if (at.type == cudaMemoryTypeUnregistered) ++memcpy_pageable; else ++memcpy_other;
      } else ++other;
    }
    std::printf("%s plain capture nodes: kernel=%d memset=%d memcpy(pageable src)=%d memcpy(other)=%d other=%d\n",
                phase == CUDSS_PHASE_SOLVE ? "SOLVE" : "FACTORIZATION",
                kernels, memsets, memcpy_pageable, memcpy_other, other);
    cudaGraphDestroy(g);
  }
  for (auto& p : phases) {
    std::string a = try_plain(c, st, p.phase);
    std::string b = try_while(c, st, p.phase, 3);
    std::printf("%-24s plain graph: %-40s while-node body: %s\n", p.name, a.c_str(), b.c_str());
  }
  // Correctness after graph replays: factor + solve captured in one plain graph.
  {
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaMemsetAsync(c.dsol, 0, sizeof(double) * c.k.dim, st));
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    CS(cudssExecute(c.h, CUDSS_PHASE_FACTORIZATION, c.cfg, c.data, c.A, c.x, c.b));
    CS(cudssExecute(c.h, CUDSS_PHASE_SOLVE, c.cfg, c.data, c.A, c.x, c.b));
    CK(cudaStreamEndCapture(st, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
    // Change values in place (stable addresses), replay, check residual.
    for (double& v : c.k.val) if (v == 4.0) v = 5.0;
    CK(cudaMemcpy(c.dval, c.k.val.data(), sizeof(double) * c.k.val.size(), cudaMemcpyHostToDevice));
    CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));
    std::printf("graph replay after in-place value update residual %.3e\n", residual(c));
  }
  try_patched_while(c, st);
  return 0;
}
