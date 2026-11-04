// Benchmark (DESIGN 10.6 optimization assessment): the SpMV kernel against
// alternative custom CSR / CSC kernels on the MPC shapes (n = 169, p = 64, m = 2521, nnz(P) = 8977,
// nnz(A) = 920, nnz(G) = 249801, G about 59% stored and 82% explicit zeros).
// Synthetic random patterns with those sizes. Times: CUDA events around a
// batch of calls, warmed, median of repeats; "graph" replays one captured
// graph holding the batch (GPU time without host enqueue cost), "stream"
// enqueues the batch eagerly (includes host launch overhead). Results before
// 2026-11-04 also timed cuSPARSE (removed as a dependency).
//
// Usage: bench_spmv [reps=15] [batch=200]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "common/graph.hpp"
#include "common/handles.hpp"
#include "common/reduce.cuh"
#include "linalg/spmv/spmv.hpp"
#include "spmv_test_util.cuh"

using namespace cupiqp;
using namespace cupiqp::spmv_test;

namespace {

// LANES threads per row, shuffle reduction (LANES = 1: thread per row).
// With perm != nullptr the values are gathered (CSC view of a CSR matrix).
template <int LANES>
__global__ void csr_lanes(int rows, const int* __restrict__ indptr, const int* __restrict__ idx,
                          const double* __restrict__ val, const int* __restrict__ perm,
                          const double* __restrict__ x, double alpha, double* __restrict__ y) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = gid / LANES, lane = gid % LANES;
  double s = 0.0;
  if (row < rows) {
    const int end = indptr[row + 1];
    if (perm)
      for (int k = indptr[row] + lane; k < end; k += LANES) s += val[perm[k]] * x[idx[k]];
    else
      for (int k = indptr[row] + lane; k < end; k += LANES) s += val[k] * x[idx[k]];
  }
#pragma unroll
  for (int o = LANES / 2; o > 0; o /= 2) s += __shfl_down_sync(0xffffffffu, s, o, LANES);
  if (row < rows && lane == 0) y[row] = alpha * s;
}

// One block per row (column of the CSC view), cub block reduction.
template <int BLOCK>
__global__ void csr_block(int rows, const int* __restrict__ indptr, const int* __restrict__ idx,
                          const double* __restrict__ val, const int* __restrict__ perm,
                          const double* __restrict__ x, double alpha, double* __restrict__ y) {
  const int row = blockIdx.x;
  double s = 0.0;
  const int end = indptr[row + 1];
  if (perm)
    for (int k = indptr[row] + threadIdx.x; k < end; k += BLOCK) s += val[perm[k]] * x[idx[k]];
  else
    for (int k = indptr[row] + threadIdx.x; k < end; k += BLOCK) s += val[k] * x[idx[k]];
  s = block_sum<BLOCK>(s);
  if (threadIdx.x == 0) y[row] = alpha * s;
}

// Transposed product by atomic scatter, thread per stored entry (y zeroed first).
__global__ void coo_scatter(int nnz, const int* __restrict__ row_idx, const int* __restrict__ idx,
                            const double* __restrict__ val, const double* __restrict__ x, double alpha,
                            double* __restrict__ y) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < nnz) atomicAdd(&y[idx[k]], alpha * val[k] * x[row_idx[k]]);
}

// Transposed product with a shared-memory accumulator per block (cols <= SMEM),
// warp per row, then one atomic per column and block (y zeroed first).
template <int SMEM>
__global__ void smem_scatter(int rows, int cols, const int* __restrict__ indptr, const int* __restrict__ idx,
                             const double* __restrict__ val, const double* __restrict__ x, double alpha,
                             double* __restrict__ y) {
  __shared__ double acc[SMEM];
  for (int j = threadIdx.x; j < cols; j += blockDim.x) acc[j] = 0.0;
  __syncthreads();
  const int warps = blockDim.x / 32, lane = threadIdx.x % 32;
  for (int r = blockIdx.x * warps + threadIdx.x / 32; r < rows; r += gridDim.x * warps) {
    const double xr = x[r];
    for (int k = indptr[r] + lane; k < indptr[r + 1]; k += 32) atomicAdd(&acc[idx[k]], val[k] * xr);
  }
  __syncthreads();
  for (int j = threadIdx.x; j < cols; j += blockDim.x) atomicAdd(&y[j], alpha * acc[j]);
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

struct Timer {
  cudaStream_t s;
  int reps, batch;
  // Returns {stream median us/call, graph median us/call}.
  std::pair<double, double> run(const std::function<void(cudaStream_t)>& call) {
    CudaEvent a = CudaEvent::create(true), b = CudaEvent::create(true);
    auto many = [&](cudaStream_t st) {
      for (int i = 0; i < batch; ++i) call(st);
    };
    for (int w = 0; w < 3; ++w) many(s);
    CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<double> ts, tg;
    for (int r = 0; r < reps; ++r) {
      a.record(s);
      many(s);
      b.record(s);
      CUPIQP_CUDA_CHECK(cudaEventSynchronize(b.get()));
      float ms = 0;
      CUPIQP_CUDA_CHECK(cudaEventElapsedTime(&ms, a.get(), b.get()));
      ts.push_back(1e3 * ms / batch);
    }
    Counters c;
    GraphExec exec;
    exec.instantiate(capture(s, many, &c));
    exec.launch(s, &c);
    for (int r = 0; r < reps; ++r) {
      a.record(s);
      exec.launch(s, &c);
      b.record(s);
      CUPIQP_CUDA_CHECK(cudaEventSynchronize(b.get()));
      float ms = 0;
      CUPIQP_CUDA_CHECK(cudaEventElapsedTime(&ms, a.get(), b.get()));
      tg.push_back(1e3 * ms / batch);
    }
    return {median(ts), median(tg)};
  }
};

// CSR -> CSC pattern with a value permutation (perm[k_csc] = k_csr).
struct Csc {
  std::vector<int32_t> indptr, indices, perm;
};
Csc transpose_pattern(const HostCsr& M) {
  Csc T;
  T.indptr.assign(M.cols + 1, 0);
  for (int32_t c : M.indices) T.indptr[c + 1]++;
  std::partial_sum(T.indptr.begin(), T.indptr.end(), T.indptr.begin());
  T.indices.resize(M.nnz());
  T.perm.resize(M.nnz());
  std::vector<int32_t> next(T.indptr.begin(), T.indptr.end() - 1);
  for (int32_t r = 0; r < M.rows; ++r)
    for (int32_t k = M.indptr[r]; k < M.indptr[r + 1]; ++k) {
      const int32_t pos = next[M.indices[k]]++;
      T.indices[pos] = r;
      T.perm[pos] = k;
    }
  return T;
}

template <class V>
CudaBuffer<V> dev(const std::vector<V>& h) {
  CudaBuffer<V> d(h.size());
  d.copy_from_host(h.data(), h.size(), nullptr);
  CUPIQP_CUDA_CHECK(cudaDeviceSynchronize());
  return d;
}

// Random CSR with an exact nnz: Bernoulli rows trimmed / padded to the target.
HostCsr random_exact(int32_t rows, int32_t cols, int32_t nnz, double zero_frac, uint64_t seed) {
  std::mt19937_64 g(seed);
  std::vector<int64_t> all(static_cast<size_t>(rows) * cols);
  std::iota(all.begin(), all.end(), 0);
  std::shuffle(all.begin(), all.end(), g);
  all.resize(nnz);
  std::sort(all.begin(), all.end());
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::normal_distribution<double> nrm(0.0, 1.0);
  HostCsr M;
  M.rows = rows;
  M.cols = cols;
  M.indptr.assign(rows + 1, 0);
  for (int64_t e : all) {
    M.indptr[e / cols + 1]++;
    M.indices.push_back(static_cast<int32_t>(e % cols));
    M.values.push_back(u(g) < zero_frac ? 0.0 : nrm(g));
  }
  std::partial_sum(M.indptr.begin(), M.indptr.end(), M.indptr.begin());
  return M;
}

double max_rel(const std::vector<double>& got, const std::vector<double>& want, const std::vector<double>& mag) {
  double r = 0;
  for (size_t i = 0; i < got.size(); ++i) r = std::max(r, std::fabs(got[i] - want[i]) / std::max(mag[i], 1e-300));
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 15;
  const int batch = argc > 2 ? std::atoi(argv[2]) : 200;
  CudaStream stream = CudaStream::create(cudaStreamNonBlocking);
  cudaStream_t s = stream.get();
  Timer timer{s, reps, batch};

  struct Mat {
    const char* name;
    HostCsr M;
  };
  std::vector<Mat> mats;
  mats.push_back({"P", random_exact(169, 169, 8977, 0.0, 1)});
  mats.push_back({"A", random_exact(64, 169, 920, 0.0, 2)});
  mats.push_back({"G", random_exact(2521, 169, 249801, 0.82, 3)});

  std::printf("# reps=%d batch=%d; us per call, median; rel = max |y - y_cpu| / (|op(M)| |x|)\n", reps, batch);
  std::printf("%-4s %-3s %-32s %10s %10s %10s %s\n", "mat", "op", "kernel", "stream_us", "graph_us", "rel", "det");
  for (auto& mt : mats) {
    HostCsr& M = mt.M;
    DeviceCsr dM(M);
    std::vector<int32_t> row_idx(M.nnz());
    for (int32_t r = 0; r < M.rows; ++r)
      for (int32_t k = M.indptr[r]; k < M.indptr[r + 1]; ++k) row_idx[k] = r;
    auto d_row = dev(row_idx);
    Csc T = transpose_pattern(M);
    auto t_ptr = dev(T.indptr), t_idx = dev(T.indices), t_perm = dev(T.perm);

    for (bool tr : {false, true}) {
      const int32_t nx = tr ? M.rows : M.cols, ny = tr ? M.cols : M.rows;
      const auto x = random_vec(nx, 5);
      CudaBuffer<double> dx = to_device(x), dy(ny);
      std::vector<double> want(ny, 0.0), mag;
      cpu_spmv(M, tr, 1.0, x, 0.0, want, &mag);
      const double* xp = dx.data();
      double* yp = dy.data();
      const double* vp = dM.values.data();

      auto report = [&](const std::string& name, const std::function<void(cudaStream_t)>& call) {
        // Determinism: 10 repeats bitwise equal?
        call(s);
        const auto first = to_host(dy, s);
        bool det = true;
        for (int i = 0; i < 10; ++i) {
          call(s);
          const auto again = to_host(dy, s);
          det = det && std::equal(first.begin(), first.end(), again.begin(),
                                  [](double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; });
        }
        const double rel = max_rel(first, want, mag);
        auto [ts, tg] = timer.run(call);
        std::printf("%-4s %-3s %-32s %10.2f %10.2f %10.2e %s\n", mt.name, tr ? "T" : "N", name.c_str(), ts, tg, rel,
                    det ? "yes" : "no");
      };

      SpMV op;
      op.init(dM.view(), tr);
      report("SpMV (" + std::to_string(op.lanes()) + " lanes)",
             [&](cudaStream_t st) { op(st, 1.0, xp, 0.0, yp); });

      if (!tr) {
        const int* ip = dM.indptr.data();
        const int* ix = dM.indices.data();
        auto lanes = [&](auto L, const char* nm) {
          constexpr int LN = decltype(L)::value;
          report(nm, [&](cudaStream_t st) {
            const unsigned grid = grid_for(static_cast<int64_t>(M.rows) * LN, 128);
            csr_lanes<LN><<<grid, 128, 0, st>>>(M.rows, ip, ix, vp, nullptr, xp, 1.0, yp);
          });
        };
        lanes(std::integral_constant<int, 1>{}, "thread/row");
        lanes(std::integral_constant<int, 4>{}, "4 lanes/row");
        lanes(std::integral_constant<int, 8>{}, "8 lanes/row");
        lanes(std::integral_constant<int, 16>{}, "16 lanes/row");
        lanes(std::integral_constant<int, 32>{}, "warp/row");
        report("block128/row (cub)", [&](cudaStream_t st) {
          csr_block<128><<<M.rows, 128, 0, st>>>(M.rows, ip, ix, vp, nullptr, xp, 1.0, yp);
        });
      } else {
        const int* tp = t_ptr.data();
        const int* ti = t_idx.data();
        const int* pp = t_perm.data();
        report("atomic scatter thread/nnz", [&](cudaStream_t st) {
          CUPIQP_CUDA_CHECK(cudaMemsetAsync(yp, 0, ny * sizeof(double), st));
          coo_scatter<<<grid_for(M.nnz(), 256), 256, 0, st>>>(M.nnz(), d_row.data(), dM.indices.data(), vp, xp, 1.0,
                                                              yp);
        });
        if (M.cols <= 256)
          report("smem scatter (warp/row) 64blk", [&](cudaStream_t st) {
            CUPIQP_CUDA_CHECK(cudaMemsetAsync(yp, 0, ny * sizeof(double), st));
            smem_scatter<256><<<64, 256, 0, st>>>(M.rows, M.cols, dM.indptr.data(), dM.indices.data(), vp, xp, 1.0,
                                                  yp);
          });
        auto lanes = [&](auto L, const char* nm) {
          constexpr int LN = decltype(L)::value;
          report(nm, [&](cudaStream_t st) {
            const unsigned grid = grid_for(static_cast<int64_t>(M.cols) * LN, 128);
            csr_lanes<LN><<<grid, 128, 0, st>>>(M.cols, tp, ti, vp, pp, xp, 1.0, yp);
          });
        };
        lanes(std::integral_constant<int, 8>{}, "CSC gather 8 lanes/col");
        lanes(std::integral_constant<int, 32>{}, "CSC gather warp/col");
        report("CSC gather block128/col (cub)", [&](cudaStream_t st) {
          csr_block<128><<<M.cols, 128, 0, st>>>(M.cols, tp, ti, vp, pp, xp, 1.0, yp);
        });
        report("CSC gather block256/col (cub)", [&](cudaStream_t st) {
          csr_block<256><<<M.cols, 256, 0, st>>>(M.cols, tp, ti, vp, pp, xp, 1.0, yp);
        });
      }
    }
  }
  return 0;
}
