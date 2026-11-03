// M4 kkt_assembly tests: KKT pattern against an independent dense
// construction, slot maps, masked G scatter, explicit zeros, diagonal update,
// address stability and allocation-free (graph-capturable) updates.
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <map>
#include <random>
#include <vector>

#include "common/buffer.hpp"
#include "common/handles.hpp"
#include "linalg/kkt_assembly/kkt_assembly.hpp"

using namespace cupiqp;

namespace {

uint64_t bits(double v) {
  uint64_t u;
  std::memcpy(&u, &v, sizeof(u));
  return u;
}

// Random pattern (rows x cols) with given density; `force_diag` controls the
// diagonal of a square pattern: 1 always stored, 0 never, -1 random.
CsrPattern random_pattern(std::mt19937& g, int rows, int cols, double density, bool symmetric,
                          int force_diag) {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<std::vector<char>> d(rows, std::vector<char>(cols, 0));
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j) d[i][j] = u(g) < density;
  if (symmetric) {
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < i; ++j) d[j][i] = d[i][j];
    if (force_diag >= 0)
      for (int i = 0; i < rows; ++i) d[i][i] = static_cast<char>(force_diag);
  }
  CsrPattern M;
  M.rows = rows;
  M.cols = cols;
  M.indptr.push_back(0);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j)
      if (d[i][j]) M.indices.push_back(j);
    M.indptr.push_back(static_cast<int32_t>(M.indices.size()));
  }
  return M;
}

template <class T>
std::vector<T> download(const T* dev, size_t count) {
  std::vector<T> h(count);
  if (count) {
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(h.data(), dev, count * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
  }
  return h;
}

template <class T>
void upload_sync(CudaBuffer<T>& dst, const std::vector<T>& src) {
  if (dst.empty()) dst.allocate(src.size());
  if (!src.empty())
    ASSERT_EQ(cudaMemcpy(dst.data(), src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice),
              cudaSuccess);
}

std::vector<int32_t> coo_rows(const CsrPattern& M) {
  std::vector<int32_t> r;
  for (int i = 0; i + 1 < static_cast<int>(M.indptr.size()); ++i)
    for (int k = M.indptr[i]; k < M.indptr[i + 1]; ++k) r.push_back(i);
  return r;
}

// Problem fixture: patterns, device values and masks, and a SparseDataView.
struct Fixture {
  int B = 1, n = 0, p = 0, m = 0;
  CsrPattern P, A, G;
  std::vector<double> Pv, Av, Gv, active;  // host (B, nnz) / (B, m)
  CudaBuffer<double> dPv, dAv, dGv, dactive;
  CudaBuffer<int32_t> dGrows;

  Fixture(uint32_t seed, int n_, int p_, int m_, int B_ = 1, double density = 0.3, int force_diag = -1)
      : B(B_), n(n_), p(p_), m(m_) {
    std::mt19937 g(seed);
    P = random_pattern(g, n, n, density, true, force_diag);
    if (p) A = random_pattern(g, p, n, density, false, -1);
    if (m) G = random_pattern(g, m, n, density, false, -1);
    std::normal_distribution<double> nd;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    auto fill = [&](std::vector<double>& v, int count) {
      v.resize(static_cast<size_t>(B) * count);
      for (auto& x : v) x = u(g) < 0.2 ? 0.0 : nd(g);  // explicit zeros included
    };
    fill(Pv, P.nnz());
    fill(Av, p ? A.nnz() : 0);
    fill(Gv, m ? G.nnz() : 0);
    active.resize(static_cast<size_t>(B) * m);
    for (auto& a : active) a = u(g) < 0.3 ? 0.0 : 1.0;
    upload_sync(dPv, Pv);
    upload_sync(dAv, Av);
    upload_sync(dGv, Gv);
    upload_sync(dactive, active);
    if (m) upload_sync(dGrows, coo_rows(G));
  }
  void sync_values() {
    upload_sync(dPv, Pv);
    upload_sync(dAv, Av);
    upload_sync(dGv, Gv);
    upload_sync(dactive, active);
  }
  SparseDataView<double> view() {
    SparseDataView<double> v;
    v.dims.B = B;
    v.dims.n = n;
    v.dims.p = p;
    v.dims.m = m;
    v.dims.nnz_P = P.nnz();
    v.dims.nnz_A = p ? A.nnz() : 0;
    v.dims.nnz_G = m ? G.nnz() : 0;
    v.P = {n, n, P.nnz(), nullptr, nullptr, nullptr, dPv.data()};
    v.A = {p, n, v.dims.nnz_A, nullptr, nullptr, nullptr, dAv.data()};
    v.G = {m, n, v.dims.nnz_G, nullptr, nullptr, dGrows.data(), dGv.data()};
    v.active_G_row = dactive.data();
    return v;
  }
};

// Independent dense construction of bmat([[P + I, ., .], [A, I, .], [G, ., I]])
// with all-ones blocks: template values (0 = not stored).
std::vector<std::vector<double>> dense_template(const Fixture& f) {
  const int dim = f.n + f.p + f.m;
  std::vector<std::vector<double>> K(dim, std::vector<double>(dim, 0.0));
  auto add = [&](const CsrPattern& M, int r0) {
    for (int i = 0; i + 1 < static_cast<int>(M.indptr.size()); ++i)
      for (int k = M.indptr[i]; k < M.indptr[i + 1]; ++k) K[r0 + i][M.indices[k]] += 1.0;
  };
  add(f.P, 0);
  if (f.p) add(f.A, f.n);
  if (f.m) add(f.G, f.n + f.p);
  for (int i = 0; i < dim; ++i) K[i][i] += 1.0;
  return K;
}

// Position of (r, c) in a CSR pattern by linear search (-1 if absent).
int find_slot(const CsrPattern& K, int r, int c) {
  for (int k = K.indptr[r]; k < K.indptr[r + 1]; ++k)
    if (K.indices[k] == c) return k;
  return -1;
}

// Expected KKT values after init/scatter (before any diagonal update).
std::vector<double> expected_values(const Fixture& f, const CsrPattern& K, int b) {
  std::vector<std::vector<double>> T = dense_template(f);
  std::vector<double> v(K.indices.size());
  for (int r = 0; r < K.rows; ++r)
    for (int k = K.indptr[r]; k < K.indptr[r + 1]; ++k) v[k] = T[r][K.indices[k]];
  auto put = [&](const CsrPattern& M, const std::vector<double>& vals, int r0, bool masked) {
    const int nnz = M.nnz();
    for (int i = 0; i + 1 < static_cast<int>(M.indptr.size()); ++i)
      for (int k = M.indptr[i]; k < M.indptr[i + 1]; ++k) {
        double x = vals[static_cast<size_t>(b) * nnz + k];
        if (masked) x = x * f.active[static_cast<size_t>(b) * f.m + i];
        v[find_slot(K, r0 + i, M.indices[k])] = x;
      }
  };
  put(f.P, f.Pv, 0, false);
  if (f.p) put(f.A, f.Av, f.n, false);
  if (f.m) put(f.G, f.Gv, f.n + f.p, true);
  return v;
}

void expect_bitwise(const std::vector<double>& got, const std::vector<double>& want, const char* what) {
  ASSERT_EQ(got.size(), want.size()) << what;
  size_t bad = 0;
  for (size_t i = 0; i < got.size(); ++i)
    if (bits(got[i]) != bits(want[i]) && bad++ < 5)
      ADD_FAILURE() << what << " mismatch at " << i << ": " << got[i] << " vs " << want[i];
  EXPECT_EQ(bad, 0u) << what;
}

std::vector<double> row_of(const std::vector<double>& all, int b, int nnz) {
  return std::vector<double>(all.begin() + static_cast<size_t>(b) * nnz,
                             all.begin() + static_cast<size_t>(b + 1) * nnz);
}

struct Shape {
  uint32_t seed;
  int n, p, m, B, force_diag;
};

class KktShapes : public ::testing::TestWithParam<Shape> {};

TEST_P(KktShapes, PatternMatchesDenseBmat) {
  const Shape s = GetParam();
  Fixture f(s.seed, s.n, s.p, s.m, s.B, 0.3, s.force_diag);
  KktHostMaps h = build_kkt_host_maps(f.P, f.A, f.G);
  auto T = dense_template(f);
  const int dim = s.n + s.p + s.m;
  CsrPattern want;
  std::vector<double> want_vals;
  want.indptr.push_back(0);
  for (int r = 0; r < dim; ++r) {
    for (int c = 0; c < dim; ++c)
      if (T[r][c] != 0.0) {
        want.indices.push_back(c);
        want_vals.push_back(T[r][c]);
      }
    want.indptr.push_back(static_cast<int32_t>(want.indices.size()));
  }
  EXPECT_EQ(h.kkt.rows, dim);
  EXPECT_EQ(h.kkt.cols, dim);
  EXPECT_EQ(h.kkt.indptr, want.indptr);
  EXPECT_EQ(h.kkt.indices, want.indices);
  expect_bitwise(h.template_values, want_vals, "template");

  // Diagonal slots and P-diagonal maps.
  for (int i = 0; i < dim; ++i) {
    const int slot = i < s.n ? h.diag_x[i] : i < s.n + s.p ? h.diag_y[i - s.n] : h.diag_z[i - s.n - s.p];
    EXPECT_EQ(slot, find_slot(h.kkt, i, i)) << "diag row " << i;
  }
  std::vector<int32_t> src, dst;
  auto prow = coo_rows(f.P);
  for (int k = 0; k < f.P.nnz(); ++k)
    if (prow[k] == f.P.indices[k]) {
      src.push_back(k);
      dst.push_back(f.P.indices[k]);
    }
  EXPECT_EQ(h.P_diag_src, src);
  EXPECT_EQ(h.P_diag_dst, dst);
}

TEST_P(KktShapes, InitScatterAndDiagonal) {
  const Shape s = GetParam();
  Fixture f(s.seed, s.n, s.p, s.m, s.B, 0.3, s.force_diag);
  CudaStream st = CudaStream::create();
  KktAssembly kkt;
  kkt.init(f.P, f.A, f.G, f.view(), st.get());
  const CsrPattern& K = kkt.host_pattern();
  const int nnz = kkt.nnz();
  EXPECT_EQ(kkt.dim(), s.n + s.p + s.m);
  EXPECT_EQ(download(kkt.indptr(), K.indptr.size()), K.indptr);
  EXPECT_EQ(download(kkt.indices(), K.indices.size()), K.indices);

  auto vals = download(kkt.values(), static_cast<size_t>(s.B) * nnz);
  for (int b = 0; b < s.B; ++b) expect_bitwise(row_of(vals, b, nnz), expected_values(f, K, b), "init");

  // P_diag: stored P diagonal, 0 where not stored.
  auto pd = download(kkt.P_diag(), static_cast<size_t>(s.B) * s.n);
  auto prow = coo_rows(f.P);
  for (int b = 0; b < s.B; ++b) {
    std::vector<double> want(s.n, 0.0);
    for (int k = 0; k < f.P.nnz(); ++k)
      if (prow[k] == f.P.indices[k]) want[prow[k]] = f.Pv[static_cast<size_t>(b) * f.P.nnz() + k];
    expect_bitwise(row_of(pd, b, s.n), want, "P_diag");
  }

  // Diagonal update with z_reg including zeros.
  std::mt19937 g(s.seed + 7);
  std::normal_distribution<double> nd;
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<double> xr(static_cast<size_t>(s.B) * s.n), dl(s.B), zr(static_cast<size_t>(s.B) * s.m);
  for (auto& x : xr) x = std::abs(nd(g)) * 1e-3;
  for (auto& x : dl) x = std::abs(nd(g)) * 1e-2;
  for (auto& x : zr) x = u(g) < 0.3 ? 0.0 : std::abs(nd(g));
  CudaBuffer<double> dxr, ddl, dzr;
  upload_sync(dxr, xr);
  upload_sync(ddl, dl);
  upload_sync(dzr, zr);
  kkt.update_diagonal(dxr.data(), ddl.data(), dzr.data(), st.get());
  vals = download(kkt.values(), static_cast<size_t>(s.B) * nnz);
  const KktHostMaps& h = kkt.host_maps();
  for (int b = 0; b < s.B; ++b) {
    std::vector<double> want = expected_values(f, K, b);
    for (int i = 0; i < s.n; ++i) want[h.diag_x[i]] = pd[static_cast<size_t>(b) * s.n + i] + xr[static_cast<size_t>(b) * s.n + i];
    for (int i = 0; i < s.p; ++i) want[h.diag_y[i]] = -dl[b];
    for (int i = 0; i < s.m; ++i) {
      const double z = zr[static_cast<size_t>(b) * s.m + i];
      want[h.diag_z[i]] = z > 0.0 ? -z : -1.0;
    }
    expect_bitwise(row_of(vals, b, nnz), want, "diag");
  }
}

INSTANTIATE_TEST_SUITE_P(Shapes, KktShapes,
                         ::testing::Values(Shape{1, 9, 3, 7, 1, -1}, Shape{2, 12, 0, 5, 1, -1},
                                           Shape{3, 8, 4, 0, 1, -1}, Shape{4, 10, 0, 0, 1, 0},
                                           Shape{5, 6, 2, 9, 1, 1}, Shape{6, 11, 3, 6, 2, -1},
                                           Shape{7, 1, 1, 1, 1, -1}));

TEST(KktAssembly, ExplicitZerosAndBoundSwitches) {
  Fixture f(11, 10, 3, 8);
  // Start with all values zero and every G row inactive.
  std::fill(f.Pv.begin(), f.Pv.end(), 0.0);
  std::fill(f.Av.begin(), f.Av.end(), 0.0);
  std::fill(f.Gv.begin(), f.Gv.end(), 0.0);
  std::fill(f.active.begin(), f.active.end(), 0.0);
  f.sync_values();
  CudaStream st = CudaStream::create();
  KktAssembly kkt;
  kkt.init(f.P, f.A, f.G, f.view(), st.get());
  const CsrPattern& K = kkt.host_pattern();
  ASSERT_EQ(K.nnz(), build_kkt_host_maps(f.P, f.A, f.G).kkt.nnz());  // zeros keep their slots
  expect_bitwise(download(kkt.values(), kkt.nnz()), expected_values(f, K, 0), "zeros");

  // The explicit zeros become nonzero and G rows switch active / inactive.
  std::mt19937 g(5);
  std::normal_distribution<double> nd;
  for (auto& x : f.Pv) x = nd(g);
  for (auto& x : f.Av) x = nd(g);
  for (auto& x : f.Gv) x = nd(g);
  for (int i = 0; i < f.m; ++i) f.active[i] = (i % 3 == 0) ? 0.0 : 1.0;
  f.sync_values();
  double* addr = kkt.values();
  kkt.scatter(f.view(), true, true, true, st.get());
  EXPECT_EQ(kkt.values(), addr);
  auto vals = download(kkt.values(), kkt.nnz());
  expect_bitwise(vals, expected_values(f, K, 0), "nonzero");
  // Inactive G rows: every G entry of the row is zero (-0.0 for negative G,
  // as G * 0.0 in the reference).
  const KktHostMaps& h = kkt.host_maps();
  for (int k = 0; k < f.G.nnz(); ++k)
    if (f.active[h.G_rows[k]] == 0.0) EXPECT_EQ(vals[h.G_slots[k]], 0.0);

  // Partial flags: only A changes.
  std::vector<double> before = vals;
  for (auto& x : f.Pv) x = nd(g);
  for (auto& x : f.Av) x = nd(g);
  f.sync_values();
  kkt.scatter(f.view(), false, true, false, st.get());
  vals = download(kkt.values(), kkt.nnz());
  for (int k = 0; k < kkt.nnz(); ++k) {
    bool is_A = false;
    for (int s : h.A_slots) is_A |= (s == k);
    if (!is_A) EXPECT_EQ(bits(vals[k]), bits(before[k])) << k;
  }
  for (int k = 0; k < f.A.nnz(); ++k) EXPECT_EQ(bits(vals[h.A_slots[k]]), bits(f.Av[k]));
}

// Scatter + diagonal update are captured in global mode (which rejects
// cudaMalloc / synchronizing calls) and replayed: results match eager runs,
// and buffer addresses never change.
TEST(KktAssembly, GraphCaptureNoAllocationStableAddresses) {
  Fixture f(21, 14, 4, 12);
  CudaStream st = CudaStream::create(cudaStreamNonBlocking);
  KktAssembly kkt;
  kkt.init(f.P, f.A, f.G, f.view(), st.get());
  const double* v0 = kkt.values();
  const double* pd0 = kkt.P_diag();
  const int32_t* ip0 = kkt.indptr();
  const int32_t* ix0 = kkt.indices();

  std::vector<double> xr(f.n, 1e-4), dl(1, 3e-3), zr(f.m);
  for (int i = 0; i < f.m; ++i) zr[i] = (i % 4 == 0) ? 0.0 : 0.5 + i;
  CudaBuffer<double> dxr, ddl, dzr;
  upload_sync(dxr, xr);
  upload_sync(ddl, dl);
  upload_sync(dzr, zr);

  size_t free0 = 0, total = 0;
  ASSERT_EQ(cudaMemGetInfo(&free0, &total), cudaSuccess);
  cudaGraph_t graph = nullptr;
  ASSERT_EQ(cudaStreamBeginCapture(st.get(), cudaStreamCaptureModeGlobal), cudaSuccess);
  kkt.scatter(f.view(), true, true, true, st.get());
  kkt.update_diagonal(dxr.data(), ddl.data(), dzr.data(), st.get());
  ASSERT_EQ(cudaStreamEndCapture(st.get(), &graph), cudaSuccess);
  cudaGraphExec_t exec = nullptr;
  ASSERT_EQ(cudaGraphInstantiate(&exec, graph, 0), cudaSuccess);

  // New values, replay twice.
  std::mt19937 g(9);
  std::normal_distribution<double> nd;
  for (auto& x : f.Pv) x = nd(g);
  for (auto& x : f.Gv) x = nd(g);
  for (int i = 0; i < f.m; ++i) f.active[i] = (i % 2) ? 1.0 : 0.0;
  f.sync_values();
  ASSERT_EQ(cudaGraphLaunch(exec, st.get()), cudaSuccess);
  ASSERT_EQ(cudaGraphLaunch(exec, st.get()), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(st.get()), cudaSuccess);
  auto replay = download(kkt.values(), kkt.nnz());

  kkt.scatter(f.view(), true, true, true, st.get());
  kkt.update_diagonal(dxr.data(), ddl.data(), dzr.data(), st.get());
  ASSERT_EQ(cudaStreamSynchronize(st.get()), cudaSuccess);
  expect_bitwise(replay, download(kkt.values(), kkt.nnz()), "graph vs eager");

  EXPECT_EQ(kkt.values(), v0);
  EXPECT_EQ(kkt.P_diag(), pd0);
  EXPECT_EQ(kkt.indptr(), ip0);
  EXPECT_EQ(kkt.indices(), ix0);
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
}

TEST(KktAssembly, RejectsInconsistentInputs) {
  Fixture f(31, 6, 2, 3);
  CsrPattern Abad = f.A;
  Abad.cols = 5;
  EXPECT_THROW(build_kkt_host_maps(f.P, Abad, f.G), Error);
  KktAssembly kkt;
  auto v = f.view();
  v.dims.m = 2;
  EXPECT_THROW(kkt.init(f.P, f.A, f.G, v, nullptr), Error);
}

// Timing of the version-1 kernels on the MPC shape (n = 169, p = 64,
// m = 2521, nnz(G) = 249801); for the optimization assessment. Disabled by
// default: run with --gtest_also_run_disabled_tests.
TEST(KktAssembly, DISABLED_TimingMpcShape) {
  const int n = 169, p = 64, m = 2521;
  Fixture f(41, n, p, m, 1, 249801.0 / (static_cast<double>(m) * n), -1);
  printf("nnz P %d A %d G %d\n", f.P.nnz(), f.A.nnz(), f.G.nnz());
  CudaStream st = CudaStream::create(cudaStreamNonBlocking);
  KktAssembly kkt;
  kkt.init(f.P, f.A, f.G, f.view(), st.get());
  CudaBuffer<double> xr(n), dl(1), zr(m);
  xr.zero_async(st.get());
  dl.zero_async(st.get());
  zr.zero_async(st.get());
  CudaEvent e0 = CudaEvent::create(true), e1 = CudaEvent::create(true);
  auto time = [&](const char* name, auto fn) {
    for (int i = 0; i < 10; ++i) fn();
    const int reps = 200;
    cudaEventRecord(e0.get(), st.get());
    for (int i = 0; i < reps; ++i) fn();
    cudaEventRecord(e1.get(), st.get());
    cudaEventSynchronize(e1.get());
    float ms = 0;
    cudaEventElapsedTime(&ms, e0.get(), e1.get());
    // Same calls captured into one graph (no per-launch host overhead).
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaStreamBeginCapture(st.get(), cudaStreamCaptureModeGlobal);
    for (int i = 0; i < reps; ++i) fn();
    cudaStreamEndCapture(st.get(), &graph);
    cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphLaunch(exec, st.get());
    cudaEventRecord(e0.get(), st.get());
    cudaGraphLaunch(exec, st.get());
    cudaEventRecord(e1.get(), st.get());
    cudaEventSynchronize(e1.get());
    float ms_graph = 0;
    cudaEventElapsedTime(&ms_graph, e0.get(), e1.get());
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    printf("%-28s eager %8.2f us/call   in graph %8.2f us/call\n", name, 1000.0 * ms / reps,
           1000.0 * ms_graph / reps);
  };
  auto v = f.view();
  time("scatter P+A+G (+P_diag)", [&] { kkt.scatter(v, true, true, true, st.get()); });
  time("scatter G only", [&] { kkt.scatter(v, false, false, true, st.get()); });
  time("scatter P only (+P_diag)", [&] { kkt.scatter(v, true, false, false, st.get()); });
  time("update_diagonal", [&] { kkt.update_diagonal(xr.data(), dl.data(), zr.data(), st.get()); });
}

}  // namespace
