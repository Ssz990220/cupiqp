// Module data tests: CSR validation, block consistency, value storage and the
// finite-bound masks against a CPU implementation of the reference kernel.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "common/buffer.hpp"
#include "problem/data/sparse_data.hpp"

using namespace cupiqp;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
const double kNaN = std::numeric_limits<double>::quiet_NaN();

CsrPattern pat(int32_t rows, int32_t cols, std::vector<int32_t> indptr, std::vector<int32_t> indices) {
  CsrPattern M;
  M.rows = rows;
  M.cols = cols;
  M.indptr = std::move(indptr);
  M.indices = std::move(indices);
  return M;
}

template <class T>
struct Dev {
  CudaBuffer<T> buf;
  explicit Dev(const std::vector<T>& h) : buf(h.size()) {
    buf.copy_from_host(h.data(), h.size(), nullptr);
    CUPIQP_CUDA_CHECK(cudaStreamSynchronize(nullptr));
  }
  const T* ptr() const { return buf.data(); }
};

template <class T>
std::vector<T> download(const T* d, size_t n) {
  std::vector<T> h(n);
  if (n) CUPIQP_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}

// Bitwise equality of doubles (NaN-safe).
void expect_bits(const std::vector<double>& a, const std::vector<double>& b, const char* what) {
  ASSERT_EQ(a.size(), b.size()) << what;
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(0, std::memcmp(&a[i], &b[i], sizeof(double))) << what << "[" << i << "]: " << a[i] << " vs " << b[i];
  }
}

void expect_invalid(const std::function<void()>& f, const std::string& needle) {
  try {
    f();
    ADD_FAILURE() << "expected InvalidArgument containing '" << needle << "'";
  } catch (const Error& e) {
    EXPECT_EQ(e.code(), ErrorCode::InvalidArgument);
    EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
  }
}

// CPU implementation of the reference finite_bound_masks_kernel (B = 1).
struct CpuMasks {
  std::vector<double> all, gRow, xBound;
  double count = 0;
};
CpuMasks cpu_masks(int m, int n, const std::vector<double>* hl, const std::vector<double>* hu,
                   const std::vector<double>* xl, const std::vector<double>* xu) {
  CpuMasks r;
  std::vector<double> fhl, fhu, fxl, fxu;
  r.gRow.assign(m, 0);
  r.xBound.assign(n, 0);
  for (int i = 0; i < m; ++i) {
    double fl = 0, fu = 0;
    if (hl) { fl = (*hl)[i] > -PIQP_INF ? 1 : 0; fhl.push_back(fl); r.count += fl; }
    if (hu) { fu = (*hu)[i] < PIQP_INF ? 1 : 0; fhu.push_back(fu); r.count += fu; }
    r.gRow[i] = std::max(fl, fu);
  }
  for (int j = 0; j < n; ++j) {
    double fl = 0, fu = 0;
    if (xl) { fl = (*xl)[j] > -PIQP_INF ? 1 : 0; fxl.push_back(fl); r.count += fl; }
    if (xu) { fu = (*xu)[j] < PIQP_INF ? 1 : 0; fxu.push_back(fu); r.count += fu; }
    r.xBound[j] = std::max(fl, fu);
  }
  for (auto* v : {&fhl, &fhu, &fxl, &fxu}) r.all.insert(r.all.end(), v->begin(), v->end());
  return r;
}

void expect_masks(SparseData& d, const CpuMasks& want) {
  CUPIQP_CUDA_CHECK(cudaDeviceSynchronize());
  auto v = d.view();
  expect_bits(download(v.finite_mask_all, v.dims.num_ineq()), want.all, "finite_mask_all");
  expect_bits(download(v.active_G_row, v.dims.m), want.gRow, "active_G_row");
  expect_bits(download(v.active_x_bound, v.dims.n), want.xBound, "active_x_bound");
  EXPECT_EQ(download(v.num_finite_bounds, 1)[0], want.count);
}

// A small problem with every block: n = 3, p = 1, m = 2.
struct Problem {
  CsrPattern P = pat(3, 3, {0, 2, 3, 5}, {0, 2, 1, 0, 2});
  CsrPattern A = pat(1, 3, {0, 2}, {0, 2});
  CsrPattern G = pat(2, 3, {0, 1, 3}, {1, 0, 2});
  std::vector<double> Pv{4, 0.0, 3, 0.0, 5}, c{1, 2, 3}, Av{1, 0.0}, b{2}, Gv{1, 2, 0.0};
  std::vector<double> hl{-1, -kInf}, hu{kInf, 4}, xl{-1, -2, -kInf}, xu{1, kInf, 1e20};
};

}  // namespace

// ------------------------------------------------------------------ validation
TEST(ValidateCsr, AcceptsValidPatterns) {
  EXPECT_NO_THROW(validate_csr(pat(3, 3, {0, 2, 3, 5}, {0, 2, 1, 0, 2}), -1, "P"));
  EXPECT_NO_THROW(validate_csr(pat(3, 3, {0, 0, 0, 0}, {}), -1, "P"));          // empty rows
  EXPECT_NO_THROW(validate_csr(pat(0, 4, {0}, {}), 4, "A"));                    // no rows
  EXPECT_NO_THROW(validate_csr(pat(2, 5, {0, 1, 1}, {4}), 5, "G"));             // empty row in between
}

TEST(ValidateCsr, RejectsEveryReferenceCase) {
  expect_invalid([] { validate_csr(pat(2, 2, {1, 2, 3}, {0, 1}), -1, "P"); }, "P indptr must start at 0");
  expect_invalid([] { validate_csr(pat(2, 2, {0, 2, 1}, {0, 1}), -1, "P"); }, "never decrease");
  expect_invalid([] { validate_csr(pat(2, 2, {0, 1, 3}, {0, 1}), -1, "P"); }, "end at nnz = 2");
  expect_invalid([] { validate_csr(pat(2, 2, {0, 1, 2}, {0, 2}), -1, "P"); }, "column indices must lie in [0, 2)");
  expect_invalid([] { validate_csr(pat(2, 2, {0, 1, 2}, {-1, 0}), -1, "P"); }, "column indices must lie");
  expect_invalid([] { validate_csr(pat(2, 2, {0, 2, 3}, {1, 0, 1}), -1, "P"); }, "strictly increasing");
  expect_invalid([] { validate_csr(pat(2, 2, {0, 2, 2}, {1, 1}), -1, "P"); }, "strictly increasing");  // duplicate
  expect_invalid([] { validate_csr(pat(2, 3, {0, 0, 0}, {}), -1, "P"); }, "P must have 2 columns");     // not square
  expect_invalid([] { validate_csr(pat(1, 3, {0, 0}, {}), 4, "A"); }, "A must have 4 columns");
  expect_invalid([] { validate_csr(pat(2, 2, {0, 0}, {}), -1, "P"); }, "indptr must have length");
  expect_invalid([] { validate_csr(pat(0, 0, {}, {}), -1, "P"); }, "at least one entry");
}

TEST(ValidateCsr, SortedAcrossRowBoundaryIsFine) {
  // Index decreases from one row to the next: allowed.
  EXPECT_NO_THROW(validate_csr(pat(2, 3, {0, 2, 3}, {1, 2, 0}), 3, "G"));
}

// ------------------------------------------------------------------ init rules
TEST(SparseDataInit, BlockConsistencyRules) {
  Problem q;
  Dev<double> Pv(q.Pv), c(q.c), Av(q.Av), b(q.b), Gv(q.Gv), hl(q.hl);
  auto base = [&] {
    ValuesView v;
    v.P = Pv.ptr();
    v.c = c.ptr();
    return v;
  };
  {
    SparseData d;
    ValuesView v = base();
    v.A = Av.ptr();
    expect_invalid([&] { d.init(q.P, &q.A, nullptr, v, nullptr); }, "A and b must either both");
    EXPECT_FALSE(d.initialized());
  }
  {
    SparseData d;
    ValuesView v = base();
    v.b = b.ptr();
    expect_invalid([&] { d.init(q.P, nullptr, nullptr, v, nullptr); }, "A and b must either both");
  }
  {
    SparseData d;
    ValuesView v = base();
    v.G = Gv.ptr();
    expect_invalid([&] { d.init(q.P, nullptr, &q.G, v, nullptr); }, "Either h_l or h_u");
  }
  {
    SparseData d;
    ValuesView v = base();
    v.h_l = hl.ptr();
    expect_invalid([&] { d.init(q.P, nullptr, nullptr, v, nullptr); }, "must be None when G is None");
  }
  {
    SparseData d;
    ValuesView v = base();
    v.P = nullptr;
    expect_invalid([&] { d.init(q.P, nullptr, nullptr, v, nullptr); }, "P values");
  }
  {
    SparseData d;
    ValuesView v = base();
    v.c = nullptr;
    expect_invalid([&] { d.init(q.P, nullptr, nullptr, v, nullptr); }, "requires c");
  }
  {
    SparseData d;  // bad pattern is rejected before anything else
    CsrPattern bad = pat(1, 3, {0, 2}, {2, 0});
    ValuesView v = base();
    v.A = Av.ptr();
    v.b = b.ptr();
    expect_invalid([&] { d.init(q.P, &bad, nullptr, v, nullptr); }, "A column indices must be strictly");
    EXPECT_FALSE(d.initialized());
  }
  {
    SparseData d;
    ValuesView v = base();
    d.init(q.P, nullptr, nullptr, v, nullptr);
    expect_invalid([&] { d.init(q.P, nullptr, nullptr, v, nullptr); }, "only be called once");
  }
}

TEST(SparseDataInit, AbsentBlocksGiveValidEmptyViews) {
  Problem q;
  Dev<double> Pv(q.Pv), c(q.c);
  ValuesView v;
  v.P = Pv.ptr();
  v.c = c.ptr();
  SparseData d;
  d.init(q.P, nullptr, nullptr, v, nullptr);
  const Dims& dm = d.dims();
  EXPECT_EQ(dm.n, 3);
  EXPECT_EQ(dm.p, 0);
  EXPECT_EQ(dm.m, 0);
  EXPECT_EQ(dm.num_ineq(), 0);
  auto view = d.view();
  EXPECT_EQ(view.A.rows, 0);
  EXPECT_EQ(view.A.cols, 3);
  EXPECT_EQ(view.A.nnz, 0);
  ASSERT_NE(view.A.indptr, nullptr);
  ASSERT_NE(view.G.indptr, nullptr);
  EXPECT_EQ(download(view.A.indptr, 1)[0], 0);
  EXPECT_EQ(download(view.G.indptr, 1)[0], 0);
  EXPECT_EQ(view.b, nullptr);
  EXPECT_EQ(view.h_l, nullptr);
  EXPECT_EQ(view.x_u, nullptr);
  EXPECT_EQ(view.finite_mask_hl(), nullptr);
  expect_masks(d, cpu_masks(0, 3, nullptr, nullptr, nullptr, nullptr));
  EXPECT_EQ(d.A_pattern().rows, 0);
  EXPECT_EQ(d.G_pattern().cols, 3);
}

TEST(SparseDataInit, StoresPatternsRowsAndValuesWithExplicitZeros) {
  Problem q;
  Dev<double> Pv(q.Pv), c(q.c), Av(q.Av), b(q.b), Gv(q.Gv), hl(q.hl), hu(q.hu), xl(q.xl), xu(q.xu);
  ValuesView v{Pv.ptr(), c.ptr(), Av.ptr(), b.ptr(), Gv.ptr(), hl.ptr(), hu.ptr(), xl.ptr(), xu.ptr()};
  SparseData d;
  d.init(q.P, &q.A, &q.G, v, nullptr);
  CUPIQP_CUDA_CHECK(cudaDeviceSynchronize());
  const Dims& dm = d.dims();
  EXPECT_EQ(dm.n, 3);
  EXPECT_EQ(dm.p, 1);
  EXPECT_EQ(dm.m, 2);
  EXPECT_EQ(dm.num_hl, 2);
  EXPECT_EQ(dm.num_hu, 2);
  EXPECT_EQ(dm.num_xl, 3);
  EXPECT_EQ(dm.num_xu, 3);
  EXPECT_EQ(dm.nnz_P, 5);
  EXPECT_EQ(dm.nnz_A, 2);
  EXPECT_EQ(dm.nnz_G, 3);
  auto view = d.view();
  EXPECT_EQ(download(view.P.indptr, 4), q.P.indptr);
  EXPECT_EQ(download(view.P.indices, 5), q.P.indices);
  EXPECT_EQ(download(view.P.row_idx, 5), (std::vector<int32_t>{0, 0, 1, 2, 2}));
  EXPECT_EQ(download(view.G.row_idx, 3), (std::vector<int32_t>{0, 1, 1}));
  EXPECT_EQ(download(view.A.row_idx, 2), (std::vector<int32_t>{0, 0}));
  // Explicit zeros keep their slot, in stored-entry order.
  expect_bits(download(view.P.values, 5), q.Pv, "P");
  expect_bits(download(view.A.values, 2), q.Av, "A");
  expect_bits(download(view.G.values, 3), q.Gv, "G");
  expect_bits(download(view.c, 3), q.c, "c");
  expect_bits(download(view.b, 1), q.b, "b");
  expect_bits(download(view.h_l, 2), q.hl, "h_l");
  expect_bits(download(view.x_u, 3), q.xu, "x_u");
  expect_masks(d, cpu_masks(2, 3, &q.hl, &q.hu, &q.xl, &q.xu));

  // Explicit zeros become nonzero in place.
  std::vector<double> P2{4, 0.5, 3, 0.5, 5};
  Dev<double> P2d(P2);
  ValuesView u;
  u.P = P2d.ptr();
  UpdateMask mk = d.set_values(u, nullptr);
  EXPECT_TRUE(mk.P);
  EXPECT_FALSE(mk.c || mk.A || mk.b || mk.G || mk.h_l || mk.h_u || mk.x_l || mk.x_u);
  CUPIQP_CUDA_CHECK(cudaDeviceSynchronize());
  expect_bits(download(view.P.values, 5), P2, "P after update");
  expect_bits(download(view.c, 3), q.c, "c untouched");
}

// ------------------------------------------------------------------ masks
TEST(FiniteMasks, AllFiniteOneSidedMixedAndSwitches) {
  Problem q;
  std::vector<double> hl{-1, -2}, hu{1, 2}, xl{-1, -1, -1}, xu{1, 1, 1};
  Dev<double> Pv(q.Pv), c(q.c), Gv(q.Gv), hld(hl), hud(hu), xld(xl), xud(xu);

  // All finite, every side.
  {
    SparseData d;
    d.init(q.P, nullptr, &q.G, {Pv.ptr(), c.ptr(), nullptr, nullptr, Gv.ptr(), hld.ptr(), hud.ptr(), xld.ptr(), xud.ptr()}, nullptr);
    expect_masks(d, cpu_masks(2, 3, &hl, &hu, &xl, &xu));
    EXPECT_EQ(download(d.view().num_finite_bounds, 1)[0], 10.0);
  }
  // One-sided: only h_u and x_l (layout [hu | xl]).
  {
    SparseData d;
    d.init(q.P, nullptr, &q.G, {Pv.ptr(), c.ptr(), nullptr, nullptr, Gv.ptr(), nullptr, hud.ptr(), xld.ptr(), nullptr}, nullptr);
    EXPECT_EQ(d.dims().num_ineq(), 5);
    EXPECT_EQ(d.dims().off_xl(), 2);
    expect_masks(d, cpu_masks(2, 3, nullptr, &hu, &xl, nullptr));
    EXPECT_FALSE(d.has_h_l());
    EXPECT_TRUE(d.has_x_l());
  }
  // Mixed and edge values: +-inf, +-1e20, values just inside, NaN, a lower
  // bound of +inf and an upper bound of -inf (the reference counts those as
  // finite: only the one-sided comparison is applied).
  {
    std::vector<double> hl2{-kInf, kNaN}, hu2{1e20, -kInf};
    std::vector<double> xl2{-1e20, -9.999999999999999e19, kInf}, xu2{kNaN, kInf, 9.999999999999999e19};
    Dev<double> a(hl2), bb(hu2), cc(xl2), dd(xu2);
    SparseData d;
    d.init(q.P, nullptr, &q.G, {Pv.ptr(), c.ptr(), nullptr, nullptr, Gv.ptr(), a.ptr(), bb.ptr(), cc.ptr(), dd.ptr()}, nullptr);
    CpuMasks want = cpu_masks(2, 3, &hl2, &hu2, &xl2, &xu2);
    expect_masks(d, want);
    // Spot-check the reference semantics explicitly.
    EXPECT_EQ(want.all, (std::vector<double>{0, 0, 0, 1, 0, 1, 1, 0, 0, 1}));
    EXPECT_EQ(want.gRow, (std::vector<double>{0, 1}));
    EXPECT_EQ(want.xBound, (std::vector<double>{0, 1, 1}));
    EXPECT_EQ(want.count, 4.0);

    // Finite -> infinite and back through set_values (masks refresh).
    std::vector<double> hl3{-3, -3}, xu3{kInf, kInf, kInf};
    Dev<double> e(hl3), f(xu3);
    ValuesView u;
    u.h_l = e.ptr();
    u.x_u = f.ptr();
    UpdateMask mk = d.set_values(u, nullptr);
    EXPECT_TRUE(mk.h_l && mk.x_u);
    EXPECT_TRUE(mk.any_ineq_bound() && mk.any_box_bound());
    expect_masks(d, cpu_masks(2, 3, &hl3, &hu2, &xl2, &xu3));
    // And to all finite.
    ValuesView u2{nullptr, nullptr, nullptr, nullptr, nullptr, hld.ptr(), hud.ptr(), xld.ptr(), xud.ptr()};
    d.set_values(u2, nullptr);
    expect_masks(d, cpu_masks(2, 3, &hl, &hu, &xl, &xu));
  }
}

TEST(FiniteMasks, BlockKernelMatchesReferenceKernelOnLargeRandom) {
  const int n = 5000, m = 3000;
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> U(0, 1);
  auto vec = [&](int k, bool lower) {
    std::vector<double> v(k);
    for (auto& x : v) {
      double r = U(rng);
      x = r < 0.2 ? (lower ? -kInf : kInf) : r < 0.25 ? (lower ? -1e20 : 1e20) : r < 0.27 ? kNaN : (lower ? -r : r);
    }
    return v;
  };
  auto hl = vec(m, true), hu = vec(m, false), xl = vec(n, true), xu = vec(n, false);
  CsrPattern P = pat(n, n, {}, {});
  P.indptr.resize(n + 1);
  for (int i = 0; i <= n; ++i) P.indptr[i] = i;
  P.indices.resize(n);
  for (int i = 0; i < n; ++i) P.indices[i] = i;
  CsrPattern G = pat(m, n, {}, {});
  G.indptr.resize(m + 1);
  for (int i = 0; i <= m; ++i) G.indptr[i] = i;
  G.indices.resize(m);
  for (int i = 0; i < m; ++i) G.indices[i] = i;
  std::vector<double> ones(n, 1.0);
  Dev<double> Pv(ones), c(ones), Gv(std::vector<double>(m, 1.0)), a(hl), b(hu), cc(xl), d4(xu);
  SparseData d;
  d.init(P, nullptr, &G, {Pv.ptr(), c.ptr(), nullptr, nullptr, Gv.ptr(), a.ptr(), b.ptr(), cc.ptr(), d4.ptr()}, nullptr);
  CpuMasks want = cpu_masks(m, n, &hl, &hu, &xl, &xu);
  expect_masks(d, want);
  d.update_finite_masks_reference(nullptr);
  expect_masks(d, want);
}

// ------------------------------------------------------------------ set_values
TEST(SetValues, RejectsAbsentBoundSideWithoutSideEffects) {
  Problem q;
  Dev<double> Pv(q.Pv), c(q.c), Gv(q.Gv), hu(q.hu), xl(q.xl);
  SparseData d;
  d.init(q.P, nullptr, &q.G, {Pv.ptr(), c.ptr(), nullptr, nullptr, Gv.ptr(), nullptr, hu.ptr(), nullptr, nullptr}, nullptr);
  std::vector<double> c2{9, 9, 9};
  Dev<double> c2d(c2);
  ValuesView u;
  u.c = c2d.ptr();
  u.x_l = xl.ptr();
  expect_invalid([&] { d.set_values(u, nullptr); }, "Cannot set x_l: no x_l block was provided at setup()");
  u.x_l = nullptr;
  u.h_l = hu.ptr();
  expect_invalid([&] { d.set_values(u, nullptr); }, "Cannot set h_l");
  CUPIQP_CUDA_CHECK(cudaDeviceSynchronize());
  expect_bits(download(d.view().c, 3), q.c, "c unchanged after a rejected update");
  SparseData fresh;
  expect_invalid([&] { fresh.set_values(u, nullptr); }, "init() must be called");
}

TEST(SetValues, FromHostPinnedAndPageable) {
  Problem q;
  Dev<double> Pv(q.Pv), c(q.c), Av(q.Av), b(q.b), Gv(q.Gv), hl(q.hl), hu(q.hu), xl(q.xl), xu(q.xu);
  SparseData d;
  d.init(q.P, &q.A, &q.G, {Pv.ptr(), c.ptr(), Av.ptr(), b.ptr(), Gv.ptr(), hl.ptr(), hu.ptr(), xl.ptr(), xu.ptr()}, nullptr);
  // Pageable.
  std::vector<double> G2{7, 8, 9}, hu2{-kInf, kInf};
  ValuesView u;
  u.G = G2.data();
  u.h_u = hu2.data();
  UpdateMask mk = d.set_values_from_host(u, nullptr);
  EXPECT_TRUE(mk.G && mk.h_u && !mk.P);
  CUPIQP_CUDA_CHECK(cudaDeviceSynchronize());
  expect_bits(download(d.view().G.values, 3), G2, "G");
  expect_masks(d, cpu_masks(2, 3, &q.hl, &hu2, &q.xl, &q.xu));
  // Pinned.
  CudaPinnedBuffer<double> pin(3);
  pin[0] = -5;
  pin[1] = kNaN;
  pin[2] = 1e20;
  ValuesView u2;
  u2.x_l = pin.data();
  d.set_values_from_host(u2, nullptr);
  std::vector<double> xl2{-5, kNaN, 1e20};
  expect_masks(d, cpu_masks(2, 3, &q.hl, &hu2, &xl2, &q.xu));
}

TEST(SetValues, RepeatedUpdatesKeepAddressesAndAllocateNothing) {
  Problem q;
  Dev<double> Pv(q.Pv), c(q.c), Av(q.Av), b(q.b), Gv(q.Gv), hl(q.hl), hu(q.hu), xl(q.xl), xu(q.xu);
  std::vector<double> hl_inf{-kInf, -kInf};
  Dev<double> hlinf(hl_inf);
  SparseData d;
  ValuesView all{Pv.ptr(), c.ptr(), Av.ptr(), b.ptr(), Gv.ptr(), hl.ptr(), hu.ptr(), xl.ptr(), xu.ptr()};
  d.init(q.P, &q.A, &q.G, all, nullptr);
  CUPIQP_CUDA_CHECK(cudaDeviceSynchronize());
  const auto v0 = d.view();
  size_t free0 = 0, total0 = 0;
  CUPIQP_CUDA_CHECK(cudaMemGetInfo(&free0, &total0));
  CudaPinnedBuffer<double> pin(3);
  for (int i = 0; i < 3; ++i) pin[i] = i;
  for (int it = 0; it < 200; ++it) {
    ValuesView u = all;
    u.h_l = (it % 2) ? hlinf.ptr() : hl.ptr();
    EXPECT_TRUE(d.set_values(u, nullptr).any_matrix());
    ValuesView h;
    h.c = pin.data();
    d.set_values_from_host(h, nullptr);
  }
  CUPIQP_CUDA_CHECK(cudaDeviceSynchronize());
  size_t free1 = 0, total1 = 0;
  CUPIQP_CUDA_CHECK(cudaMemGetInfo(&free1, &total1));
  // The GPU is shared with other processes: require no growth from this one by
  // checking that free memory did not shrink by an allocation granule.
  EXPECT_GE(free1 + (2u << 20), free0);
  const auto v1 = d.view();
  EXPECT_EQ(v0.P.values, v1.P.values);
  EXPECT_EQ(v0.A.values, v1.A.values);
  EXPECT_EQ(v0.G.values, v1.G.values);
  EXPECT_EQ(v0.c, v1.c);
  EXPECT_EQ(v0.b, v1.b);
  EXPECT_EQ(v0.h_l, v1.h_l);
  EXPECT_EQ(v0.h_u, v1.h_u);
  EXPECT_EQ(v0.x_l, v1.x_l);
  EXPECT_EQ(v0.x_u, v1.x_u);
  EXPECT_EQ(v0.finite_mask_all, v1.finite_mask_all);
  EXPECT_EQ(v0.active_G_row, v1.active_G_row);
  EXPECT_EQ(v0.active_x_bound, v1.active_x_bound);
  EXPECT_EQ(v0.num_finite_bounds, v1.num_finite_bounds);
  EXPECT_EQ(v0.P.indptr, v1.P.indptr);
  EXPECT_EQ(v0.P.row_idx, v1.P.row_idx);
  // Last iteration (it = 199) used the infinite h_l.
  expect_masks(d, cpu_masks(2, 3, &hl_inf, &q.hu, &q.xl, &q.xu));
}

TEST(SparseDataInit, InequalityBlockWithZeroRows) {
  // G with zero rows: the bound vectors are empty and may be nullptr.
  Problem q;
  Dev<double> Pv(q.Pv), c(q.c), xu(q.xu);
  CsrPattern G0 = pat(0, 3, {0}, {});
  SparseData d;
  d.init(q.P, nullptr, &G0, {Pv.ptr(), c.ptr(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, xu.ptr()}, nullptr);
  EXPECT_EQ(d.dims().m, 0);
  EXPECT_EQ(d.dims().num_ineq(), 3);
  expect_masks(d, cpu_masks(0, 3, nullptr, nullptr, nullptr, &q.xu));
}
