// ipm/solver: device-side warm start from the previous solve. See
// warm_start_device.hpp.
#include <cuda_runtime.h>

#include <cub/device/device_radix_sort.cuh>
#include <cuda/std/tuple>

#include <algorithm>

#include "common/check.hpp"
#include "common/reduce.cuh"
#include "ipm/solver/warm_start_device.hpp"

namespace cupiqp {

namespace {

constexpr int kBlock = 256;

// (group, key), group most significant.
struct WarmRowKeyDecomposer {
  __host__ __device__ cuda::std::tuple<int32_t&, int64_t&> operator()(WarmRowKey& k) const {
    return {k.group, k.key};
  }
};

__device__ __forceinline__ bool key_less(const WarmRowKey& a, int32_t g, int64_t k) {
  return a.group < g || (a.group == g && a.key < k);
}

__global__ void iota_kernel(int32_t* p, int32_t n) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = i;
}

// Pack the current keys as the previous keys of the next solve.
__global__ void pack_keys_kernel(const int64_t* keys, const int32_t* groups, WarmRowKey* out, int32_t n) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  WarmRowKey k;
  k.key = keys[i];
  k.group = groups ? groups[i] : 0;
  k.pad = 0;
  out[i] = k;
}

// match[i]: i itself when its previous key is equal, else the first equal
// entry of the sorted previous keys (stable sort: lowest row), else -1.
__global__ void match_kernel(const int64_t* keys, const int32_t* groups, const WarmRowKey* prev,
                             const WarmRowKey* sorted, const int32_t* sorted_rows, int32_t* match, int32_t n) {
  int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t k = keys[i];
  const int32_t g = groups ? groups[i] : 0;
  const WarmRowKey self = prev[i];
  if (self.key == k && self.group == g) {
    match[i] = i;
    return;
  }
  int32_t lo = 0, hi = n;  // lower_bound
  while (lo < hi) {
    const int32_t mid = (lo + hi) >> 1;
    if (key_less(sorted[mid], g, k)) lo = mid + 1;
    else hi = mid;
  }
  match[i] = (lo < n && sorted[lo].key == k && sorted[lo].group == g) ? sorted_rows[lo] : -1;
}

__global__ void decide_kernel(const int32_t* prev_status, int32_t cold_if_not_solved, int32_t* cond) {
  const bool warm = !cold_if_not_solved || prev_status[0] == 0;  // 0 = STATUS_SOLVED
  cond[0] = warm ? 0 : 1;
  cond[1] = warm ? 1 : 0;
}

struct StartArgs {
  int32_t n, p, m, num_hl, num_hu, num_xl, num_xu;
  int32_t key_begin, key_count;
  const int32_t* match;  // nullptr: identity
  const double* prev_dual;
  const double *h_l, *h_u, *x_l, *x_u;
  const double* finite_mask_all;
  const double *delta, *delta_inv, *delta_b, *delta_b_inv, *cost_scaling;  // nullptr: unscaled
  double s_min, z_min, z_unmatched;
  int32_t previous_y;
  double *x, *y, *s_all, *z_all;
};

// One thread per entry of [x | y | s_all/z_all]. Scaled coordinates:
// s_sc = s * D, z_sc = z * D^-1 * c, y_sc = y * D^-1 * c, with D the row
// factor (delta for G rows, delta_b for box rows) and c the cost scaling
// (RuizEquilibration::scale_solution). The bounds in the data are already
// scaled, so the slack at x = 0 is the scaled bound itself.
__global__ void build_start_kernel(StartArgs a) {
  const int32_t ni = a.num_hl + a.num_hu + a.num_xl + a.num_xu;
  int32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.n + a.p + ni) return;
  const double c = a.cost_scaling ? a.cost_scaling[0] : 1.0;
  if (t < a.n) {
    a.x[t] = 0.0;
    return;
  }
  t -= a.n;
  if (t < a.p) {
    const double yd = a.delta_inv ? a.delta_inv[a.n + t] : 1.0;
    a.y[t] = a.previous_y ? a.prev_dual[t] * yd * c : 0.0;
    return;
  }
  const int32_t j = t - a.p;  // packed inequality index [hl | hu | xl | xu]
  if (!(a.finite_mask_all[j] > 0.5)) {
    a.s_all[j] = 0.0;
    a.z_all[j] = 0.0;
    return;
  }
  const double* zprev = a.prev_dual + a.p;
  double s0, dfac, dinv, zprev_v;
  if (j < a.num_hl + a.num_hu) {
    const bool upper = j >= a.num_hl;
    const int32_t k = upper ? j - a.num_hl : j;
    const int32_t base = upper ? a.num_hl : 0;
    int32_t src = k;
    if (a.match && k >= a.key_begin && k < a.key_begin + a.key_count) {
      const int32_t mm = a.match[k - a.key_begin];
      src = mm >= 0 ? mm + a.key_begin : -1;
    }
    zprev_v = src >= 0 ? zprev[base + src] : a.z_unmatched;
    s0 = upper ? a.h_u[k] : -a.h_l[k];
    dfac = a.delta ? a.delta[a.n + a.p + k] : 1.0;
    dinv = a.delta_inv ? a.delta_inv[a.n + a.p + k] : 1.0;
  } else {
    const bool upper = j >= a.num_hl + a.num_hu + a.num_xl;
    const int32_t k = upper ? j - (a.num_hl + a.num_hu + a.num_xl) : j - (a.num_hl + a.num_hu);
    zprev_v = zprev[j];
    s0 = upper ? a.x_u[k] : -a.x_l[k];
    dfac = a.delta_b ? a.delta_b[k] : 1.0;
    dinv = a.delta_b_inv ? a.delta_b_inv[k] : 1.0;
  }
  a.s_all[j] = fmax(s0, a.s_min * dfac);
  a.z_all[j] = fmax(zprev_v, a.z_min) * dinv * c;
}

// Ring slot iter % slots <- [y | z_all | s_all] of the current iterate.
__global__ void ring_store_kernel(const double* dual, const double* s_all, int32_t num_dual, int32_t ni,
                                  const int32_t* current_iter, int32_t slots, double* ring) {
  int32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t w = num_dual + ni;
  if (t >= w) return;
  double* slot = ring + int64_t(current_iter[0] % slots) * w;
  slot[t] = t < num_dual ? dual[t] : s_all[t - num_dual];
}

struct RecordArgs {
  int32_t n, p, num_hl, num_hu, num_xl, num_xu;
  const double* dual;  // end-of-solve [y | z_all], scaled
  const double* ring;  // nullptr without snapshot
  int32_t slots, back;
  const int32_t* current_iter;
  const int32_t* status;
  const double *delta, *delta_b, *cost_scaling_inv;  // nullptr: unscaled
  double* prev_dual;
  int32_t* prev_status;
};

// Unscale (RuizEquilibration::unscale_solution): y = y_sc * D * c^-1,
// z = z_sc * D * c^-1.
__global__ void record_previous_kernel(RecordArgs a) {
  const int32_t ni = a.num_hl + a.num_hu + a.num_xl + a.num_xu;
  const int32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t nd = a.p + ni;
  if (t == 0) a.prev_status[0] = a.status[0];
  if (t >= nd) return;
  const double* src = a.dual;
  if (a.ring && a.back > 0) {
    const int32_t it = a.current_iter[0] - a.back;
    if (it >= 1) src = a.ring + int64_t(it % a.slots) * (nd + ni);
  }
  const double ci = a.cost_scaling_inv ? a.cost_scaling_inv[0] : 1.0;
  double f = 1.0;
  if (t < a.p) {
    if (a.delta) f = a.delta[a.n + t];
  } else {
    const int32_t j = t - a.p;
    if (j < a.num_hl + a.num_hu) {
      const int32_t k = j < a.num_hl ? j : j - a.num_hl;
      if (a.delta) f = a.delta[a.n + a.p + k];
    } else {
      const int32_t k = j < a.num_hl + a.num_hu + a.num_xl ? j - (a.num_hl + a.num_hu)
                                                           : j - (a.num_hl + a.num_hu + a.num_xl);
      if (a.delta_b) f = a.delta_b[k];
    }
  }
  a.prev_dual[t] = a.delta ? src[t] * f * ci : src[t];
}

}  // namespace

void WarmStartDevice::configure(const Dims& d, const WarmStartOptions& o, cudaStream_t s) {
  CUPIQP_REQUIRE(d.B == 1, "set_warm_start(): batch size 1 only.");
  CUPIQP_REQUIRE(o.s_min > 0.0 && o.z_min > 0.0 && o.z_unmatched >= 0.0,
                 "set_warm_start(): s_min and z_min must be positive, z_unmatched nonnegative.");
  CUPIQP_REQUIRE(o.snapshot_slots >= 0 && o.snapshot_back >= 0 && o.snapshot_back <= o.snapshot_slots,
                 "set_warm_start(): need 0 <= snapshot_back <= snapshot_slots.");
  int32_t begin = 0, count = 0;
  if (o.row_match == RowMatch::Keys) {
    begin = o.key_row_begin;
    count = o.key_row_count < 0 ? d.m - begin : o.key_row_count;
    CUPIQP_REQUIRE(begin >= 0 && count >= 0 && begin + count <= d.m && d.m > 0,
                   "set_warm_start(): keyed rows must lie in [0, m).");
  }
  const bool first = !configured_;
  if (first) {
    prev_dual_.allocate(static_cast<size_t>(d.num_dual()));
    prev_status_.allocate(1);
    cond_.allocate(2);
    host_flag_.allocate(1);
    prev_dual_.zero_async(s);
    prev_status_.zero_async(s);
    cond_.zero_async(s);
  }
  // Keys: reallocate when the keyed range or the group flag changes.
  const bool groups = o.row_match == RowMatch::Keys && o.row_groups;
  if (first || count != key_count_ || groups != (groups_.size() > 0) ||
      (o.row_match == RowMatch::Keys) != keyed()) {
    keys_ = CudaBuffer<int64_t>(static_cast<size_t>(count));
    groups_ = CudaBuffer<int32_t>(groups ? static_cast<size_t>(count) : 0);
    keys_host_ = CudaPinnedBuffer<int64_t>(static_cast<size_t>(count));
    groups_host_ = CudaPinnedBuffer<int32_t>(groups ? static_cast<size_t>(count) : 0);
    prev_keys_ = CudaBuffer<WarmRowKey>(static_cast<size_t>(count));
    sorted_keys_ = CudaBuffer<WarmRowKey>(static_cast<size_t>(count));
    iota_ = CudaBuffer<int32_t>(static_cast<size_t>(count));
    sorted_rows_ = CudaBuffer<int32_t>(static_cast<size_t>(count));
    match_ = CudaBuffer<int32_t>(static_cast<size_t>(count));
    keys_.zero_async(s);
    groups_.zero_async(s);
    prev_keys_.zero_async(s);
    if (count) {
      for (int32_t i = 0; i < count; ++i) keys_host_.data()[i] = 0;
      for (size_t i = 0; i < groups_host_.size(); ++i) groups_host_.data()[i] = 0;
      iota_kernel<<<grid_for(count, kBlock), kBlock, 0, s>>>(iota_.data(), count);
      CUPIQP_CUDA_CHECK_LAUNCH();
      size_t bytes = 0;
      CUPIQP_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, bytes, prev_keys_.data(), sorted_keys_.data(),
                                                        iota_.data(), sorted_rows_.data(), count,
                                                        WarmRowKeyDecomposer{}, s));
      sort_temp_ = CudaBuffer<unsigned char>(std::max<size_t>(bytes, 1));
      sort_temp_bytes_ = bytes;
    } else {
      sort_temp_ = CudaBuffer<unsigned char>();
      sort_temp_bytes_ = 0;
    }
  }
  slot_width_ = d.num_dual() + d.num_ineq();
  const size_t ring_len = o.snapshot_slots > 0 ? static_cast<size_t>(o.snapshot_slots + 1) * slot_width_ : 0;
  if (ring_.size() != ring_len) {
    ring_ = CudaBuffer<double>(ring_len);
    ring_.zero_async(s);
  }
  d_ = d;
  opt_ = o;
  key_begin_ = begin;
  key_count_ = count;
  configured_ = true;
}

RowKeyBuffers WarmStartDevice::key_buffers(bool host) {
  RowKeyBuffers b;
  b.count = key_count_;
  b.keys = host ? keys_host_.data() : keys_.data();
  b.groups = groups_.size() ? (host ? groups_host_.data() : groups_.data()) : nullptr;
  return b;
}

void WarmStartDevice::upload_keys(cudaStream_t s) {
  if (!key_count_) return;
  CUPIQP_CUDA_CHECK(cudaMemcpyAsync(keys_.data(), keys_host_.data(), sizeof(int64_t) * key_count_,
                                    cudaMemcpyHostToDevice, s));
  if (groups_.size())
    CUPIQP_CUDA_CHECK(cudaMemcpyAsync(groups_.data(), groups_host_.data(), sizeof(int32_t) * key_count_,
                                      cudaMemcpyHostToDevice, s));
}

void WarmStartDevice::match_rows(cudaStream_t s) {
  if (!key_count_) return;
  size_t bytes = sort_temp_bytes_;
  CUPIQP_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(sort_temp_.data(), bytes, prev_keys_.data(), sorted_keys_.data(),
                                                    iota_.data(), sorted_rows_.data(), key_count_,
                                                    WarmRowKeyDecomposer{}, s));
  match_kernel<<<grid_for(key_count_, kBlock), kBlock, 0, s>>>(
      keys_.data(), groups_.size() ? groups_.data() : nullptr, prev_keys_.data(), sorted_keys_.data(),
      sorted_rows_.data(), match_.data(), key_count_);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void WarmStartDevice::commit_keys(cudaStream_t s) {
  if (!key_count_) return;
  pack_keys_kernel<<<grid_for(key_count_, kBlock), kBlock, 0, s>>>(
      keys_.data(), groups_.size() ? groups_.data() : nullptr, prev_keys_.data(), key_count_);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void WarmStartDevice::decide(cudaStream_t s) {
  decide_kernel<<<1, 1, 0, s>>>(prev_status_.data(), opt_.cold_if_not_solved ? 1 : 0, cond_.data());
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void WarmStartDevice::build_start(cudaStream_t s, const SparseDataView<double>& data, const ScalingView<double>& sc,
                                  bool scaled, const VariablesView<double>& r) {
  StartArgs a{};
  a.n = d_.n; a.p = d_.p; a.m = d_.m;
  a.num_hl = d_.num_hl; a.num_hu = d_.num_hu; a.num_xl = d_.num_xl; a.num_xu = d_.num_xu;
  a.key_begin = key_begin_;
  a.key_count = key_count_;
  a.match = key_count_ ? match_.data() : nullptr;
  a.prev_dual = prev_dual_.data();
  a.h_l = data.h_l; a.h_u = data.h_u; a.x_l = data.x_l; a.x_u = data.x_u;
  a.finite_mask_all = data.finite_mask_all;
  if (scaled) {
    a.delta = sc.delta; a.delta_inv = sc.delta_inv; a.delta_b = sc.delta_b; a.delta_b_inv = sc.delta_b_inv;
    a.cost_scaling = sc.cost_scaling;
  }
  a.s_min = opt_.s_min; a.z_min = opt_.z_min; a.z_unmatched = opt_.z_unmatched;
  a.previous_y = opt_.previous_y ? 1 : 0;
  a.x = r.x(); a.y = r.y(); a.s_all = r.s_all(); a.z_all = r.z_all();
  const int64_t total = int64_t(d_.n) + d_.p + d_.num_ineq();
  build_start_kernel<<<grid_for(total, kBlock), kBlock, 0, s>>>(a);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void WarmStartDevice::ring_store(cudaStream_t s, const VariablesView<double>& r, const int32_t* current_iter) {
  if (!ring_enabled()) return;
  ring_store_kernel<<<grid_for(slot_width_, kBlock), kBlock, 0, s>>>(
      r.dual, r.s_all(), d_.num_dual(), d_.num_ineq(), current_iter, opt_.snapshot_slots + 1, ring_.data());
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void WarmStartDevice::record_previous(cudaStream_t s, const VariablesView<double>& r, const ScalingView<double>& sc,
                                      bool scaled, const int32_t* current_iter, const int32_t* status) {
  RecordArgs a{};
  a.n = d_.n; a.p = d_.p;
  a.num_hl = d_.num_hl; a.num_hu = d_.num_hu; a.num_xl = d_.num_xl; a.num_xu = d_.num_xu;
  a.dual = r.dual;
  a.ring = ring_enabled() ? ring_.data() : nullptr;
  a.slots = opt_.snapshot_slots + 1;
  a.back = opt_.snapshot_back;
  a.current_iter = current_iter;
  a.status = status;
  if (scaled) {
    a.delta = sc.delta; a.delta_b = sc.delta_b; a.cost_scaling_inv = sc.cost_scaling_inv;
  }
  a.prev_dual = prev_dual_.data();
  a.prev_status = prev_status_.data();
  const int32_t nd = d_.num_dual();
  record_previous_kernel<<<grid_for(std::max(nd, 1), kBlock), kBlock, 0, s>>>(a);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

}  // namespace cupiqp
