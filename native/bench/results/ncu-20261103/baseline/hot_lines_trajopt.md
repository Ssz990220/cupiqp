## trajopt assemble_kernel
total samples 5973
| line | samples % | warp insts | source |
|---|---|---|---|
| 115 | 43.6 | 4299223 | `for (int t = lane; t < cs; t += 32) acc[Cc[kbs + t]] += as * Cv[kbs + t];` |
| 121 | 13.8 | 30306 | `for (int j = tid; j <= i; j += blockDim.x) {` |
| 124 | 5.7 | 137082 | `for (int v = 0; v < W; ++v) s += sh[v * n + j];` |
| 105 | 5.1 | 68096 | `a = w[k] * K[kb + off];` |
| 373 | 4.9 | 1170616 | `ret = __nvvm_shfl_idx_sync(mask, var, srcLane, c);` |
| 87 | 4.2 | 1015105 | `for (int t = tid; t < (W + 1) * (i + 1); t += blockDim.x) sh[t % (i + 1) + (t / (i + 1)) *` |
| 104 | 3.1 | 19456 | `kb = Kbase[k];` |
| 109 | 2.1 | 468327 | `while (mask) {` |
| 110 | 2.1 | 292654 | `return __nvvm_bar_warp_sync(mask);` |
| 116 | 2.1 | 146327 | `__syncwarp();` |

## trajopt chol_panel
total samples 490
| line | samples % | warp insts | source |
|---|---|---|---|
| 227 | 25.5 | 5355 | `for (int t = c + 1; t < kTile; ++t) x[t] = fma(-x[c], Ls[t][c], x[t]);` |
| 373 | 14.7 | 8904 | `ret = __nvvm_shfl_idx_sync(mask, var, srcLane, c);` |
| 149 | 11.4 | 1536 | `double y = double(rsqrtf(float(d)));` |
| 150 | 7.3 | 1024 | `y = fma(y, fma(-0.5 * d * y, y, 0.5), y);` |
| 179 | 6.9 | 7694 | `if (lane >= t) a[t] = fma(-a[j], ltj, a[t]);` |
| 151 | 5.3 | 768 | `y = fma(y, fma(-0.5 * d * y, y, 0.5), y);` |
| 148 | 5.1 | 696 | `if (d > 1e-36 && d < 1e36) {` |
| 168 | 3.9 | 2688 | `if (fail == 0 && g0 + j < n && !(d > 0.0 && d <= 1.7976931348623157e308)) fail = g0 + j + ` |
| 198 | 3.3 | 2898 | `a[t] = gi < n ? (t <= lane ? H[size_t(g0 + t) * n + gi] : 0.0) : (t == lane ? 1.0 : 0.0);` |
| 232 | 2.7 | 1008 | `if (g0 + t < n) H[size_t(g0 + t) * n + r] = x[t];` |

## trajopt chol_update
total samples 673
| line | samples % | warp insts | source |
|---|---|---|---|
| 259 | 62.3 | 105720 | `for (int l = 0; l < kTile; ++l) v = fma(Li[tx][l], Lj[ty][l], v);` |
| 247 | 12.5 | 70784 | `Li[tx][l] = (ri + tx < n && gc < n) ? H[size_t(gc) * n + ri + tx] : 0.0;` |
| 260 | 10.4 | 6167 | `H[size_t(gcol) * n + gr] -= v;` |
| 239 | 3.7 | 1568 | `__global__ void __launch_bounds__(kTile * kSub) chol_update_kernel(double* __restrict__ H,` |
| 252 | 3.0 | 12544 | `Lj[c][l] = (rj + c < n && gc < n) ? H[size_t(gc) * n + rj + c] : 0.0;` |
| 244 | 3.0 | 9856 | `const int tx = threadIdx.x, ty = threadIdx.y;  // tx: row within tile, ty: column within s` |
| 256 | 1.9 | 3584 | `if (gr < n && gcol < n && gr >= gcol) {` |
| 262 | 0.7 | 881 | `}` |
| 241 | 0.7 | 3136 | `if (jt > it) return;` |
| 245 | 0.6 | 14336 | `for (int l = ty; l < kTile; l += kSub) {` |

## trajopt recover
total samples 1382
| line | samples % | warp insts | source |
|---|---|---|---|
| 393 | 42.5 | 355996 | `for (int q = lane; q < len; q += 32) s = fma(K[kb + q], x[Kidx[kb + q]], s);` |
| 395 | 26.6 | 83837 | `if (lane == 0) x[n + k] = w[k] * (s - b[n + k]);` |
| 409 | 16.2 | 64490 | `ret = __nvvm_shfl_bfly_sync(mask, var, laneMask, c);` |
| 22 | 5.6 | 38694 | `for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);` |
| 385 | 2.2 | 6456 | `__global__ void recover_kernel(const double* __restrict__ K, const int32_t* __restrict__ K` |
| 396 | 2.0 | 6449 | `}` |
| 391 | 1.9 | 64490 | `const int kb = Kbase[k], len = Mlen[k];` |
| 390 | 1.7 | 25810 | `if (k >= mm) return;` |
| 536 | 0.4 | 6449 | `hi = __shfl_xor_sync(mask, hi, laneMask, width);` |
| 414 | 0.4 | 12898 | `return (unsigned int) __shfl_xor_sync(mask, (int)var, laneMask, width);` |

## trajopt reduce_rhs
total samples 1044
| line | samples % | warp insts | source |
|---|---|---|---|
| 277 | 69.3 | 56774 | `s = fma(K[Cslot[e]], w[k] * b[n + k], s);` |
| 275 | 10.9 | 54573 | `for (int e = Cptr[i] + tid; e < Cptr[i + 1]; e += kRhsBlock) {` |
| 282 | 8.9 | 1928 | `if (tid == 0) {` |
| 287 | 2.9 | 723 | `x[i] = t + sum;` |
| 409 | 2.2 | 9640 | `ret = __nvvm_shfl_bfly_sync(mask, var, laneMask, c);` |
| 268 | 1.3 | 964 | `__global__ void __launch_bounds__(kRhsBlock) reduce_rhs_kernel(` |
| 22 | 1.1 | 5784 | `for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);` |
| 286 | 1.0 | 1446 | `for (int v = 0; v < kRhsBlock / 32; ++v) sum += part[v];` |
| 280 | 0.6 | 4820 | `if (tid % 32 == 0) part[tid / 32] = s;` |
| 276 | 0.6 | 8934 | `const int k = Crow[e];` |

## trajopt trsv
total samples 241
| line | samples % | warp insts | source |
|---|---|---|---|
| 330 | 41.1 | 5776 | `for (int c = 0; c < kSeg; ++c) xv[c] = c0 + c < n ? xs[c0 + c] : 0.0;` |
| 366 | 15.4 | 4480 | `for (int c = 0; c < kSeg; ++c) xv[c] = c0 + c < n ? xs[c0 + c] : 0.0;` |
| 337 | 6.2 | 5732 | `if (c0 + c < n) s = fma(L[size_t(c0 + c) * n + i], xv[c], s);` |
| 373 | 5.0 | 2776 | `if (c0 + c < n) s = fma(L[size_t(j) * n + c0 + c], xv[c], s);` |
| 346 | 4.1 | 1360 | `for (int kk = N - 1; kk >= 0; --kk) {` |
| 409 | 3.7 | 1968 | `ret = __nvvm_shfl_bfly_sync(mask, var, laneMask, c);` |
| 309 | 3.7 | 528 | `for (int kk = 0; kk < N; ++kk) {` |
| 414 | 3.3 | 1344 | `return (unsigned int) __shfl_xor_sync(mask, (int)var, laneMask, width);` |
| 536 | 2.9 | 624 | `hi = __shfl_xor_sync(mask, hi, laneMask, width);` |
| 370 | 2.1 | 2320 | `if (j < g0) {` |

