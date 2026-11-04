# remove-cudss-20261104: how these results were produced

Build: fresh CMake build directory `build/native-nocudss` of commit
`70abe7f` (feature/remove-cudss), RTX 4070 SUPER, GPU otherwise idle, runs
one at a time, same WSL session (2026-11-04).

```bash
# 1. Auto graph check (one configuration per dataset, library defaults)
BUILD=build/native-nocudss bash native/bench/tools/run_auto_graph_check.sh \
  native/bench/results/remove-cudss-20261104 20
#    -> {mpc,trajopt}/auto, {mpc,trajopt}-auto.log

# 2. Harness path (Python shim), default settings
.venv/bin/python native/bench/harness/replay_updates.py --backend native --dataset mpc --repeats 20 \
  --native-build build/native-nocudss --output native/bench/results/remove-cudss-20261104/mpc/harness-native
.venv/bin/python native/bench/harness/replay_updates.py --backend native --dataset trajopt --repeats 20 \
  --native-build build/native-nocudss --output native/bench/results/remove-cudss-20261104/trajopt/harness-native
#    -> {mpc,trajopt}/harness-native, {mpc,trajopt}-harness-native.log

# 3. Extra row: defaults + staged inputs (native C++ replay), checked like 1.
build/native-nocudss/bench/cupiqp_native_replay --data native/bench/data/mpc --out \
  native/bench/results/remove-cudss-20261104/mpc/staged --repeats 20 --staged
.venv/bin/python -I native/bench/tools/check_solutions.py \
  --results native/bench/results/remove-cudss-20261104/mpc/staged \
  --bundle /mnt/c/Users/91354/Projects/gpu-qp-benchmark/qp_benchmark_v2_20261102
#    (same for trajopt with --bundle .../datasets/trajopt-20261102)
```

Baseline for comparison: `native/bench/results/reduced-auto-wholegraph-20261103`
(row `reduced-auto`: reduced KKT backend, Auto = whole-solve graph, cuSPARSE
SpMV, commit e55e2c8 + the Auto change).
