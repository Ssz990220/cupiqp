// Native MPC / Trajopt snapshot replay benchmark (public C++ API only).
//
// Mirrors gpu-qp-benchmark/benchmark/replay_mpc_updates.py: one persistent
// solver per CSR pattern group, explicit zeros retained, every matrix and
// vector value replaced through update() in the original stored order, FP64,
// eps_abs = eps_duality_gap_abs = 1e-8, relative tolerances 0, 10 Ruiz passes
// recomputed after matrix updates. Per repeat and trajectory: the first
// snapshot is solved as an excluded seed, then each chronological transition
// is timed. Two modes, alternating order: gpu_resident (values already on the
// GPU) and host_values (pinned host packet copied into a fixed GPU stage).
// Two full warm-up passes are excluded. Output: trials.jsonl (one line per
// timed transition), seeds.jsonl, setup.json and solutions.bin (x, y,
// z = z_u - z_l of every timed transition) for the independent KKT checker.
//
// Usage: cupiqp_native_replay --data <export dir> --out <results dir>
//          [--repeats 20] [--graph-mode iteration|none|device] [--custom-spmv]
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cupiqp/cupiqp.hpp"

namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;

#define CK(x)                                                                                   \
  do {                                                                                          \
    cudaError_t e_ = (x);                                                                       \
    if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_)); \
  } while (0)

namespace {

// ------------------------------------------------------------------ data
struct Group {
  std::string file;
  int32_t n = 0, p = 0, m = 0, nnzP = 0, nnzA = 0, nnzC = 0, cases = 0;
  cupiqp::CsrPattern P, A, C;
  std::vector<double> values;  // cases x total, order P, A, C, q, b, l, u
  size_t total = 0;
  std::vector<std::string> ids;
  std::vector<std::string> traj;
  std::vector<int> step;
  size_t off(const char* k) const {
    size_t o = 0;
    const char* names[] = {"P", "A", "C", "q", "b", "l", "u"};
    const size_t sizes[] = {size_t(nnzP), size_t(nnzA), size_t(nnzC), size_t(n), size_t(p), size_t(m), size_t(m)};
    for (int i = 0; i < 7; ++i) {
      if (std::strcmp(names[i], k) == 0) return o;
      o += sizes[i];
    }
    throw std::runtime_error("bad key");
  }
};

void read_pattern(std::ifstream& f, cupiqp::CsrPattern& M, int rows, int cols, int nnz) {
  M.rows = rows;
  M.cols = cols;
  M.indptr.resize(rows + 1);
  M.indices.resize(nnz);
  f.read(reinterpret_cast<char*>(M.indptr.data()), sizeof(int32_t) * (rows + 1));
  f.read(reinterpret_cast<char*>(M.indices.data()), sizeof(int32_t) * nnz);
}

Group load_group(const fs::path& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path.string());
  char magic[8];
  f.read(magic, 8);
  if (std::memcmp(magic, "CPQPSNP1", 8) != 0) throw std::runtime_error("bad magic in " + path.string());
  Group g;
  g.file = path.filename().string();
  int32_t hdr[7];
  f.read(reinterpret_cast<char*>(hdr), sizeof hdr);
  g.n = hdr[0]; g.p = hdr[1]; g.m = hdr[2]; g.nnzP = hdr[3]; g.nnzA = hdr[4]; g.nnzC = hdr[5]; g.cases = hdr[6];
  read_pattern(f, g.P, g.n, g.n, g.nnzP);
  read_pattern(f, g.A, g.p, g.n, g.nnzA);
  read_pattern(f, g.C, g.m, g.n, g.nnzC);
  g.total = size_t(g.nnzP) + g.nnzA + g.nnzC + g.n + g.p + 2 * size_t(g.m);
  g.values.resize(g.total * g.cases);
  f.read(reinterpret_cast<char*>(g.values.data()), sizeof(double) * g.values.size());
  if (!f) throw std::runtime_error("short read in " + path.string());
  return g;
}

// Minimal extraction from the exporter's manifest.json (known, flat format).
std::vector<std::string> json_strings_after(const std::string& s, size_t from, const std::string& key, size_t until) {
  std::vector<std::string> out;
  std::string pat = "\"" + key + "\": ";
  size_t pos = from;
  while ((pos = s.find(pat, pos)) != std::string::npos && pos < until) {
    pos += pat.size();
    if (s[pos] == '"') {
      size_t e = s.find('"', pos + 1);
      out.push_back(s.substr(pos + 1, e - pos - 1));
    } else {
      size_t e = s.find_first_of(",}\n", pos);
      out.push_back(s.substr(pos, e - pos));
    }
  }
  return out;
}

void attach_manifest(const fs::path& dir, std::vector<Group>& groups) {
  std::ifstream f(dir / "manifest.json");
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string s = ss.str();
  for (auto& g : groups) {
    size_t at = s.find("\"file\": \"" + g.file + "\"");
    if (at == std::string::npos) throw std::runtime_error("group not in manifest: " + g.file);
    size_t meta = s.find("\"case_meta\"", at);
    size_t end = s.find("\"file\": \"", at + 10);
    if (end == std::string::npos) end = s.size();
    g.ids = json_strings_after(s, meta, "id", end);
    auto tn = json_strings_after(s, meta, "trajectory_name", end);
    auto tt = json_strings_after(s, meta, "trajectory", end);
    auto cs = json_strings_after(s, meta, "control_step", end);
    auto si = json_strings_after(s, meta, "sqp_iteration", end);
    g.traj.resize(g.ids.size());
    g.step.resize(g.ids.size());
    for (size_t i = 0; i < g.ids.size(); ++i) {
      g.traj[i] = (i < tn.size() && tn[i] != "null") ? tn[i] : (i < tt.size() ? tt[i] : g.file);
      std::string st = (i < cs.size() && cs[i] != "null") ? cs[i] : (i < si.size() ? si[i] : "0");
      g.step[i] = st == "null" ? 0 : std::stoi(st);
    }
    if (static_cast<int>(g.ids.size()) != g.cases) throw std::runtime_error("manifest case count mismatch");
  }
}

// ------------------------------------------------------------------ stats
struct Trial {
  std::string group, traj, mode, from_case, to_case;
  int repeat = 0, from_step = 0, to_step = 0, status = -9, iter = 0;
  double wall_ms = 0;
  float upload_ms = 0, update_ms = 0, solve_ms = 0, download_ms = 0, gpu_x_ready_ms = 0;
  double primal_obj = 0, primal_res = 0, dual_res = 0, duality_gap = 0;
};

std::string trial_json(const Trial& t) {
  char buf[1024];
  std::snprintf(buf, sizeof buf,
                "{\"group\":\"%s\",\"trajectory\":\"%s\",\"mode\":\"%s\",\"repeat\":%d,\"from_case\":\"%s\","
                "\"to_case\":\"%s\",\"from_step\":%d,\"to_step\":%d,\"step_gap\":%d,\"status\":\"%s\",\"iterations\":%d,"
                "\"cpu_x_ready_wall_ms\":%.6f,\"upload_event_ms\":%.6f,\"update_event_ms\":%.6f,"
                "\"solve_event_ms\":%.6f,\"x_download_event_ms\":%.6f,\"gpu_x_ready_event_ms\":%.6f,"
                "\"primal_obj\":%.17g,\"primal_res\":%.6e,\"dual_res\":%.6e,\"duality_gap\":%.6e}",
                t.group.c_str(), t.traj.c_str(), t.mode.c_str(), t.repeat, t.from_case.c_str(), t.to_case.c_str(),
                t.from_step, t.to_step, t.to_step - t.from_step,
                cupiqp::status_name(static_cast<cupiqp::Status>(t.status)), t.iter, t.wall_ms, t.upload_ms,
                t.update_ms, t.solve_ms, t.download_ms, t.gpu_x_ready_ms, t.primal_obj, t.primal_res, t.dual_res,
                t.duality_gap);
  return buf;
}

// ------------------------------------------------------------------ replay
struct Replay {
  const Group& g;
  cupiqp::SparseSolver solver;
  double* d_packets = nullptr;   // all cases, device
  double* h_packets = nullptr;   // all cases, pinned host
  double* d_stage = nullptr;     // fixed GPU stage for host_values
  double* h_x = nullptr;         // pinned x
  double *d_y = nullptr, *d_z = nullptr;
  cudaEvent_t ev[5];
  std::vector<double> sol_y, sol_zu, sol_zl;

  Replay(const Group& grp, cupiqp::GraphMode mode, bool custom_spmv) : g(grp) {
    auto& st = solver.settings();
    st.eps_abs = 1e-8;
    st.eps_duality_gap_abs = 1e-8;
    st.eps_rel = 0.0;
    st.eps_duality_gap_rel = 0.0;
    st.preconditioner_iter = 10;
    st.preconditioner_scale_cost = false;
    st.preconditioner_reuse_on_update = false;
    st.iterative_refinement_always_enabled = false;
    st.enable_cuda_graph = mode != cupiqp::GraphMode::None;
    st.graph_mode = mode;
    st.verbose = false;
    st.custom_spmv = custom_spmv;
    CK(cudaMalloc(&d_packets, sizeof(double) * g.values.size()));
    CK(cudaMemcpy(d_packets, g.values.data(), sizeof(double) * g.values.size(), cudaMemcpyHostToDevice));
    CK(cudaMallocHost(&h_packets, sizeof(double) * g.values.size()));
    std::memcpy(h_packets, g.values.data(), sizeof(double) * g.values.size());
    CK(cudaMalloc(&d_stage, sizeof(double) * g.total));
    CK(cudaMallocHost(&h_x, sizeof(double) * g.n));
    for (auto& e : ev) CK(cudaEventCreate(&e));
    sol_y.resize(g.p);
    sol_zu.resize(g.m);
    sol_zl.resize(g.m);
  }
  ~Replay() {
    cudaStreamSynchronize(solver.stream());
    cudaFree(d_packets); cudaFreeHost(h_packets); cudaFree(d_stage); cudaFreeHost(h_x);
    for (auto& e : ev) cudaEventDestroy(e);
  }

  cupiqp::ValuesView view(const double* base) const {
    cupiqp::ValuesView v;
    v.P = base + g.off("P"); v.A = base + g.off("A"); v.G = base + g.off("C");
    v.c = base + g.off("q"); v.b = base + g.off("b"); v.h_l = base + g.off("l"); v.h_u = base + g.off("u");
    return v;
  }

  double setup_ms = 0;
  void setup(int case_index) {
    auto t0 = clk::now();
    solver.setup(g.P, view(d_packets + g.total * case_index), &g.A, &g.C);
    CK(cudaStreamSynchronize(solver.stream()));
    setup_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  }

  // One update + solve + x download; returns the trial (timings when measure).
  Trial step(int case_index, bool host_values, bool measure) {
    cudaStream_t s = solver.stream();
    Trial t;
    if (measure) CK(cudaEventRecord(ev[0], s));
    auto t0 = clk::now();
    const double* src = d_packets + g.total * case_index;
    if (host_values) {
      CK(cudaMemcpyAsync(d_stage, h_packets + g.total * case_index, sizeof(double) * g.total,
                         cudaMemcpyHostToDevice, s));
      src = d_stage;
    }
    if (measure) CK(cudaEventRecord(ev[1], s));
    solver.update(view(src));
    if (measure) CK(cudaEventRecord(ev[2], s));
    solver.solve();
    if (measure) CK(cudaEventRecord(ev[3], s));
    solver.download_x(h_x);
    if (measure) CK(cudaEventRecord(ev[4], s));
    CK(cudaStreamSynchronize(s));
    t.wall_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    if (measure) {
      CK(cudaEventElapsedTime(&t.upload_ms, ev[0], ev[1]));
      CK(cudaEventElapsedTime(&t.update_ms, ev[1], ev[2]));
      CK(cudaEventElapsedTime(&t.solve_ms, ev[2], ev[3]));
      CK(cudaEventElapsedTime(&t.download_ms, ev[3], ev[4]));
      CK(cudaEventElapsedTime(&t.gpu_x_ready_ms, ev[0], ev[4]));
    }
    // Outside the timed interval: status, iterations, residuals, duals.
    cupiqp::Info info = solver.info();
    t.status = static_cast<int>(info.status);
    t.iter = info.iter;
    t.primal_obj = info.primal_obj;
    t.primal_res = info.primal_res;
    t.dual_res = info.dual_res;
    t.duality_gap = info.duality_gap;
    auto r = solver.result();
    if (g.p) CK(cudaMemcpy(sol_y.data(), r.y, sizeof(double) * g.p, cudaMemcpyDeviceToHost));
    std::fill(sol_zu.begin(), sol_zu.end(), 0.0);
    std::fill(sol_zl.begin(), sol_zl.end(), 0.0);
    if (r.z_u) CK(cudaMemcpy(sol_zu.data(), r.z_u, sizeof(double) * g.m, cudaMemcpyDeviceToHost));
    if (r.z_l) CK(cudaMemcpy(sol_zl.data(), r.z_l, sizeof(double) * g.m, cudaMemcpyDeviceToHost));
    return t;
  }
};

}  // namespace

int main(int argc, char** argv) {
  try {
    fs::path data, out;
    int repeats = 20;
    std::string gm = "iteration";
    bool custom_spmv = false;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
        return argv[++i];
      };
      if (a == "--data") data = next();
      else if (a == "--out") out = next();
      else if (a == "--repeats") repeats = std::stoi(next());
      else if (a == "--graph-mode") gm = next();
      else if (a == "--custom-spmv") custom_spmv = true;
      else throw std::runtime_error("unknown argument " + a);
    }
    if (data.empty() || out.empty()) throw std::runtime_error("--data and --out are required");
    cupiqp::GraphMode mode = gm == "none" ? cupiqp::GraphMode::None
                             : gm == "device" ? cupiqp::GraphMode::SolveGraphDeviceLoop
                                              : cupiqp::GraphMode::IterationGraphHostLoop;
    if (fs::exists(out)) throw std::runtime_error("output exists: " + out.string());
    fs::create_directories(out);

    std::vector<Group> groups;
    for (auto& e : fs::directory_iterator(data))
      if (e.path().extension() == ".bin") groups.push_back(load_group(e.path()));
    std::sort(groups.begin(), groups.end(), [](const Group& a, const Group& b) { return a.file < b.file; });
    attach_manifest(data, groups);

    std::ofstream trials(out / "trials.jsonl"), seeds(out / "seeds.jsonl"), setup(out / "setup.json");
    std::ofstream sol(out / "solutions.bin", std::ios::binary);
    std::ofstream solidx(out / "solutions-index.jsonl");
    setup << "{\"version\":\"" << cupiqp::version_string() << "\",\"graph_mode\":\"" << gm
          << "\",\"custom_spmv\":" << (custom_spmv ? "true" : "false") << ",\"repeats\":" << repeats
          << ",\"groups\":[";
    int64_t sol_offset = 0, timed = 0;

    for (size_t gi = 0; gi < groups.size(); ++gi) {
      const Group& g = groups[gi];
      // Trajectories in manifest order, cases sorted by step.
      std::vector<std::string> order;
      std::map<std::string, std::vector<int>> by_traj;
      for (int i = 0; i < g.cases; ++i) {
        if (!by_traj.count(g.traj[i])) order.push_back(g.traj[i]);
        by_traj[g.traj[i]].push_back(i);
      }
      for (auto& [k, v] : by_traj)
        std::sort(v.begin(), v.end(), [&](int a, int b) { return g.step[a] < g.step[b]; });

      Replay rp(g, mode, custom_spmv);
      rp.setup(0);
      // First solve after setup: includes the first graph capture (reported separately).
      auto tf = clk::now();
      rp.solver.solve();
      CK(cudaStreamSynchronize(rp.solver.stream()));
      double first_solve_ms = std::chrono::duration<double, std::milli>(clk::now() - tf).count();
      cupiqp::Counters c_setup = rp.solver.counters();

      for (int w = 0; w < 2; ++w)  // two excluded warm-up passes, alternating modes
        for (auto& tr : order) {
          int i = 0;
          for (int ci : by_traj[tr]) rp.step(ci, (w + i++) % 2 == 1, false);
        }
      cupiqp::Counters c_warm = rp.solver.counters();

      for (int rep = 0; rep < repeats; ++rep) {
        for (size_t ti = 0; ti < order.size(); ++ti) {
          const auto& cs = by_traj[order[ti]];
          const bool gpu_first = (rep + ti) % 2 == 0;
          for (int mi = 0; mi < 2; ++mi) {
            const bool host_values = gpu_first ? mi == 1 : mi == 0;
            Trial seed = rp.step(cs[0], host_values, false);
            seeds << "{\"group\":\"" << g.file << "\",\"repeat\":" << rep << ",\"trajectory\":\"" << order[ti]
                  << "\",\"mode\":\"" << (host_values ? "host_values" : "gpu_resident") << "\",\"case\":\""
                  << g.ids[cs[0]] << "\",\"status\":\"" << cupiqp::status_name(static_cast<cupiqp::Status>(seed.status))
                  << "\",\"iterations\":" << seed.iter << "}\n";
            for (size_t k = 1; k < cs.size(); ++k) {
              Trial t = rp.step(cs[k], host_values, true);
              t.group = g.file; t.traj = order[ti]; t.repeat = rep;
              t.mode = host_values ? "host_values" : "gpu_resident";
              t.from_case = g.ids[cs[k - 1]]; t.to_case = g.ids[cs[k]];
              t.from_step = g.step[cs[k - 1]]; t.to_step = g.step[cs[k]];
              trials << trial_json(t) << "\n";
              // Solution for the independent KKT check: x, y, z = z_u - z_l.
              sol.write(reinterpret_cast<const char*>(rp.h_x), sizeof(double) * g.n);
              sol.write(reinterpret_cast<const char*>(rp.sol_y.data()), sizeof(double) * g.p);
              std::vector<double> z(g.m);
              for (int r = 0; r < g.m; ++r) z[r] = rp.sol_zu[r] - rp.sol_zl[r];
              sol.write(reinterpret_cast<const char*>(z.data()), sizeof(double) * g.m);
              solidx << "{\"trial\":" << timed << ",\"case\":\"" << g.ids[cs[k]] << "\",\"offset\":" << sol_offset
                     << ",\"n\":" << g.n << ",\"p\":" << g.p << ",\"m\":" << g.m << "}\n";
              sol_offset += sizeof(double) * (g.n + g.p + g.m);
              ++timed;
            }
          }
        }
        std::fprintf(stderr, "group %s: repeat %d/%d done (timed so far %lld)\n", g.file.c_str(), rep + 1,
                     repeats, static_cast<long long>(timed));
      }
      cupiqp::Counters c_end = rp.solver.counters();
      auto delta = [](const cupiqp::Counters& a, const cupiqp::Counters& b) {
        std::ostringstream o;
        o << "{\"host_syncs\":" << b.host_syncs - a.host_syncs << ",\"graph_launches\":"
          << b.graph_launches - a.graph_launches << ",\"graph_captures\":" << b.graph_captures - a.graph_captures
          << ",\"cudss_device_allocs\":" << b.cudss_device_allocs - a.cudss_device_allocs
          << ",\"cudss_device_frees\":" << b.cudss_device_frees - a.cudss_device_frees
          << ",\"cudss_analysis_calls\":" << b.cudss_analysis_calls - a.cudss_analysis_calls
          << ",\"updates\":" << b.updates - a.updates << ",\"solves\":" << b.solves - a.solves << "}";
        return o.str();
      };
      setup << (gi ? "," : "") << "{\"file\":\"" << g.file << "\",\"n\":" << g.n << ",\"p\":" << g.p << ",\"m\":"
            << g.m << ",\"nnz\":[" << g.nnzP << "," << g.nnzA << "," << g.nnzC << "],\"cases\":" << g.cases
            << ",\"setup_ms\":" << rp.setup_ms << ",\"first_solve_incl_capture_ms\":" << first_solve_ms
            << ",\"counters_setup_and_first_solve\":" << delta(cupiqp::Counters{}, c_setup)
            << ",\"counters_steady_state\":" << delta(c_warm, c_end) << "}";
    }
    setup << "],\"timed_transitions\":" << timed << "}\n";
    std::fprintf(stderr, "done: %lld timed transitions -> %s\n", static_cast<long long>(timed), out.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
