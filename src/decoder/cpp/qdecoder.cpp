// Fast-tier syndrome decoder: normalized min-sum BP (fixed-point int16, SIMD across shots)
// with early termination, followed by bit-packed OSD-CS post-processing for BP failures.
//
// C++20, compiled with clang/zig (ext_vector_type + __builtin_elementwise_* saturating ops).
// C ABI only; see decoder/fast.py for the cffi binding.
//
// Layout: every V16 vector holds the same Tanner-graph quantity for QD_LANES independent shots,
// so one pass over the graph decodes QD_LANES syndromes (AVX2: 16 x int16; AVX-512: 32 x int16).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#ifdef _WIN32
#define QD_API extern "C" __declspec(dllexport)
#else
#define QD_API extern "C" __attribute__((visibility("default")))
#endif

#ifndef QD_LANES
#define QD_LANES 16
#endif

typedef int16_t V __attribute__((ext_vector_type(QD_LANES)));
static constexpr int L = QD_LANES;
static constexpr int16_t LLR_MAX = 8000;  // keep |x| away from int16 overflow in sums

static inline V vsat_add(V a, V b) { return __builtin_elementwise_add_sat(a, b); }
static inline V vsat_sub(V a, V b) { return __builtin_elementwise_sub_sat(a, b); }
static inline V vmin(V a, V b) { return __builtin_elementwise_min(a, b); }
static inline V vsel(V mask, V a, V b) { return (mask & a) | (~mask & b); }
static inline V vsplat(int16_t x) { V v; for (int i = 0; i < L; i++) v[i] = x; return v; }

static inline uint64_t now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Dec {
    int m = 0, n = 0, E = 0, W = 0;  // W = words per column bitset (over m checks)
    std::vector<int> chk_ptr, chk_var;   // CSR by check; edge e = position in chk_var
    std::vector<int> var_ptr, var_edge;  // CSR by variable -> edge ids
    std::vector<int16_t> ch;             // channel LLR (fixed point)
    std::vector<float> w;                // real-valued LLR weight per column (OSD cost)
    std::vector<uint64_t> cols;          // n * W bitsets (column j over checks)
    int max_iter = 20, osd_lambda = 32, osd_pairs = 8;
    // scratch
    std::vector<V> r, tot;
};

// ---------------------------------------------------------------- creation
QD_API void* qd_create(int m, int n, const int* row_ptr, const int* col_idx, const double* prior,
                       int max_iter, double llr_scale, int osd_lambda) {
    Dec* d = new Dec();
    d->m = m; d->n = n; d->E = row_ptr[m];
    d->chk_ptr.assign(row_ptr, row_ptr + m + 1);
    d->chk_var.assign(col_idx, col_idx + d->E);
    std::vector<int> deg(n, 0);
    for (int e = 0; e < d->E; e++) deg[col_idx[e]]++;
    d->var_ptr.assign(n + 1, 0);
    for (int j = 0; j < n; j++) d->var_ptr[j + 1] = d->var_ptr[j] + deg[j];
    d->var_edge.resize(d->E);
    std::vector<int> fill(d->var_ptr.begin(), d->var_ptr.end() - 1);
    for (int c = 0; c < m; c++)
        for (int e = row_ptr[c]; e < row_ptr[c + 1]; e++) d->var_edge[fill[col_idx[e]]++] = e;
    d->ch.resize(n); d->w.resize(n);
    for (int j = 0; j < n; j++) {
        double p = std::min(std::max(prior[j], 1e-12), 1.0 - 1e-12);
        double l = std::log((1.0 - p) / p);
        d->w[j] = (float)l;
        d->ch[j] = (int16_t)std::max(1.0, std::min((double)LLR_MAX / 4, std::round(l * llr_scale)));
    }
    d->W = (m + 63) / 64;
    d->cols.assign((size_t)n * d->W, 0);
    for (int c = 0; c < m; c++)
        for (int e = row_ptr[c]; e < row_ptr[c + 1]; e++)
            d->cols[(size_t)col_idx[e] * d->W + (c >> 6)] |= 1ULL << (c & 63);
    d->max_iter = max_iter;
    d->osd_lambda = osd_lambda;
    d->r.resize(d->E); d->tot.resize(n);
    return d;
}

QD_API void qd_destroy(void* h) { delete (Dec*)h; }
QD_API int qd_lanes() { return L; }

// ---------------------------------------------------------------- OSD-CS (scalar, bit-packed)
// Information set from the least reliable (lowest posterior LLR) columns, order-0 solution plus
// a sweep over single flips of the first `lambda` dependent columns and pairs among the first
// `pairs` of them; returns the minimum-weight (sum of channel LLRs) consistent error.
static bool osd(Dec& d, const uint8_t* synd, const int16_t* post, uint8_t* out) {
    const int m = d.m, n = d.n, W = d.W;
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return post[a] < post[b]; });

    struct Row { std::vector<uint64_t> vec, comb; int pivot; int col; };
    std::vector<Row> basis;
    basis.reserve(m);
    const int CW = (m + 63) / 64;  // comb bitset over basis indices (rank <= m)
    struct Dep { int col; std::vector<uint64_t> comb; };
    std::vector<Dep> deps;

    auto reduce = [&](std::vector<uint64_t>& vec, std::vector<uint64_t>& comb) {
        for (const Row& b : basis)
            if ((vec[b.pivot >> 6] >> (b.pivot & 63)) & 1) {
                for (int k = 0; k < W; k++) vec[k] ^= b.vec[k];
                for (int k = 0; k < CW; k++) comb[k] ^= b.comb[k];
            }
    };
    int rank = 0;
    for (int idx = 0; idx < n && (rank < m || (int)deps.size() < d.osd_lambda); idx++) {
        int j = order[idx];
        std::vector<uint64_t> vec(d.cols.begin() + (size_t)j * W, d.cols.begin() + (size_t)(j + 1) * W);
        std::vector<uint64_t> comb(CW, 0);
        reduce(vec, comb);
        int piv = -1;
        for (int k = 0; k < W; k++)
            if (vec[k]) { piv = k * 64 + __builtin_ctzll(vec[k]); break; }
        if (piv >= 0) {
            comb[rank >> 6] ^= 1ULL << (rank & 63);
            basis.push_back({std::move(vec), std::move(comb), piv, j});
            rank++;
        } else if ((int)deps.size() < d.osd_lambda) {
            deps.push_back({j, std::move(comb)});
        }
    }
    // reduce the syndrome
    std::vector<uint64_t> sv(W, 0), sc(CW, 0);
    for (int c = 0; c < m; c++) if (synd[c]) sv[c >> 6] |= 1ULL << (c & 63);
    reduce(sv, sc);
    for (int k = 0; k < W; k++) if (sv[k]) { std::memset(out, 0, n); return false; }  // inconsistent syndrome: no stale BP bits
    auto cost_of = [&](const std::vector<uint64_t>& comb, int extra1, int extra2) {
        float s = 0;
        for (int t = 0; t < rank; t++) if ((comb[t >> 6] >> (t & 63)) & 1) s += d.w[basis[t].col];
        if (extra1 >= 0) s += d.w[extra1];
        if (extra2 >= 0) s += d.w[extra2];
        return s;
    };
    // order 0
    std::vector<uint64_t> best = sc;
    int b1 = -1, b2 = -1;
    float bc = cost_of(sc, -1, -1);
    std::vector<uint64_t> cand(CW);
    for (size_t a = 0; a < deps.size(); a++) {
        for (int k = 0; k < CW; k++) cand[k] = sc[k] ^ deps[a].comb[k];
        float c1 = cost_of(cand, deps[a].col, -1);
        if (c1 < bc) { bc = c1; best = cand; b1 = deps[a].col; b2 = -1; }
        for (size_t b = a + 1; b < deps.size() && (int)b < d.osd_pairs; b++) {
            if ((int)a >= d.osd_pairs) break;
            for (int k = 0; k < CW; k++) cand[k] = sc[k] ^ deps[a].comb[k] ^ deps[b].comb[k];
            float c2 = cost_of(cand, deps[a].col, deps[b].col);
            if (c2 < bc) { bc = c2; best = cand; b1 = deps[a].col; b2 = deps[b].col; }
        }
    }
    std::memset(out, 0, n);
    for (int t = 0; t < rank; t++) if ((best[t >> 6] >> (t & 63)) & 1) out[basis[t].col] ^= 1;
    if (b1 >= 0) out[b1] ^= 1;
    if (b2 >= 0) out[b2] ^= 1;
    return true;
}

// ---------------------------------------------------------------- BP over one chunk of <= L shots
// status[l]: 0 = BP converged, 1 = needs OSD (then OSD ok => 2, OSD failed => 3), iters[l] = BP iterations
static void decode_chunk(Dec& d, int nshots, const uint8_t* synd /*nshots*m*/, uint8_t* out /*nshots*n*/,
                         int* iters, uint8_t* status, bool use_osd) {
    const int m = d.m, n = d.n;
    std::vector<V> s(m);
    for (int c = 0; c < m; c++) {
        V v = vsplat(0);
        for (int l = 0; l < nshots; l++) v[l] = synd[(size_t)l * m + c] ? -1 : 0;  // mask form
        s[c] = v;
    }
    V* r = d.r.data();
    V* tot = d.tot.data();
    for (int e = 0; e < d.E; e++) r[e] = vsplat(0);
    for (int j = 0; j < n; j++) tot[j] = vsplat(d.ch[j]);

    static_assert(QD_LANES <= 64, "lane mask is 64-bit");
    uint64_t active = (nshots >= 64) ? ~0ULL : ((1ULL << nshots) - 1ULL);
    for (int l = 0; l < nshots; l++) { iters[l] = 0; status[l] = 1; }
    std::memset(out, 0, (size_t)nshots * n);

    auto syndrome_check = [&]() -> V {  // lane mask: -1 where some check unsatisfied
        V bad = vsplat(0);
        for (int c = 0; c < m; c++) {
            V acc = s[c];
            for (int e = d.chk_ptr[c]; e < d.chk_ptr[c + 1]; e++) acc ^= (V)(tot[d.chk_var[e]] >> 15);
            bad |= acc;
        }
        return bad;
    };
    auto finish_lanes = [&](V bad, int it) {
        for (int l = 0; l < nshots; l++) {
            if (!((active >> l) & 1)) continue;
            if (bad[l] == 0) {
                active &= ~(1ULL << l);
                iters[l] = it;
                status[l] = 0;
                for (int j = 0; j < n; j++) out[(size_t)l * n + j] = tot[j][l] < 0;
            }
        }
    };
    finish_lanes(syndrome_check(), 0);

    for (int it = 1; it <= d.max_iter && active; it++) {
        // check-node update (normalized min-sum, alpha = 0.75) using q = tot - r_old
        for (int c = 0; c < m; c++) {
            const int e0 = d.chk_ptr[c], e1 = d.chk_ptr[c + 1];
            V m1 = vsplat(INT16_MAX), m2 = vsplat(INT16_MAX), idx = vsplat(-1), sgn = s[c];
            for (int e = e0; e < e1; e++) {
                V q = vsat_sub(tot[d.chk_var[e]], r[e]);
                V sq = (V)(q >> 15);
                V a = __builtin_elementwise_abs(q);
                sgn ^= sq;
                V lt = (V)(a < m1);
                m2 = vsel(lt, m1, vmin(m2, a));
                idx = vsel(lt, vsplat((int16_t)(e - e0)), idx);
                m1 = vsel(lt, a, m1);
            }
            V n1 = m1 - (m1 >> 2), n2 = m2 - (m2 >> 2);
            for (int e = e0; e < e1; e++) {
                V q = vsat_sub(tot[d.chk_var[e]], r[e]);
                V sq = (V)(q >> 15);
                V mag = vsel((V)(idx == vsplat((int16_t)(e - e0))), n2, n1);
                V sign = sgn ^ sq;                        // -1 => negative message
                r[e] = (mag ^ sign) - sign;
            }
        }
        // variable-node update
        for (int j = 0; j < n; j++) {
            V t = vsplat(d.ch[j]);
            for (int k = d.var_ptr[j]; k < d.var_ptr[j + 1]; k++) t = vsat_add(t, r[d.var_edge[k]]);
            tot[j] = __builtin_elementwise_max(vmin(t, vsplat(LLR_MAX)), vsplat(-LLR_MAX));
        }
        finish_lanes(syndrome_check(), it);
        if (!active) break;
    }
    // leftover lanes: take the final hard decision as BP output, optionally run OSD
    std::vector<int16_t> post(n);
    for (int l = 0; l < nshots; l++) {
        if (!((active >> l) & 1)) continue;
        iters[l] = d.max_iter;
        for (int j = 0; j < n; j++) { post[j] = tot[j][l]; out[(size_t)l * n + j] = post[j] < 0; }
        if (use_osd) status[l] = osd(d, synd + (size_t)l * m, post.data(), out + (size_t)l * n) ? 2 : 3;
        else status[l] = 1;
    }
}

// ---------------------------------------------------------------- public decode entry points
// Decode `shots` syndromes (row-major uint8, shots x m). Writes errors (shots x n), per-shot BP
// iterations, status (0 BP ok, 1 BP failed (no OSD), 2 OSD ok, 3 OSD failed) and amortized latency
// in ns (chunk wall-time / shots in chunk).
QD_API void qd_decode_batch(void* h, int shots, const uint8_t* synd, uint8_t* out, int* iters,
                            uint8_t* status, uint64_t* latency_ns, int use_osd) {
    Dec& d = *(Dec*)h;
    for (int s0 = 0; s0 < shots; s0 += L) {
        int cnt = std::min(L, shots - s0);
        uint64_t t0 = now_ns();
        decode_chunk(d, cnt, synd + (size_t)s0 * d.m, out + (size_t)s0 * d.n, iters + s0, status + s0, use_osd);
        uint64_t dt = now_ns() - t0;
        for (int l = 0; l < cnt; l++) latency_ns[s0 + l] = dt / cnt;
    }
}

// Streaming-style single-shot decoding: each syndrome is decoded alone (lane 0), timed per shot.
// `bp_ns` receives BP-only time, `total_ns` BP + OSD time.
QD_API void qd_decode_stream(void* h, int shots, const uint8_t* synd, uint8_t* out, int* iters,
                             uint8_t* status, uint64_t* bp_ns, uint64_t* total_ns, int use_osd) {
    Dec& d = *(Dec*)h;
    for (int s = 0; s < shots; s++) {
        uint64_t t0 = now_ns();
        decode_chunk(d, 1, synd + (size_t)s * d.m, out + (size_t)s * d.n, iters + s, status + s, 0);
        uint64_t t1 = now_ns();
        if (status[s] == 1 && use_osd) {
            std::vector<int16_t> post(d.n);
            for (int j = 0; j < d.n; j++) post[j] = d.tot[j][0];
            status[s] = osd(d, synd + (size_t)s * d.m, post.data(), out + (size_t)s * d.n) ? 2 : 3;
        }
        uint64_t t2 = now_ns();
        bp_ns[s] = t1 - t0;
        total_ns[s] = t2 - t0;
    }
}

// OSD-CS driven by externally supplied posterior LLRs (e.g. from the GNN tier); int16 fixed point
// with the same scale as the channel LLRs. Returns 1 on success, 0 if the syndrome is inconsistent.
QD_API int qd_osd_external(void* h, const uint8_t* synd, const int16_t* post, uint8_t* out) {
    return osd(*(Dec*)h, synd, post, out) ? 1 : 0;
}

QD_API int qd_channel_llr(void* h, int16_t* out) {
    Dec& d = *(Dec*)h;
    std::memcpy(out, d.ch.data(), sizeof(int16_t) * d.n);
    return d.n;
}
