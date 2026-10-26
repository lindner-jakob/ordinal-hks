#ifndef HKS_SIM_COMMON_HPP
#define HKS_SIM_COMMON_HPP

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct Fenwick {
    int n = 0;
    std::vector<long long> bit;

    Fenwick() = default;
    explicit Fenwick(int n_) : n(n_), bit(n_ + 1, 0) {}
    explicit Fenwick(const std::vector<long long>& a) : n((int)a.size()), bit(n + 1, 0) {
        for (int i = 0; i < n; ++i) add(i, a[i]);
    }

    void add(int idx, long long delta) {
        for (int j = idx + 1; j <= n; j += j & -j) bit[j] += delta;
    }

    long long prefix(int end) const { // sum on [0,end)
        long long s = 0;
        for (int j = end; j > 0; j -= j & -j) s += bit[j];
        return s;
    }

    long long range_sum(int l, int r) const {
        return prefix(r + 1) - prefix(l);
    }

    int kth(long long k) const { // 1-indexed rank -> 0-indexed position
        int idx = 0;
        int step = 1;
        while ((step << 1) <= n) step <<= 1;
        for (; step; step >>= 1) {
            int nxt = idx + step;
            if (nxt <= n && bit[nxt] < k) {
                idx = nxt;
                k -= bit[nxt];
            }
        }
        return idx;
    }
};

enum RuleCode { LOWER_MEDIAN = 0, K_LARGE = 1 };

struct Instance {
    int n = 0;
    std::vector<double> pos;
    std::vector<int> L, R;
};

inline uint64_t split_seed(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Deterministic C++ profile generator.  rho means x_1(i) ~ U[0,n/rho].
inline Instance make_instance_cpp(int n, double rho, uint64_t seed) {
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> U(0.0, n / rho);

    Instance ins;
    ins.n = n;
    ins.pos.resize(n);
    ins.L.resize(n);
    ins.R.resize(n);

    for (double& x : ins.pos) x = U(gen);
    std::sort(ins.pos.begin(), ins.pos.end());

    for (int i = 0; i < n; ++i) {
        ins.L[i] = int(std::lower_bound(ins.pos.begin(), ins.pos.end(), ins.pos[i] - 1.0) - ins.pos.begin());
        ins.R[i] = int(std::upper_bound(ins.pos.begin(), ins.pos.end(), ins.pos[i] + 1.0) - ins.pos.begin()) - 1;
    }
    return ins;
}

// SplitMix64 activation generator, matching the exact Python/Numba drivers for Figs. 3--4.
struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t s) : state(s) {}
    uint64_t next() {
        state += 0x9e3779b97f4a7c15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    long long uniform_below(long long total) {
        const uint64_t M = (1ULL << 62);
        const uint64_t limit = M - (M % uint64_t(total));
        while (true) {
            uint64_t r = next() & (M - 1);
            if (r < limit) return (long long)(r % uint64_t(total));
        }
    }
};

// Exact token-process simulation.  uniform random choice is among unstable agents,
// hence occupied vertices are sampled with weight equal to their unstable multiplicity.
inline long long simulate_splitmix(const Instance& ins, RuleCode rule, int k, uint64_t activation_seed) {
    int n = ins.n;
    std::vector<long long> counts(n, 1);
    Fenwick mass(counts);

    auto target = [&](int v) -> int {
        long long m = mass.range_sum(ins.L[v], ins.R[v]);
        long long rank_asc;
        if (rule == LOWER_MEDIAN) {
            rank_asc = (m + 1) / 2; // numerical lower median
        } else {
            rank_asc = std::max(1LL, m - (long long)k + 1);
        }
        long long absolute_rank = mass.prefix(ins.L[v]) + rank_asc;
        return mass.kth(absolute_rank);
    };

    std::vector<long long> unstable_weight(n, 0);
    for (int v = 0; v < n; ++v) {
        if (target(v) != v) unstable_weight[v] = 1;
    }
    Fenwick unstable(unstable_weight);
    SplitMix64 rng(activation_seed);

    long long steps = 0;
    while (true) {
        long long total = unstable.prefix(n);
        if (total == 0) return steps;

        int source = unstable.kth(rng.uniform_below(total) + 1);
        int dest = target(source);

        counts[source]--;
        mass.add(source, -1);
        counts[dest]++;
        mass.add(dest, +1);

        int lo = std::min(ins.L[source], ins.L[dest]);
        int hi = std::max(ins.R[source], ins.R[dest]);
        for (int v = lo; v <= hi; ++v) {
            long long oldw = unstable_weight[v];
            long long neww = 0;
            if (counts[v] > 0 && target(v) != v) neww = counts[v];
            if (neww != oldw) {
                unstable_weight[v] = neww;
                unstable.add(v, neww - oldw);
            }
        }
        ++steps;
    }
}

// Figure 1 used std::mt19937_64 for activation; retain that exact variant.
inline long long simulate_mt19937(const Instance& ins, RuleCode rule, int k, uint64_t activation_seed) {
    int n = ins.n;
    std::vector<long long> counts(n, 1);
    Fenwick mass(counts);

    auto target = [&](int v) -> int {
        long long m = mass.range_sum(ins.L[v], ins.R[v]);
        long long rank_asc = (rule == LOWER_MEDIAN) ? (m + 1) / 2 : std::max(1LL, m - (long long)k + 1);
        return mass.kth(mass.prefix(ins.L[v]) + rank_asc);
    };

    std::vector<long long> unstable_weight(n, 0);
    for (int v = 0; v < n; ++v) if (target(v) != v) unstable_weight[v] = 1;
    Fenwick unstable(unstable_weight);
    std::mt19937_64 rng(activation_seed);

    long long steps = 0;
    while (true) {
        long long total = unstable.prefix(n);
        if (total == 0) return steps;
        std::uniform_int_distribution<long long> pick(1, total);
        int source = unstable.kth(pick(rng));
        int dest = target(source);

        counts[source]--; mass.add(source, -1);
        counts[dest]++;   mass.add(dest, +1);

        int lo = std::min(ins.L[source], ins.L[dest]);
        int hi = std::max(ins.R[source], ins.R[dest]);
        for (int v = lo; v <= hi; ++v) {
            long long oldw = unstable_weight[v];
            long long neww = 0;
            if (counts[v] > 0 && target(v) != v) neww = counts[v];
            if (neww != oldw) {
                unstable_weight[v] = neww;
                unstable.add(v, neww - oldw);
            }
        }
        ++steps;
    }
}

#endif
