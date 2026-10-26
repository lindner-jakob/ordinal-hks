#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using std::int64_t;
using std::string;
using std::uint64_t;
using std::vector;

namespace {

constexpr double EPSILON = 1.0;
constexpr int GENERATOR_VERSION = 1;

uint64_t splitmix_hash(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

uint64_t fnv1a(const string& s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) {
        h ^= uint64_t(c);
        h *= 1099511628211ULL;
    }
    return h;
}

struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t seed) : state(seed) {}

    uint64_t next() {
        state += 0x9e3779b97f4a7c15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    int64_t below(int64_t total) {
        const uint64_t M = (1ULL << 62);
        const uint64_t limit = M - M % uint64_t(total);
        while (true) {
            uint64_t r = next() & (M - 1);
            if (r < limit) return int64_t(r % uint64_t(total));
        }
    }
};

struct Fenwick {
    int n = 0;
    vector<int64_t> bit;

    Fenwick() = default;
    explicit Fenwick(const vector<int64_t>& values)
        : n(int(values.size())), bit(n + 1, 0) {
        for (int i = 0; i < n; ++i) add(i, values[i]);
    }

    void add(int index, int64_t delta) {
        for (int j = index + 1; j <= n; j += j & -j) bit[j] += delta;
    }

    int64_t prefix(int end) const {
        int64_t ans = 0;
        for (int j = end; j > 0; j -= j & -j) ans += bit[j];
        return ans;
    }

    int64_t range_sum(int left, int right) const {
        return prefix(right + 1) - prefix(left);
    }

    int kth(int64_t rank) const {
        int index = 0;
        int step = 1;
        while ((step << 1) <= n) step <<= 1;
        for (; step; step >>= 1) {
            int next = index + step;
            if (next <= n && bit[next] < rank) {
                index = next;
                rank -= bit[next];
            }
        }
        return index;
    }
};

// Empty support sites never become occupied again under ordinal updates.
// A successor structure therefore skips them without changing the dynamics.
struct OccupiedSuccessor {
    vector<int> parent;
    explicit OccupiedSuccessor(int n) : parent(n + 1) {
        std::iota(parent.begin(), parent.end(), 0);
    }
    int find(int v) {
        int root = v;
        while (parent[root] != root) root = parent[root];
        while (parent[v] != v) {
            int next = parent[v];
            parent[v] = root;
            v = next;
        }
        return root;
    }
    void erase(int v) { parent[v] = find(v + 1); }
};

enum class RuleCode { LOWER_MEDIAN, K_LARGE, K_SMALL };

struct Rule {
    string name;
    RuleCode code;
    int k;
    int id;
};

struct ProfileSpec {
    string distribution;
    string span_mode = "agent_density";
    string occupancy = "singleton";
    int n = 0;
    int ell = 0;
    double rho = 0.0;
    int cluster_count = 0;
    int id = 0;
};

struct Profile {
    int n = 0;
    vector<double> position;
    vector<int64_t> initial_count;
    vector<int> left;
    vector<int> right;
    vector<int> component;
    int component_count = 0;
    int largest_component_agents = 0;
    int largest_component_support = 0;
    double span = 0.0;
};

vector<int64_t> balanced_counts(int n, int ell, std::mt19937_64& rng) {
    vector<int64_t> count(ell, n / ell);
    vector<int> order(ell);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    for (int i = 0; i < n % ell; ++i) ++count[order[i]];
    return count;
}

vector<int64_t> multinomial_counts(int n, int ell, std::mt19937_64& rng) {
    vector<int64_t> count(ell, 1);
    std::uniform_int_distribution<int> pick(0, ell - 1);
    for (int i = ell; i < n; ++i) ++count[pick(rng)];
    return count;
}

void finalize_profile(Profile& p) {
    const int ell = int(p.position.size());
    if (ell == 0 || int(p.initial_count.size()) != ell)
        throw std::runtime_error("invalid empty or mismatched profile");
    p.left.resize(ell);
    p.right.resize(ell);
    for (int i = 0; i < ell; ++i) {
        p.left[i] = int(std::lower_bound(
            p.position.begin(), p.position.end(), p.position[i] - EPSILON)
            - p.position.begin());
        p.right[i] = int(std::upper_bound(
            p.position.begin(), p.position.end(), p.position[i] + EPSILON)
            - p.position.begin()) - 1;
    }

    p.component.resize(ell);
    int cid = 0;
    p.component[0] = 0;
    for (int i = 1; i < ell; ++i) {
        if (p.position[i] - p.position[i - 1] > EPSILON) ++cid;
        p.component[i] = cid;
    }
    p.component_count = cid + 1;

    vector<int> support_size(p.component_count, 0);
    vector<int64_t> agent_size(p.component_count, 0);
    for (int i = 0; i < ell; ++i) {
        ++support_size[p.component[i]];
        agent_size[p.component[i]] += p.initial_count[i];
    }
    p.largest_component_support = *std::max_element(support_size.begin(), support_size.end());
    p.largest_component_agents = int(*std::max_element(agent_size.begin(), agent_size.end()));
    p.span = p.position.back() - p.position.front();
}

Profile make_profile(const ProfileSpec& spec, uint64_t seed) {
    if (spec.n <= 0 || spec.ell <= 0 || spec.ell > spec.n || spec.rho <= 0)
        throw std::runtime_error("invalid profile specification");

    std::mt19937_64 rng(seed);
    Profile p;
    p.n = spec.n;
    p.position.resize(spec.ell);

    double domain_length = 0.0;
    if (spec.span_mode == "support_density")
        domain_length = double(spec.ell) / spec.rho;
    else
        domain_length = double(spec.n) / spec.rho;

    if (spec.distribution == "uniform" || spec.distribution == "uniform_discrete") {
        std::uniform_real_distribution<double> uniform(0.0, domain_length);
        for (double& x : p.position) x = uniform(rng);
    } else if (spec.distribution == "beta22") {
        std::gamma_distribution<double> gamma(2.0, 1.0);
        for (double& x : p.position) {
            const double a = gamma(rng);
            const double b = gamma(rng);
            x = domain_length * a / (a + b);
        }
    } else if (spec.distribution == "jittered_lattice") {
        std::uniform_real_distribution<double> jitter(-0.2, 0.2);
        for (int i = 0; i < spec.ell; ++i)
            p.position[i] = (double(i) + 0.5 + jitter(rng)) * domain_length / spec.ell;
    } else if (spec.distribution == "clustered_blocks") {
        const int clusters = std::max(1, spec.cluster_count);
        const double block_width = 4.0;
        const double block_gap = 2.0;
        std::uniform_real_distribution<double> within(0.0, block_width);
        for (int i = 0; i < spec.ell; ++i) {
            int c = int((int64_t(i) * clusters) / spec.ell);
            if (c >= clusters) c = clusters - 1;
            p.position[i] = c * (block_width + block_gap) + within(rng);
        }
    } else {
        throw std::runtime_error("unknown distribution: " + spec.distribution);
    }

    std::sort(p.position.begin(), p.position.end());
    if (spec.occupancy == "balanced")
        p.initial_count = balanced_counts(spec.n, spec.ell, rng);
    else if (spec.occupancy == "multinomial")
        p.initial_count = multinomial_counts(spec.n, spec.ell, rng);
    else {
        if (spec.ell != spec.n)
            throw std::runtime_error("singleton occupancy requires ell=n");
        p.initial_count.assign(spec.ell, 1);
    }

    finalize_profile(p);
    return p;
}

struct Outcome {
    int64_t steps = 0;
    bool truncated = false;
    int final_opinions = 0;
    int final_components = 0;
    int consensus_components = 0;
    bool global_consensus = false;
    bool component_consensus = false;
    double component_consensus_fraction = 0.0;
    double final_opinions_per_initial_component = 0.0;
    double largest_final_opinion_fraction = 0.0;
    double largest_final_component_fraction = 0.0;
    int64_t moves_left = 0;
    int64_t moves_right = 0;
    int64_t cut_crossings = 0;
    double metric_distance = 0.0;
    vector<int64_t> final_count;
};

int64_t ascending_rank(int64_t neighborhood_size, const Rule& rule) {
    if (rule.code == RuleCode::LOWER_MEDIAN)
        return (neighborhood_size + 1) / 2;
    if (rule.code == RuleCode::K_LARGE)
        return std::max<int64_t>(1, neighborhood_size - rule.k + 1);
    return std::min<int64_t>(rule.k, neighborhood_size);
}

Outcome summarize_outcome(const Profile& profile, const vector<int64_t>& count, Outcome out) {
    const int ell = int(count.size());
    vector<int> survivors(profile.component_count, 0);
    int64_t largest_opinion = 0;
    int64_t largest_final_component = 0;
    int64_t current_component_mass = 0;
    int previous_occupied = -1;

    for (int v = 0; v < ell; ++v) {
        if (count[v] == 0) continue;
        ++out.final_opinions;
        ++survivors[profile.component[v]];
        largest_opinion = std::max(largest_opinion, count[v]);
        if (previous_occupied < 0 || profile.position[v] - profile.position[previous_occupied] > EPSILON) {
            if (previous_occupied >= 0) largest_final_component = std::max(largest_final_component, current_component_mass);
            ++out.final_components;
            current_component_mass = count[v];
        } else {
            current_component_mass += count[v];
        }
        previous_occupied = v;
    }
    largest_final_component = std::max(largest_final_component, current_component_mass);

    for (int value : survivors) out.consensus_components += (value == 1);
    out.global_consensus = (out.final_opinions == 1);
    out.component_consensus = (out.consensus_components == profile.component_count);
    out.component_consensus_fraction = double(out.consensus_components) / profile.component_count;
    out.final_opinions_per_initial_component = double(out.final_opinions) / profile.component_count;
    out.largest_final_opinion_fraction = double(largest_opinion) / profile.n;
    out.largest_final_component_fraction = double(largest_final_component) / profile.n;
    out.final_count = count;
    return out;
}

Outcome simulate(const Profile& profile, const Rule& rule, uint64_t activation_seed,
                 int64_t maximum_steps = 20'000'000'000LL) {
    const int ell = int(profile.position.size());
    vector<int64_t> count = profile.initial_count;
    Fenwick mass(count);

    auto target = [&](int vertex) {
        int64_t m = mass.range_sum(profile.left[vertex], profile.right[vertex]);
        int64_t rank = ascending_rank(m, rule);
        int64_t absolute_rank = mass.prefix(profile.left[vertex]) + rank;
        return mass.kth(absolute_rank);
    };

    vector<int64_t> unstable_weight(ell, 0);
    for (int v = 0; v < ell; ++v)
        if (target(v) != v) unstable_weight[v] = count[v];
    Fenwick unstable(unstable_weight);
    OccupiedSuccessor occupied(ell);
    SplitMix64 rng(activation_seed);
    Outcome out;

    while (out.steps < maximum_steps) {
        int64_t total_unstable = unstable.prefix(ell);
        if (total_unstable == 0) return summarize_outcome(profile, count, out);

        int source = unstable.kth(rng.below(total_unstable) + 1);
        int destination = target(source);
        if (destination < source) ++out.moves_left;
        if (destination > source) ++out.moves_right;
        out.cut_crossings += std::abs(destination - source);
        out.metric_distance += std::abs(profile.position[destination] - profile.position[source]);

        --count[source];
        mass.add(source, -1);
        ++count[destination];
        mass.add(destination, 1);

        if (count[source] == 0) {
            unstable.add(source, -unstable_weight[source]);
            unstable_weight[source] = 0;
            occupied.erase(source);
        }

        int lo = std::min(profile.left[source], profile.left[destination]);
        int hi = std::max(profile.right[source], profile.right[destination]);
        for (int v = occupied.find(lo); v <= hi; v = occupied.find(v + 1)) {
            int64_t new_weight = 0;
            if (count[v] > 0 && target(v) != v) new_weight = count[v];
            if (new_weight != unstable_weight[v]) {
                unstable.add(v, new_weight - unstable_weight[v]);
                unstable_weight[v] = new_weight;
            }
        }
        ++out.steps;
    }

    out.truncated = true;
    return summarize_outcome(profile, count, out);
}

// Deliberately simple reference implementation for exact small-instance tests.
Outcome simulate_naive(const Profile& profile, const Rule& rule, uint64_t activation_seed,
                       int64_t maximum_steps = 10'000'000) {
    const int ell = int(profile.position.size());
    vector<int64_t> count = profile.initial_count;
    SplitMix64 rng(activation_seed);
    Outcome out;

    auto target = [&](int vertex) {
        int64_t m = 0;
        for (int u = profile.left[vertex]; u <= profile.right[vertex]; ++u) m += count[u];
        int64_t rank = ascending_rank(m, rule);
        int64_t cumulative = 0;
        for (int u = profile.left[vertex]; u <= profile.right[vertex]; ++u) {
            cumulative += count[u];
            if (cumulative >= rank) return u;
        }
        throw std::runtime_error("naive target rank not found");
    };

    while (out.steps < maximum_steps) {
        vector<int64_t> unstable_weight(ell, 0);
        int64_t total_unstable = 0;
        for (int v = 0; v < ell; ++v) {
            if (count[v] > 0 && target(v) != v) {
                unstable_weight[v] = count[v];
                total_unstable += count[v];
            }
        }
        if (total_unstable == 0) return summarize_outcome(profile, count, out);

        int64_t draw = rng.below(total_unstable);
        int source = 0;
        while (draw >= unstable_weight[source]) {
            draw -= unstable_weight[source];
            ++source;
        }
        int destination = target(source);
        if (destination < source) ++out.moves_left;
        if (destination > source) ++out.moves_right;
        out.cut_crossings += std::abs(destination - source);
        out.metric_distance += std::abs(profile.position[destination] - profile.position[source]);
        --count[source];
        ++count[destination];
        ++out.steps;
    }
    out.truncated = true;
    return summarize_outcome(profile, count, out);
}

vector<Rule> standard_rules() {
    return {
        {"lower_median", RuleCode::LOWER_MEDIAN, 0, 0},
        {"5-large", RuleCode::K_LARGE, 5, 1},
        {"25-large", RuleCode::K_LARGE, 25, 2},
        {"max", RuleCode::K_LARGE, 1, 3},
    };
}

vector<ProfileSpec> distribution_specs() {
    const int n = 4000;
    const double rho = 10.0;
    vector<ProfileSpec> specs;
    int id = 0;
    specs.push_back({"uniform", "agent_density", "singleton", n, n, rho, 0, id++});
    specs.push_back({"beta22", "agent_density", "singleton", n, n, rho, 0, id++});
    specs.push_back({"jittered_lattice", "agent_density", "singleton", n, n, rho, 0, id++});
    specs.push_back({"clustered_blocks", "clustered", "singleton", n, n, rho, 10, id++});

    for (int ell : {4000, 2000, 1000, 400})
        specs.push_back({"uniform_discrete", "agent_density", "balanced", n, ell, rho, 0, id++});
    for (int ell : {2000, 1000, 400})
        specs.push_back({"uniform_discrete", "agent_density", "multinomial", n, ell, rho, 0, id++});
    for (int ell : {2000, 1000, 400})
        specs.push_back({"uniform_discrete", "support_density", "balanced", n, ell, rho, 0, id++});
    return specs;
}

vector<ProfileSpec> scaling_specs() {
    vector<ProfileSpec> specs;
    int id = 0;
    for (int n : {500, 1000, 2000, 4000, 8000}) {
        for (double rho : {10.0, 20.0, 35.0})
            specs.push_back({"uniform", "agent_density", "singleton", n, n, rho, 0, id++});
    }
    return specs;
}

vector<ProfileSpec> rank_density_specs() {
    vector<ProfileSpec> specs;
    int id = 0;
    for (double rho : {5.0, 10.0, 20.0, 35.0})
        specs.push_back({"uniform", "agent_density", "singleton", 4000, 4000, rho, 0, id++});
    return specs;
}

vector<Rule> rank_density_rules(double rho) {
    vector<Rule> rules{{"lower_median", RuleCode::LOWER_MEDIAN, 0, 0}};
    std::set<int> ks{1};
    for (double c : {0.5, 0.75, 1.0, 1.25, 1.5, 2.0})
        ks.insert(std::max(1, int(std::llround(c * (rho + 1.0)))));
    int id = 1;
    for (int k : ks) rules.push_back({"k-large", RuleCode::K_LARGE, k, id++});
    return rules;
}

void write_header(std::ostream& out) {
    out << "generator_version,design,distribution,span_mode,occupancy,n,target_ell,rho,"
           "cluster_count,trial,rule,rule_id,k,steps,T_over_n,initial_ell,initial_components,"
           "initial_largest_component_agents,initial_largest_component_support,initial_span,"
           "consensus_possible,global_consensus,component_consensus,component_consensus_fraction,"
           "final_opinions,final_components,consensus_components,"
           "final_opinions_per_initial_component,largest_final_opinion_fraction,"
           "largest_final_component_fraction,moves_left,moves_right,cut_crossings,"
           "metric_distance,truncated,profile_seed,activation_seed\n";
}

void write_row(std::ostream& out, const string& design, const ProfileSpec& spec,
               int trial, const Rule& rule, const Profile& profile,
               const Outcome& outcome, uint64_t profile_seed, uint64_t activation_seed) {
    out << GENERATOR_VERSION << ',' << design << ',' << spec.distribution << ','
        << spec.span_mode << ',' << spec.occupancy << ',' << spec.n << ',' << spec.ell << ','
        << spec.rho << ',' << spec.cluster_count << ',' << trial << ',' << rule.name << ','
        << rule.id << ',' << rule.k << ',' << outcome.steps << ','
        << std::setprecision(12) << double(outcome.steps) / spec.n << ','
        << profile.position.size() << ',' << profile.component_count << ','
        << profile.largest_component_agents << ',' << profile.largest_component_support << ','
        << profile.span << ',' << (profile.component_count == 1) << ','
        << outcome.global_consensus << ',' << outcome.component_consensus << ','
        << outcome.component_consensus_fraction << ',' << outcome.final_opinions << ','
        << outcome.final_components << ',' << outcome.consensus_components << ','
        << outcome.final_opinions_per_initial_component << ','
        << outcome.largest_final_opinion_fraction << ','
        << outcome.largest_final_component_fraction << ',' << outcome.moves_left << ','
        << outcome.moves_right << ',' << outcome.cut_crossings << ','
        << outcome.metric_distance << ',' << outcome.truncated << ','
        << profile_seed << ',' << activation_seed << '\n';
}

int run_selftest() {
    int tests = 0;
    vector<Rule> rules = {
        {"lower_median", RuleCode::LOWER_MEDIAN, 0, 0},
        {"max", RuleCode::K_LARGE, 1, 1},
        {"3-large", RuleCode::K_LARGE, 3, 2},
        {"2-small", RuleCode::K_SMALL, 2, 3},
    };
    for (int n : {7, 12, 19, 25}) {
        for (int trial = 1; trial <= 20; ++trial) {
            for (string occupancy : {string("singleton"), string("balanced"), string("multinomial")}) {
                int ell = occupancy == "singleton" ? n : std::max(2, n / 3);
                ProfileSpec spec{"uniform_discrete", "agent_density", occupancy,
                                 n, ell, 3.0, 0, trial};
                uint64_t pseed = splitmix_hash(0x51f15eEDULL ^ uint64_t(n * 1000 + trial * 17 + ell));
                Profile profile = make_profile(spec, pseed);
                for (const Rule& rule : rules) {
                    uint64_t aseed = splitmix_hash(pseed ^ uint64_t(rule.id + 1));
                    Outcome fast = simulate(profile, rule, aseed);
                    Outcome slow = simulate_naive(profile, rule, aseed);
                    ++tests;
                    if (fast.steps != slow.steps || fast.final_count != slow.final_count ||
                        fast.cut_crossings != slow.cut_crossings) {
                        std::cerr << "SELFTEST FAILURE n=" << n << " trial=" << trial
                                  << " occupancy=" << occupancy << " rule=" << rule.name << '\n';
                        return 1;
                    }
                }
            }
        }
    }
    std::cerr << "selftest passed: " << tests << " exact optimized-vs-naive comparisons\n";
    return 0;
}

void usage(const char* program) {
    std::cerr << "Usage:\n  " << program
              << " --design selftest\n  " << program
              << " --design {distributions|scaling|rank_density|density_scaling} --reps N --output FILE\n"
              << "  density_scaling options: --ns 500,1000,... --rhos 10,12,... --ks 1,5,...\n"
              << "  --trial-start N (default 1; deterministic shards)\n";
}

vector<double> parse_values(const string& text) {
    std::stringstream stream(text);
    vector<double> values;
    string part;
    while (std::getline(stream, part, ',')) values.push_back(std::stod(part));
    if (values.empty()) throw std::runtime_error("empty parameter list");
    return values;
}

}  // namespace

int main(int argc, char** argv) {
    string design;
    string output_path;
    int repetitions = 100;
    int trial_start = 1;
    vector<double> ns{250,500,1000,2000,4000,8000,16000,32000};
    vector<double> rhos{10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40};
    vector<double> ks{1,5,10};
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--design" && i + 1 < argc) design = argv[++i];
        else if (arg == "--reps" && i + 1 < argc) repetitions = std::stoi(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (arg == "--ns" && i + 1 < argc) ns = parse_values(argv[++i]);
        else if (arg == "--rhos" && i + 1 < argc) rhos = parse_values(argv[++i]);
        else if (arg == "--ks" && i + 1 < argc) ks = parse_values(argv[++i]);
        else if (arg == "--trial-start" && i + 1 < argc) trial_start = std::stoi(argv[++i]);
        else {
            usage(argv[0]);
            return 2;
        }
    }
    if (design == "selftest") return run_selftest();
    if (design.empty() || output_path.empty() || repetitions <= 0 || trial_start <= 0) {
        usage(argv[0]);
        return 2;
    }

    vector<ProfileSpec> specs;
    if (design == "distributions") specs = distribution_specs();
    else if (design == "scaling") specs = scaling_specs();
    else if (design == "rank_density") specs = rank_density_specs();
    else if (design == "density_scaling") {
        int id = 0;
        for (double n : ns) for (double rho : rhos) {
            if (n < 1 || n != std::floor(n) || rho <= 0)
                throw std::runtime_error("invalid n or rho");
            specs.push_back({"uniform", "agent_density", "singleton", int(n), int(n), rho, 0, id++});
        }
    }
    else {
        usage(argv[0]);
        return 2;
    }

    std::ofstream file(output_path);
    if (!file) {
        std::cerr << "cannot open output file: " << output_path << '\n';
        return 2;
    }
    write_header(file);

    const uint64_t master = splitmix_hash(0x20261007ULL ^ fnv1a(design));
    int completed = 0;
    for (const ProfileSpec& spec : specs) {
        for (int trial = trial_start; trial < trial_start + repetitions; ++trial) {
            uint64_t profile_seed = splitmix_hash(master ^ uint64_t(spec.id + 1) * 0xD1B54A32D192ED03ULL
                                                  ^ uint64_t(trial));
            if (design == "density_scaling") {
                std::ostringstream key;
                key << spec.n << ':' << std::setprecision(17) << spec.rho;
                profile_seed = splitmix_hash(master ^ fnv1a(key.str()) ^ splitmix_hash(uint64_t(trial)));
            }
            Profile profile = make_profile(spec, profile_seed);
            vector<Rule> rules = design == "rank_density" ? rank_density_rules(spec.rho)
                                                           : standard_rules();
            if (design == "density_scaling") {
                rules = {{"lower_median", RuleCode::LOWER_MEDIAN, 0, 0}};
                for (double k : ks) {
                    if (k < 1 || k != std::floor(k)) throw std::runtime_error("invalid k");
                    rules.push_back({k == 1 ? "max" : std::to_string(int(k)) + "-large",
                                     RuleCode::K_LARGE, int(k), int(k)});
                }
            }
            for (const Rule& rule : rules) {
                uint64_t activation_seed = splitmix_hash(
                    profile_seed ^ fnv1a(rule.name) ^ uint64_t(rule.k + 1) * 0x94D049BB133111EBULL);
                Outcome outcome = simulate(profile, rule, activation_seed);
                write_row(file, design, spec, trial, rule, profile, outcome,
                          profile_seed, activation_seed);
                if (outcome.truncated) {
                    std::cerr << "warning: truncated run for spec=" << spec.id
                              << " trial=" << trial << " rule=" << rule.name << '\n';
                }
            }
        }
        ++completed;
        std::cerr << "completed " << completed << '/' << specs.size()
                  << " configurations for design " << design << '\n';
    }
    return 0;
}
