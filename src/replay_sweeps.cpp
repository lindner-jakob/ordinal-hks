// Stream bridge for the original NumPy-generated confidence graphs.
// Input: n rule_code k activation_seed, then n inclusive (left,right) pairs.
// Output: the stopping time for each graph. No archived stopping time is read.
#include <iostream>
#include "hks_sim_common.hpp"

int main() {
    int n, code, k;
    uint64_t seed;
    while (std::cin >> n >> code >> k >> seed) {
        if (n <= 0 || (code != 0 && code != 1) || k < 0) return 2;
        Instance ins;
        ins.n = n;
        ins.L.resize(n);
        ins.R.resize(n);
        for (int i = 0; i < n; ++i) {
            if (!(std::cin >> ins.L[i] >> ins.R[i]) || ins.L[i] < 0 ||
                ins.L[i] > i || ins.R[i] < i || ins.R[i] >= n) return 2;
        }
        std::cout << simulate_splitmix(ins, static_cast<RuleCode>(code), k, seed)
                  << std::endl;
    }
    return std::cin.eof() ? 0 : 2;
}
