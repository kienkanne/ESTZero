#include <algorithm>
#include <cmath>
#include <print>
#include <vector>
#include <libint2.hpp>

#include "integrals.h"
#include "tensor.h"

Matrix build_one_center(libint2::Engine& engine, const libint2::BasisSet& obs) {
    Matrix mat (obs.nbf(), obs.nbf());
    mat.setZero();

    auto shell2bf = obs.shell2bf(); // maps shell index to basis function index

    const auto& buf_vec = engine.results(); // will point to computed shell sets

    for(auto s1=0;  s1!=obs.size(); ++s1) {
    for(auto s2=s1; s2!=obs.size(); ++s2) {
        engine.compute(obs[s1], obs[s2]);

        auto ints = buf_vec[0];  // location of the computed integrals
        if (ints == nullptr)
            continue;  // nullptr returned if the entire shell-set was screened out

        auto bf1 = shell2bf[s1];  // first basis function in first shell
        auto n1 = obs[s1].size(); // number of basis functions in first shell
        auto bf2 = shell2bf[s2];  // first basis function in second shell
        auto n2 = obs[s2].size(); // number of basis functions in second shell

        // integrals are packed into ints_shellset in row-major (C) form
        for(auto f1=0; f1!=n1; ++f1) {
        for(auto f2=0; f2!=n2; ++f2) {
            mat(bf1+f1, bf2+f2) = ints[f1*n2+f2];
            if (bf1+f1 != bf2+f2) { mat(bf2+f2, bf1+f1) = ints[f1*n2+f2]; }
        }}
    }}

    return mat;
}


namespace {
void report_schwarz(int verbose, size_t skipped, size_t total) {
    if (verbose < 1 || total == 0) return;
    std::println("Schwarz screening: skipped {} / {} shell quartets ({:.1f}%)",
                  skipped, total, 100.0 * static_cast<float>(skipped) / static_cast<float>(total));
}
} // namespace

Matrix compute_schwarz_bounds(libint2::Engine& engine, const libint2::BasisSet& obs) {
    int n_shells = static_cast<int>(obs.size());
    Matrix Q(n_shells, n_shells);
    Q.setZero();

    const auto& buf_vec = engine.results();

    for (int s1 = 0; s1 != n_shells; ++s1) {
    for (int s2 = s1; s2 != n_shells; ++s2) { // unique pairs only
        engine.compute(obs[s1], obs[s2], obs[s1], obs[s2]);
        auto ints = buf_vec[0];

        double max_val = 0.0;
        if (ints != nullptr) {
            int n1 = obs[s1].size();
            int n2 = obs[s2].size();

            for (int f1 = 0; f1 != n1; ++f1) {
            for (int f2 = 0; f2 != n2; ++f2) {
                // Only check diagonal elements (f1 f2|f1 f2) within the (s1 s2 s1 s2) shell
                // quartet row-major format (dims n1,n2,n1,n2; f3==f1, f4==f2).
                int idx = ((f1 * n2 + f2) * n1 + f1) * n2 + f2;
                max_val = std::max(max_val, std::abs(ints[idx]));
            }}
        }

        double bound = std::sqrt(max_val);
        Q(s1, s2) = bound;
        Q(s2, s1) = bound;
    }}

    return Q;
}

Tensor build_eri_tensor(libint2::Engine& engine, const libint2::BasisSet& obs, int verbose) {
    int nbf = obs.nbf();
    Tensor eri_tensor({nbf, nbf, nbf, nbf});
    eri_tensor.fill(0.0);

    auto shell2bf = obs.shell2bf();
    const auto& buf_vec = engine.results();
    Matrix Q = compute_schwarz_bounds(engine, obs);

    size_t total_quartets = 0, skipped_quartets = 0;

    for (int s1 = 0; s1 != obs.size(); ++s1) {
    for (int s2 = s1; s2 != obs.size(); ++s2) {
    for (int s3 = s1; s3 != obs.size(); ++s3) {
        int s4_min = (s1 == s3) ? s2 : s3;
    for (int s4 = s4_min;  s4 != obs.size(); ++s4) {
        ++total_quartets;
        if (Q(s1, s2) * Q(s3, s4) < SCHWARZ_TOL) { ++skipped_quartets; continue; }

        engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]);
        auto ints = buf_vec[0];
        if (ints == nullptr) continue;

        auto bf1 = shell2bf[s1]; auto n1 = obs[s1].size();
        auto bf2 = shell2bf[s2]; auto n2 = obs[s2].size();
        auto bf3 = shell2bf[s3]; auto n3 = obs[s3].size();
        auto bf4 = shell2bf[s4]; auto n4 = obs[s4].size();

        // Not fully optimized for when the shells are on a diagonal
        int index = 0;
        for (int f1 = 0; f1 != n1; ++f1) {
        for (int f2 = 0; f2 != n2; ++f2) {
        for (int f3 = 0; f3 != n3; ++f3) {
        for (int f4 = 0; f4 != n4; ++f4) {

            int i = bf1 + f1;
            int j = bf2 + f2;
            int k = bf3 + f3;
            int l = bf4 + f4;

            double val = ints[index++];

            // 8-fold symmetry (overwrites on diagonals):
            // Chemists notation: (ij||kl)
            eri_tensor(i, j, k, l) = val;
            eri_tensor(j, i, k, l) = val;
            eri_tensor(i, j, l, k) = val;
            eri_tensor(j, i, l, k) = val;
            eri_tensor(k, l, i, j) = val;
            eri_tensor(l, k, i, j) = val;
            eri_tensor(k, l, j, i) = val;
            eri_tensor(l, k, j, i) = val;

        }}}}
    }}}}

    report_schwarz(verbose, skipped_quartets, total_quartets);
    return eri_tensor;
}

std::vector<double> build_eri_vector(libint2::Engine& engine, const libint2::BasisSet& obs, int verbose) {
    int nbf = obs.nbf();
    int n_pairs = nbf * (nbf + 1) / 2;
    std::vector<double> eri_vector(n_pairs * (n_pairs + 1) / 2, 0.0);

    auto shell2bf = obs.shell2bf();
    const auto& buf_vec = engine.results();
    Matrix Q = compute_schwarz_bounds(engine, obs);

    size_t total_quartets = 0, skipped_quartets = 0;

    // Loop over shell quartets
    for (int s1 = 0; s1 != obs.size(); ++s1) {
    for (int s2 = s1; s2 != obs.size(); ++s2) {
    for (int s3 = s1; s3 != obs.size(); ++s3) {
        int s4_min = (s1 == s3) ? s2 : s3;
    for (int s4 = s4_min;  s4 != obs.size(); ++s4) {
        ++total_quartets;
        if (Q(s1, s2) * Q(s3, s4) < SCHWARZ_TOL) { ++skipped_quartets; continue; }

        engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]);
        auto ints = buf_vec[0];
        if (ints == nullptr) continue;

        auto bf1 = shell2bf[s1]; auto n1 = obs[s1].size();
        auto bf2 = shell2bf[s2]; auto n2 = obs[s2].size();
        auto bf3 = shell2bf[s3]; auto n3 = obs[s3].size();
        auto bf4 = shell2bf[s4]; auto n4 = obs[s4].size();

        // Libint2 packs 4-center integrals in row-major order: (f1, f2, f3, f4)
        // Not fully optimized for when the shells are on a diagonal
        int index = 0;
        for (int f1 = 0; f1 != n1; ++f1) {
        for (int f2 = 0; f2 != n2; ++f2) {
        for (int f3 = 0; f3 != n3; ++f3) {
        for (int f4 = 0; f4 != n4; ++f4) {
            
            // Use the current index, then increase
            double val = ints[index++];

            int i = bf1 + f1;
            int j = bf2 + f2;
            int k = bf3 + f3;
            int l = bf4 + f4;

            int canonical_idx = eri_index(i, j, k, l);
            eri_vector[canonical_idx] = val;
        }}}}
    }}}}

    report_schwarz(verbose, skipped_quartets, total_quartets);
    return eri_vector;
}

// Transform the eri_vector to eri_tensor
Tensor eri_vector_to_tensor(const std::vector<double>& eri_vector, int nbf) {
    std::vector<double> data(static_cast<size_t>(nbf) * nbf * nbf * nbf);
    size_t idx = 0;
    for (int p = 0; p < nbf; ++p) {
    for (int q = 0; q < nbf; ++q) {
    for (int r = 0; r < nbf; ++r) {
    for (int s = 0; s < nbf; ++s) {
        data[idx++] = eri_vector[eri_index(p, q, r, s)];
    }}}}
    return Tensor({nbf, nbf, nbf, nbf}, std::move(data)); // hand over nbf^4 buffer, no copy
}