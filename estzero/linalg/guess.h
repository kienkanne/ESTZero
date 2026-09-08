#ifndef GUESS
#define GUESS

#include <stdexcept>
#include <string>
#include "tensor.h"

enum class GuessScheme { CORE, GWH };

inline GuessScheme parse_guess_scheme(const std::string& g) {
    if (g == "core") return GuessScheme::CORE;
    if (g == "gwh")  return GuessScheme::GWH;
    throw std::runtime_error("Unknown guess: " + g);
}

// Empirical Wolfsberg-Helmholz scaling constant
inline constexpr double GWH_K = 1.75;

// Generalized Wolfsberg-Helmholz initial guess. Add an additional option to the core guess F = H.
inline Matrix gwh_guess(const Matrix& H, const Matrix& S) {
    int n = H.rows();
    Matrix F(n, n);
    for (int p = 0; p != n; ++p) {
    for (int q = 0; q != n; ++q) {
        F(p, q) = (p == q) ? H(p, p) : 0.5 * GWH_K * S(p, q) * (H(p, p) + H(q, q));
    }}
    return F;
}

inline Matrix initial_guess(const Matrix& H, const Matrix& S, GuessScheme guess) {
    switch (guess) {
        case GuessScheme::CORE: return H;
        case GuessScheme::GWH:  return gwh_guess(H, S);
    }
    throw std::runtime_error("initial_guess: unhandled GuessScheme");
}

#endif
