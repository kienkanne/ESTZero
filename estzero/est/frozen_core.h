#ifndef FROZEN_CORE
#define FROZEN_CORE

#include <stdexcept>
#include <vector>
#include <libint2.hpp>

// Number of core orbitals frozen for one atom, by row of the periodic table
inline int core_orbitals_for_element(int atomic_number) {
    if (atomic_number <= 2)  return 0; // H, He
    if (atomic_number <= 10) return 1; // Li-Ne: 1s
    if (atomic_number <= 18) return 5; // Na-Ar: 1s, 2s, 2p x3
    throw std::runtime_error("core_orbitals_for_element: no frozen-core count tabulated past Ar (Z=18)");
}

inline int count_frozen_core(const std::vector<libint2::Atom>& atoms) {
    int n_frozen = 0;
    for (const auto& atom : atoms) { 
        n_frozen += core_orbitals_for_element(atom.atomic_number); \
    }
    return n_frozen;
}

#endif
