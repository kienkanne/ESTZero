#ifndef TRIPLES_LEGACY
#define TRIPLES_LEGACY

#include "tensor.h"
#include "buffers.h"

Tensor t3c_ijk_spin_old(const Tensor& t2, VikingsBufferSpin& vb, const int o, const int v,
               const Tensor& F_occ, const Tensor& F_virt, int i, int j, int k);

Tensor t3c_ijk_spat_old(const Tensor& t2, VikingsBufferSpat& vb, const int o, const int v,
               const Tensor& F_occ, const Tensor& F_virt, int i, int j, int k);

// "Noddy" implementation of the perturbative triples correction: creates the full O(o^3 v^3) T3
// tensor directly, unlike calc_pert_triples's batched i/j/k loop. UHF/spin-orbital only. Direct
// line-by-line port of Psi4Numpy's spin-orbital CCSD_T module (Crawford's Programming Project #6).
// References:
// https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2306
// https://github.com/psi4/psi4numpy/blob/master/Coupled-Cluster/Spin_Orbitals/CCSD/CCSD_T.py
double calc_pert_triples_noddy(const Tensor& Iprqs, const Tensor& t1, const Tensor& t2,
                        const Tensor& FN, const int nbf, const int n_occ, Sli O, Sli V);

#endif
