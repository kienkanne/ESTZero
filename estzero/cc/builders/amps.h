#ifndef AMPS
#define AMPS

#include "tensor.h"

Tensor build_spat_T1(const Tensor& Iprqs, const Tensor& FN,
                const Tensor& Fae, const Tensor& Fmi, const Tensor& Fme,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V, bool spin);

Tensor build_spat_T2(const Tensor& Iprqs, const Tensor& tau, // (takes tau, not FN)
                const Tensor& Fae, const Tensor& Fmi, const Tensor& Fme,
                const Tensor& Wmnij, const Tensor& Wmbej, const Tensor& Wmbje, const Tensor& Zmbij,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V);

double calc_spat_cc_E(const Tensor& Iprqs, const Tensor& FN, const Tensor& tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V);


double test_mp2_energy(const Tensor& DEprqs, const Tensor& Dia, const Tensor& Dijab, const int n_occ);

double calc_spin_cc_E(const Tensor& DEprqs, const Tensor& FN, const Tensor& tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V);

Tensor build_spin_T1(const Tensor& DEprqs, const Tensor& FN,
                const Tensor& Fae, const Tensor& Fmi, const Tensor& Fme,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V, bool spin);

Tensor build_spin_T2(const Tensor& DEprqs, const Tensor& tau, // (takes tau, not FN)
                const Tensor& Fae, const Tensor& Fmi, const Tensor& Fme,
                const Tensor& Wmnij, const Tensor& Wabef, const Tensor& Wmbej,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V, bool spin);

#endif