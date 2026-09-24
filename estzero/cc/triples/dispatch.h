#ifndef TRIPLES
#define TRIPLES

#include "tensor.h"
#include "timer.h"

double calc_pert_triples_spin(const Tensor& DEprqs, const Tensor& t1, const Tensor& t2,
                            const Tensor& FN, Sli O, Sli V);

double calc_pert_triples_spat(const Tensor& Iprqs, const Tensor& DEprqs, const Tensor& t1, const Tensor& t2,
                            const Tensor& FN, Sli O, Sli V);

inline double calc_pert_triples(const Tensor& Iprqs, const Tensor& DEprqs, const Tensor& t1, const Tensor& t2,
                                const Tensor& FN, Sli O, Sli V, bool spin) {
    timing::Scoped _t("Perbutative Triples");

    return spin ? calc_pert_triples_spin(DEprqs, t1, t2, FN, O, V)
                : calc_pert_triples_spat(Iprqs, DEprqs, t1, t2, FN, O, V);
}

#endif
