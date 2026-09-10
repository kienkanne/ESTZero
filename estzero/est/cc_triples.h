#ifndef TRIPLES
#define TRIPLES

#include "tensor.h"

double calc_pert_triples(const Tensor& Iprqs, const Tensor& DEprqs, const Tensor& t1, const Tensor& t2,
                        const Tensor& FN, Sli O, Sli V, bool spin);

double calc_pert_triples_noddy(const Tensor& Iprqs, const Tensor& t1, const Tensor&t2,
                        const Tensor& FN, Sli O, Sli V);

#endif
