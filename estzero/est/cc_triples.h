#ifndef TRIPLES
#define TRIPLES

#include "tensor.h"

double calc_pert_triples(const Tensor& Iprqs, const Tensor& DEprqs, const Tensor& t1, const Tensor& t2,
                        const Tensor& FN, const int nbf, const int n_occ, Sli O, Sli V, bool spin);
                        
double calc_pert_triples_noddy(const Tensor& Iprqs, const Tensor& t1, const Tensor&t2,
                        const Tensor& FN, const int nbf, const int n_occ, Sli O, Sli V);

#endif
