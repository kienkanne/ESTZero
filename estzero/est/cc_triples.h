#ifndef TRIPLES
#define TRIPLES

#include "tensor.h"

// Vikings' formulation (Helgaker, Jorgensen & Olsen, Molecular Electronic-Structure Theory,
// Wiley, 2000, Ch. 14, pp. 794-795, Eqs. (14.6.62)-(14.6.64)), spin-orbital/UHF form. Occupied
// batched over (i,j,k): the connected T3 is built one (v,v,v) batch at a time and immediately
// contracted into the x1/x2 accumulators, so the full O(o^3 v^3) T3 is never materialized.
// Source: TODO link to pycc triples.py (t_vikings_so / t3c_ijk_so)
double calc_pert_triples(const Tensor& Iprqs, const Tensor& t1, const Tensor& t2,
                        const Tensor& FN, const int nbf, const int n_occ, Sli O, Sli V);

// Legacy "noddy" O(o^3 v^3) full-tensor implementation (Crawford Project #6 style). Kept for
// A-B comparison against the vikings path above; not called from cc_driver.h.
double calc_pert_triples_noddy(const Tensor& Iprqs, const Tensor& t1, const Tensor&t2,
                        const Tensor& FN, const int nbf, const int n_occ, Sli O, Sli V);

#endif
