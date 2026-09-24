#ifndef CC_ITER
#define CC_ITER

#include "tensor.h"

// Kept separate from E_cc so callers can report the CCSD energy and the triples correction
struct CCResult {
    double E_cc;
    Tensor t1;
    Tensor t2;
    double E_pert_t = 0.0;
};

CCResult cc_iterate(double econv, int max_iter, int verbose, const Tensor& Iprqs, 
                    const Tensor& DEprqs, const Tensor& FN, const Tensor& Dia, const Tensor& Dijab, 
                    Sli O, Sli V, bool spin, bool is_CCD = false);

#endif
