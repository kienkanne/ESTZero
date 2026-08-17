#ifndef CC_ITER
#define CC_ITER

#include "tensor.h"

struct CCResult {
    double E_ccsd;
    Tensor t1;
    Tensor t2;
};

CCResult ccsd_iterate(double econv, int max_iter, int verbose, const Tensor& ASpqrs, const Tensor& FN,
                        const Tensor& Dia, const Tensor& Dijab, Sli O, Sli V);

#endif
