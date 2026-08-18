#ifndef CC_ITER
#define CC_ITER

#include "tensor.h"

struct CCResult {
    double E_cc;
    Tensor t1;
    Tensor t2;
};

CCResult cc_iterate(double econv, int max_iter, int verbose, const Tensor& ASprqs, const Tensor& FN,
                        const Tensor& Dia, const Tensor& Dijab, Sli O, Sli V, bool is_CCD = false);

#endif
