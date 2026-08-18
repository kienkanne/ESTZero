#ifndef CC_BUILD
#define CC_BUILD

#include "tensor.h"

struct Taus {
    Tensor tau; Tensor tilde_tau;
};

Taus build_taus(const Tensor& t1, const Tensor& t2);


Tensor build_Fae(const Tensor& ASprqs, const Tensor& FN, const Tensor& tilde_tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V);
                
Tensor build_Fmi(const Tensor& ASprqs, const Tensor& FN, const Tensor& tilde_tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V);

Tensor build_Fme(const Tensor& ASprqs, const Tensor& FN,
                const Tensor& t1,  Sli O, Sli V);


Tensor build_Wmnij(const Tensor& ASprqs, const Tensor& FN, const Tensor& tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V);

Tensor build_Wabef(const Tensor& ASprqs, const Tensor& FN, const Tensor& tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V);

Tensor build_Wmbej(const Tensor& ASprqs, const Tensor& FN, const Tensor& tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V);


Tensor build_T1(const Tensor& ASprqs, const Tensor& FN,
                const Tensor& Fae, const Tensor& Fmi, const Tensor& Fme,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V);

Tensor build_T2(const Tensor& ASprqs, const Tensor& tau,
                const Tensor& Fae, const Tensor& Fmi, const Tensor& Fme,
                const Tensor& Wmnij, const Tensor& Wabef, const Tensor& Wmbej,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V);

#endif