#ifndef CC_BUILD
#define CC_BUILD

#include "tensor.h"

struct Dia_Dijab {Tensor Dia; Tensor Dijab; };
Dia_Dijab build_Dia_Dijab(const Tensor& FN, const int nbf, const int n_occ, bool spin);

struct Taus { Tensor tau; Tensor tilde_tau;};
Taus build_taus(const Tensor& t1, const Tensor& t2, bool spin);


Tensor build_Fae(const Tensor& DEprqs, const Tensor& FN, const Tensor& tilde_tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V, bool spin);
                
Tensor build_Fmi(const Tensor& DEprqs, const Tensor& FN, const Tensor& tilde_tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V, bool spin);

Tensor build_Fme(const Tensor& DEprqs, const Tensor& FN, const Tensor& tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V, bool spin);


// UHF uses DEprqs <pr||qs>, while RHF uses Iprqs (the raw ERI tensor)
Tensor build_Wmnij(const Tensor& _prqs, const Tensor& FN, const Tensor& tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V, bool spin);

// UHF uses DEprqs <pr||qs>, while RHF uses Iprqs (the raw ERI tensor)
Tensor build_Wmbej(const Tensor& _prqs, const Tensor& FN, const Tensor& tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V, bool spin, bool is_MP3 = false);


// ONLY USED BY UHF
Tensor build_Wabef(const Tensor& DEprqs, const Tensor& FN, const Tensor& tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V);

// ONLY USED BY RHF
Tensor build_Wmbje(const Tensor& Iprqs, const Tensor& FN, const Tensor& tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V, bool is_MP3 = false);

// ONLY USED BY RHF
Tensor build_Zmbij(const Tensor& Iprqs, const Tensor& FN, const Tensor& tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V);

#endif