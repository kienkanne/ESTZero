#ifndef MP3
#define MP3

#include "tensor.h"

struct MP3Result { double E_mp2; double E_mp3_total; };

// MBPT(3): "one CC iteration where all nonlinear terms are skipped" (Stanton & Bartlett,
// J. Chem. Phys. 93, 6104 (1990); the same simplification is echoed in [Stanton:1991:4334]).
// T1 stays zero throughout (no singles at this order for a canonical reference); T2 starts at
// its MP2 value; the CCD intermediates are built with every T2-quadratic self-term dropped
// (tau/tilde_tau forced to zero for Fae/Fmi/Wmnij/Wabef, Wmbej/Wmbje's own T2-dependent piece
// skipped via their `linear` flag). One T2 update follows, and the same (T2-linear) CC energy
// formula evaluated on the result gives E_MP2 + E_MP3 cumulatively -- no separate summation.
MP3Result calc_mp3_E(const Tensor& Iprqs, const Tensor& DEprqs, const Tensor& FN,
                      const Tensor& Dia, const Tensor& Dijab, Sli O, Sli V, bool spin);

#endif
