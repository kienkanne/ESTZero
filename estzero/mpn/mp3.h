#ifndef MP3
#define MP3

#include "tensor.h"

double calc_mp3_E(const Tensor& Iprqs, const Tensor& DEprqs, const Tensor& FN,
                      const Tensor& Dia, const Tensor& Dijab, Sli O, Sli V, bool spin);

#endif
