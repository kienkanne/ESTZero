#ifndef RHF
#define RHF

#include <tensor.h>
#include "wfn.h"

SCFResult rhf_scf(WaveFunction wfn, double econv, int max_iter, int verbose);

#endif
