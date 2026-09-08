#ifndef UHF
#define UHF

#include <tensor.h>
#include "wfn.h"
#include "guess.h"

SCFResult uhf_scf(WaveFunction wfn, double econv, int max_iter, int verbose, GuessScheme guess = GuessScheme::CORE);

#endif
