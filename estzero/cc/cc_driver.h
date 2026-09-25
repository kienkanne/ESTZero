#ifndef CC_DRIVER
#define CC_DRIVER

#include <string>
#include "tensor.h"
#include "cc_iter.h" // CCResult

// Forward declarations only: wfn.cc includes this header, so including wfn.h back here would create a cycle.
// Both are only taken by reference, so the full definitions are only needed in cc_driver.cc.
class WaveFunction;
struct SCFResult;

// A setup shared by cc_driver (ccd/ccsd/ccsd_t) and the MP3 module (mp3.cc). It builds the MO integral tensors
// over the full range, the normal-ordered Fock matrix, and the energy denominators.
struct CCInputs { Tensor Iprqs, DEprqs, FN, Dia, Dijab; Sli O, V; bool spin; };

// For the frozen core scheme, only the active orbitals are built from the start (sliced out C to feed transmo).
// Everything downstream can use this result directly, and the active orbital starts at 0 directly without needing
// for complex slicing. The O and V slices are adjusted accordingly to the new number of active occupied orbitals.
CCInputs build_cc_inputs(const WaveFunction& wfn, const SCFResult& sr, const std::string& reference,
                         int n_frozen = 0);

CCResult cc_driver(const WaveFunction& wfn, const SCFResult& sr, const std::string& method,
                   const std::string& reference, double econv, int max_iter, int verbose, int n_frozen = 0);

#endif
