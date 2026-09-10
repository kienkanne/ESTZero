#ifndef CC_DRIVER
#define CC_DRIVER

#include "cc_iter.h"
#include "cc_spin.h"
#include "cc_spat.h"
#include "cc_build.h"
#include "cc_triples.h"
#include "transmo.h"
#include "wfn.h"
#include <algorithm>
#include <array>
#include <stdexcept>

// A setup shared by cc_driver (ccd/ccsd/ccsd_t) and the MP3 module (mp3.cc). It builds the MO integral tensors 
// over the full range, the normal-ordered Fock matrix, and the energy denominators.
struct CCInputs { Tensor Iprqs, DEprqs, FN, Dia, Dijab; Sli O, V; bool spin; };

// For the frozen core scheme, only the active orbitals are built from the start (sliced out C to feed transmo).
// Everything downstream can use this result directly, and the active orbital starts at 0 directly without needing
// for complex slicing. The O and V slices are adjusted accordingly to the new number of active occupied orbitals.
inline CCInputs build_cc_inputs(WaveFunction& wfn, const SCFResult& sr, const std::string& reference,
                                 int n_frozen = 0) {
    CCInputs ccin;

    const int n_occ = wfn.get_n_docc();
    const int n_alpha = wfn.get_n_alpha();
    const int n_beta = wfn.get_n_beta();
    const int nbf = wfn.get_nbf();

    if (reference == "rhf") {
        ccin.spin = false;

        const int n_occ_active = n_occ - n_frozen;
        const int nbf_active = nbf - n_frozen;
        if (n_occ_active <= 0) {
            throw std::runtime_error(std::format("n_frozen ({}) >= n_occ ({})", n_frozen, n_occ));
        }

        ccin.O = Sli{0, n_occ_active};
        ccin.V = Sli{n_occ_active, nbf_active};

        Matrix C_active = sr.CA.rightCols(nbf_active);

        // transmo_spat_pqrs returns chemist notation (pq|rs), and CCSD formula requires physicist <pq|rs> = (pr|qs),
        // so convert once here (matches build_spin_Iprqs's scatter helpers doing the same for UHF).
        ccin.Iprqs = transmo_spat_pqrs(wfn.get_eri_tensor(), C_active).swapaxes(1, 2);
        ccin.DEprqs = ccin.Iprqs * 2 - ccin.Iprqs.swapaxes(2, 3);

        ccin.FN = build_diag_FN(Vector(sr.epsA.tail(nbf_active)));

        auto dd = build_Dia_Dijab(ccin.FN, ccin.O, ccin.V);
        ccin.Dia = dd.Dia; ccin.Dijab = dd.Dijab;
    }

    else if (reference == "uhf") {
        ccin.spin = true;

        const int n_alpha_active = n_alpha - n_frozen;
        const int n_beta_active = n_beta - n_frozen;
        const int nbf_active = nbf - n_frozen;

        const int n_occ_active = n_alpha_active + n_beta_active;
        if (n_alpha_active <= 0 || n_beta_active <= 0) {
            throw std::runtime_error(std::format("n_frozen ({}) >= n_occ ({})", n_frozen, n_occ_active));
        }

        ccin.O = Sli{0, n_occ_active};
        ccin.V = Sli{n_occ_active, nbf_active * 2};

        Matrix CA_active = sr.CA.rightCols(nbf_active);
        Matrix CB_active = sr.CB.rightCols(nbf_active);

        const auto& [Ipqrs_AA, Ipqrs_BB, Ipqrs_AB] =
                    transmo_spin_pqrs(wfn.get_eri_tensor(), CA_active, n_alpha_active, CB_active, n_beta_active);

        ccin.Iprqs = build_spin_Iprqs(Ipqrs_AA, Ipqrs_BB, Ipqrs_AB, n_alpha_active, n_beta_active);
        ccin.DEprqs = ccin.Iprqs - ccin.Iprqs.swapaxes(2, 3);

        ccin.FN = build_diag_spin_FN(Vector(sr.epsA.tail(nbf_active)), n_alpha_active,
                                      Vector(sr.epsB.tail(nbf_active)), n_beta_active);

        auto dd = build_Dia_Dijab(ccin.FN, ccin.O, ccin.V);
        ccin.Dia = dd.Dia; ccin.Dijab = dd.Dijab;

    } else {
        throw std::runtime_error("Unknown reference: " + reference);
    }

    return ccin;
}

inline CCResult cc_driver(WaveFunction wfn, SCFResult sr, std::string method, std::string reference,
                         double econv, int max_iter, int verbose, int n_frozen = 0) {
    bool is_CCD = (method == "ccd");

    const std::array<std::string, 3> METHODS{"ccd", "ccsd", "ccsd_t"};
    auto it = std::find(std::begin(METHODS), std::end(METHODS), method);
    if (it == std::end(METHODS)) { throw std::runtime_error("Unknown cc method: " + method); }

    CCInputs in = build_cc_inputs(wfn, sr, reference, n_frozen);

    CCResult ccresult = cc_iterate(econv, max_iter, verbose, in.Iprqs, in.DEprqs, in.FN, in.Dia, in.Dijab,
                                    in.O, in.V, in.spin, is_CCD);
    if (method == "ccsd_t") {
        ccresult.E_pert_t = calc_pert_triples(in.Iprqs, in.DEprqs, ccresult.t1, ccresult.t2, in.FN,
                                               in.O, in.V, in.spin);
    }
    return ccresult;
}
#endif
