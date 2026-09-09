#ifndef CC_DRIVER
#define CC_DRIVER

#include "cc_iter.h"
#include "cc_spin.h"
#include "cc_build.h"
#include "cc_triples.h"
#include "transmo.h"
#include "wfn.h"
#include "einsum.h"
#include <algorithm>
#include <array>
#include <stdexcept>

// A setup shared by cc_driver (ccd/ccsd/ccsd_t) and the MP3 module (mp3.cc). It builds the MO integral tensors 
// over the full range, the normal-ordered Fock matrix, and the energy denominators.
struct CCInputs { Tensor Iprqs, DEprqs, FN, Dia, Dijab; Sli O, V; bool spin; };

inline CCInputs build_cc_inputs(WaveFunction& wfn, const SCFResult& sr, const std::string& reference,
                                 int n_frozen = 0) {
    CCInputs ccin;
    const Sli All{};

    const int n_occ = wfn.get_n_docc();
    const int n_alpha = wfn.get_n_alpha();
    const int n_beta = wfn.get_n_beta();
    const int nbf = wfn.get_nbf();

    if (reference == "rhf") {
        ccin.spin = false;

        const Sli O_full{0, n_occ}; // used only to build FN
        ccin.O = Sli{n_frozen, n_occ}; // used for anything else downstream
        ccin.V = Sli{n_occ, nbf};

        const Matrix& H_ao = wfn.get_T() + wfn.get_V();
        Tensor H = mat_to_tensor(Matrix(sr.CA.transpose() * H_ao * sr.CA));

        // transmo_spat_pqrs returns chemist notation (pq|rs), and CCSD formula requires physicist <pq|rs> = (pr|qs),
        // so convert once here (matches build_spin_Iprqs's scatter helpers doing the same for UHF).
        ccin.Iprqs = transmo_spat_pqrs(wfn.get_eri_tensor(), sr.CA).swapaxes(1, 2);
        ccin.DEprqs = ccin.Iprqs * 2 - ccin.Iprqs.swapaxes(2, 3);

        ccin.FN = H + einsum("piqi->pq", {ccin.DEprqs.slice(All, O_full, All, O_full)});

        auto dd = build_Dia_Dijab(ccin.FN, ccin.O, ccin.V);
        ccin.Dia = dd.Dia; ccin.Dijab = dd.Dijab;
    }

    else if (reference == "uhf") {
        if (n_frozen > 0) {
            throw std::runtime_error("frozen core is not yet supported for UHF");
        }

        ccin.spin = true;

        ccin.O = Sli {0, n_alpha + n_beta};
        ccin.V = Sli {n_alpha + n_beta, nbf*2};

        const auto& [Ipqrs_AA, Ipqrs_BB, Ipqrs_AB] =
                    transmo_spin_pqrs(wfn.get_eri_tensor(), sr.CA, n_alpha, sr.CB, n_beta);

        ccin.Iprqs = build_spin_Iprqs(Ipqrs_AA, Ipqrs_BB, Ipqrs_AB, n_alpha, n_beta);
        ccin.DEprqs = ccin.Iprqs - ccin.Iprqs.swapaxes(2, 3);

        const Matrix& T_ints = wfn.get_T();
        const Matrix& V_ints = wfn.get_V();

        const Matrix& CA = sr.CA;
        const Matrix& CB = sr.CB;

        Tensor HA = mat_to_tensor(Matrix(CA.transpose() * (T_ints + V_ints) * CA));
        Tensor HB = mat_to_tensor(Matrix(CB.transpose() * (T_ints + V_ints) * CB));
        ccin.FN = build_spin_FN(HA, HB, ccin.DEprqs, n_alpha, n_beta);

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
        // The local width of the (possibly frozen-core-narrowed) occupied slice -- calc_pert_triples's
        // n_occ loop bound indexes into already-O-sliced tensors (t1, t2, F_occ, ...), not FN's global
        // diagonal, so it must match in.O's width, not the raw electron count.
        const int n_occ_triples = in.O.stop - in.O.start;
        ccresult.E_pert_t = calc_pert_triples(in.Iprqs, in.DEprqs, ccresult.t1, ccresult.t2, in.FN,
                                               wfn.get_nbf(), n_occ_triples, in.O, in.V, in.spin);
    }
    return ccresult;
}
#endif
