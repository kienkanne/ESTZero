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

inline CCResult cc_driver(WaveFunction wfn, SCFResult sr, std::string method, std::string reference,
                         double econv, int max_iter, int verbose) {
    Tensor Iprqs, DEprqs, FN, Dia, Dijab; 
    Sli O, V; Sli All{};
    bool spin; 
    bool is_CCD = (method == "ccd");

    const int n_occ = wfn.get_n_docc();
    const int n_alpha = wfn.get_n_alpha();
    const int n_beta = wfn.get_n_beta();
    const int nbf = wfn.get_nbf();

    const std::array<std::string, 3> METHODS{"ccd", "ccsd", "ccsd_t"};
    auto it = std::find(std::begin(METHODS), std::end(METHODS), method);
    if (it == std::end(METHODS)) { throw std::runtime_error("Unknown cc method: " + method); }

    if (reference == "rhf") {
        spin = false;
        
        O = Sli{0, n_occ};
        V = Sli{n_occ, nbf};

        const Matrix& H_ao = wfn.get_T() + wfn.get_V();
        Tensor H = mat_to_tensor(Matrix(sr.CA.transpose() * H_ao * sr.CA));

        // transmo_spat_pqrs returns chemist notation (pq|rs), and CCSD formula requires physicist <pq|rs> = (pr|qs),
        // so convert once here (matches build_spin_Iprqs's scatter helpers doing the same for UHF).
        Iprqs = transmo_spat_pqrs(wfn.get_eri_tensor(), sr.CA).swapaxes(1, 2);
        DEprqs = Iprqs * 2 - Iprqs.swapaxes(2, 3);

        FN = H + einsum("piqi->pq", {DEprqs.slice(All, O, All, O)});

        auto Dia_Dijab_ = build_Dia_Dijab(FN, nbf, n_occ, false);
        Dia = Dia_Dijab_.Dia; Dijab = Dia_Dijab_.Dijab;
    }

    else if (reference == "uhf") {
        spin = true;


        O = Sli {0, n_alpha + n_beta};
        V = Sli {n_alpha + n_beta, nbf*2};

        const auto& [Ipqrs_AA, Ipqrs_BB, Ipqrs_AB] = 
                    transmo_spin_pqrs(wfn.get_eri_tensor(), sr.CA, n_alpha, sr.CB, n_beta);

        Iprqs = build_spin_Iprqs(Ipqrs_AA, Ipqrs_BB, Ipqrs_AB, n_alpha, n_beta);
        DEprqs = Iprqs - Iprqs.swapaxes(2, 3);

        const Matrix& T_ints = wfn.get_T();
        const Matrix& V_ints = wfn.get_V();

        const Matrix& CA = sr.CA;
        const Matrix& CB = sr.CB;

        Tensor HA = mat_to_tensor(Matrix(CA.transpose() * (T_ints + V_ints) * CA));
        Tensor HB = mat_to_tensor(Matrix(CB.transpose() * (T_ints + V_ints) * CB));
        FN = build_spin_FN(HA, HB, DEprqs, n_alpha, n_beta);

        auto Dia_Dijab_ = build_Dia_Dijab(FN, nbf, n_alpha + n_beta, true);
        Dia = Dia_Dijab_.Dia; Dijab = Dia_Dijab_.Dijab;

    } else {
        throw std::runtime_error("Unknown reference: " + reference);
    }

    CCResult ccresult = cc_iterate(econv, max_iter, verbose, Iprqs, DEprqs, FN, Dia, Dijab, O, V, spin, is_CCD);
    if (method == "ccsd_t") {
        const int n_occ_triples = spin ? (n_alpha + n_beta) : n_occ;
        ccresult.E_pert_t = calc_pert_triples(Iprqs, DEprqs, ccresult.t1, ccresult.t2, FN, nbf,
                                               n_occ_triples, O, V, spin);
    }
    return ccresult;
}
#endif
