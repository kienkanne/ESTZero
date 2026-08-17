#ifndef CC_DRIVER
#define CC_DRIVER

#include "cc_iter.h"
#include "cc_spin.h"
#include "transmo.h"
#include "wfn.h"
#include <stdexcept>

inline CCResult cc_driver(WaveFunction wfn, SCFResult sr, std::string method, std::string reference,
                         double econv, int max_iter, int verbose) {
    
    if (reference == "rhf") {
        throw std::runtime_error("rhf ccsd in progress!");
    } 

    else if (reference == "uhf") {
        const Matrix& T_ints = wfn.get_T();
        const Matrix& V_ints = wfn.get_V();

        const Matrix& CA = sr.CA;
        const Matrix& CB = sr.CB;

        const int n_alpha = wfn.get_n_alpha();
        const int n_beta = wfn.get_n_beta();

        const auto& [Ipqrs_AA, Ipqrs_BB, Ipqrs_AB] = 
                    transmo_spin_pqrs(wfn.get_eri_tensor(), sr.CA, n_alpha, sr.CB, n_beta);

        Tensor ASprqs = build_spin_ASprqs(Ipqrs_AA, Ipqrs_BB, Ipqrs_AB, n_alpha, n_beta);

        Tensor HA = mat_to_tensor(Matrix(CA.transpose() * (T_ints + V_ints) * CA));
        Tensor HB = mat_to_tensor(Matrix(CB.transpose() * (T_ints + V_ints) * CB));
        Tensor FN = build_spin_FN(HA, HB, ASprqs, n_alpha, n_beta);
        auto [Dia, Dijab] = build_spin_Dia_Dijab(FN, n_alpha, n_beta);

        Sli O = Sli {0, n_alpha + n_beta};
        Sli V = Sli {n_alpha + n_beta, ASprqs.shape[0]};
        
        if (method == "ccsd") {
            return ccsd_iterate(econv, max_iter, verbose, ASprqs, FN, Dia, Dijab, O, V);
        } else {
            throw std::runtime_error(std::format("Unknown method: {}", method));
        }
        
    } else {
        throw std::runtime_error(std::format("Unknown reference: {}", reference));
    }
};

#endif
