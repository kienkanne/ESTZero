#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "utils.h"
#include "mp2.h"
#include "timer.h"


using Vector = Eigen::VectorXd;
using Tensor4D = Eigen::Tensor<double, 4>;


double calc_rhf_MP2_E(const Tensor4D& I_mo, const Vector& eps, int n_occ, double E0) {
    timing::Scoped _t("MP2 Energy Calculation");

    const int nbf = eps.size();
    const int n_virt = nbf - n_occ;

    const Vector& eps_occ  = eps.head(n_occ);
    const Vector& eps_virt = eps.tail(n_virt);

    double os_E = 0.0;
    double ss_E = 0.0;

    for (int i = 0; i < n_occ; ++i) {
    for (int a = 0; a < n_virt; ++a) {
    for (int j = 0; j < n_occ; ++j) {
    for (int b = 0; b < n_virt; ++b) {
        const double denominator = 1.0 / (eps_occ(i) - eps_virt(a) + eps_occ(j)  - eps_virt(b));

        const double direct = I_mo(i, a, j, b);
        const double exchange = I_mo(i, b, j, a);

        os_E += direct * direct * denominator;
        ss_E += direct * (direct - exchange) * denominator;
    }
    }
    }
    }

    return E0 + os_E + ss_E;
}


double calc_uhf_MP2_E(const Tensor4D& Iaa_mo, const Tensor4D& Ibb_mo, const Tensor4D& Iab_mo, 
                    const Vector& epsa, const Vector& epsb, int n_alpha, int n_beta, double E0) {
    timing::Scoped _t("MP2 Energy Calculation");

    const int nbf = epsa.size();
    const int n_avirt = nbf - n_alpha;
    const int n_bvirt = nbf - n_beta;

    const Vector& epsa_occ  = epsa.head(n_alpha);
    const Vector& epsa_virt = epsa.tail(n_avirt);
    const Vector& epsb_occ  = epsb.head(n_beta);
    const Vector& epsb_virt = epsb.tail(n_bvirt);

    double E_aa = 0.0;
    double E_bb = 0.0;
    double E_ab = 0.0;

    for (int i = 0; i < n_alpha; ++i) {
    for (int a = 0; a < n_avirt; ++a) {
    for (int j = 0; j < n_alpha; ++j) {
    for (int b = 0; b < n_avirt; ++b) {
        const double denominator = 1.0 / (epsa_occ(i) - epsa_virt(a)
                                         + epsa_occ(j)  - epsa_virt(b));

        const double direct = Iaa_mo(i, a, j, b);
        const double exchange = Iaa_mo(i, b, j, a);

        E_aa += 0.25 * (direct - exchange) * (direct - exchange) * denominator;
    }
    }
    }
    }

    for (int i = 0; i < n_beta; ++i) {
    for (int a = 0; a < n_bvirt; ++a) {
    for (int j = 0; j < n_beta; ++j) {
    for (int b = 0; b < n_bvirt; ++b) {
        const double denominator = 1.0 / (epsb_occ(i) - epsb_virt(a)
                                         + epsb_occ(j)  - epsb_virt(b));

        const double direct = Ibb_mo(i, a, j, b);
        const double exchange = Ibb_mo(i, b, j, a);

        E_bb += 0.25 * (direct - exchange) * (direct - exchange) * denominator;
    }
    }
    }
    }

    for (int i = 0; i < n_alpha; ++i) {
    for (int a = 0; a < n_avirt; ++a) {
    for (int j = 0; j < n_beta; ++j) {
    for (int b = 0; b < n_bvirt; ++b) {
        const double denominator = 1.0 / (epsa_occ(i) - epsa_virt(a)
                                         + epsb_occ(j)  - epsb_virt(b));

        const double direct = Iab_mo(i, a, j, b);

        E_ab += direct * direct * denominator;
    }
    }
    }
    }

    return E0 + E_aa + E_bb + E_ab;
}
