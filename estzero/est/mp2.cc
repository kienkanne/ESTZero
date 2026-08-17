#include <Eigen/Dense>
#include "mp2.h"
#include "tensor.h"
#include "timer.h"

double calc_rhf_mp2_E(const Tensor& Iiajb, const Vector& eps, int n_occ, double E0) {
    timing::Scoped _t("MP2 Energy Calculation");

    double E_MP2 = E0;

    const int nbf = eps.size();
    const int n_virt = nbf - n_occ;

    Tensor eps_t = vec_to_tensor(eps);

    const Tensor& eps_occ = eps_t.slice(Sli{0, n_occ});
    const Tensor& eps_virt = eps_t.slice(Sli{n_occ, -1});

    Tensor Eiajb({n_occ, n_virt, n_occ, n_virt});

    for (int i = 0; i < n_occ; ++i) {
    for (int a = 0; a < n_virt; ++a) {
    for (int j = 0; j < n_occ; ++j) {
    for (int b = 0; b < n_virt; ++b) {
        Eiajb(i, a, j, b) = 1.0 / (eps_occ(i) - eps_virt(a) + eps_occ(j)  - eps_virt(b));
    }}}}

    E_MP2 += (Iiajb * Eiajb * (2 * Iiajb - Iiajb.swapaxes(1,3))).sum();

    return E_MP2;
}


double calc_uhf_mp2_E(const Tensor& Iiajb_AA, const Tensor& Iiajb_BB, const Tensor& Iiajb_AB,
                    const Vector& epsA, const Vector& epsB, int n_alpha, int n_beta, double E0) {
    timing::Scoped _t("MP2 Energy Calculation");

    double E_MP2 = 0.0;

    const int nbf = epsA.size();
    const int n_avirt = nbf - n_alpha;
    const int n_bvirt = nbf - n_beta;

    Tensor epsA_t = vec_to_tensor(epsA);
    Tensor epsB_t = vec_to_tensor(epsB);

    const Tensor& epsA_occ = epsA_t.slice(Sli{0, n_alpha});
    const Tensor& epsA_virt = epsA_t.slice(Sli{n_alpha, -1});
    const Tensor& epsB_occ = epsB_t.slice(Sli{0, n_beta});
    const Tensor& epsB_virt = epsB_t.slice(Sli{n_beta, -1});

    {
    Tensor Eaa_iajb({n_alpha, n_avirt, n_alpha, n_avirt});

    for (int i = 0; i < n_alpha; ++i) {
    for (int a = 0; a < n_avirt; ++a) {
    for (int j = 0; j < n_alpha; ++j) {
    for (int b = 0; b < n_avirt; ++b) {
        Eaa_iajb(i, a, j, b) = 1.0 / (epsA_occ(i) - epsA_virt(a) + epsA_occ(j) - epsA_virt(b));
    }}}}

    E_MP2 += 0.25 * (((Iiajb_AA - Iiajb_AA.swapaxes(1,3))^2) * Eaa_iajb).sum();
    }

    {
    Tensor Ebb_iajb({n_beta, n_bvirt, n_beta, n_bvirt});

    for (int i = 0; i < n_beta; ++i) {
    for (int a = 0; a < n_bvirt; ++a) {
    for (int j = 0; j < n_beta; ++j) {
    for (int b = 0; b < n_bvirt; ++b) {
        Ebb_iajb(i, a, j, b) = 1.0 / (epsB_occ(i) - epsB_virt(a) + epsB_occ(j) - epsB_virt(b));
    }}}}

    E_MP2 += 0.25 * (((Iiajb_BB - Iiajb_BB.swapaxes(1,3))^2) * Ebb_iajb).sum();
    }
    
    {
    Tensor Eab_iajb({n_alpha, n_avirt, n_beta, n_bvirt});

    for (int i = 0; i < n_alpha; ++i) {
    for (int a = 0; a < n_avirt; ++a) {
    for (int j = 0; j < n_beta; ++j) {
    for (int b = 0; b < n_bvirt; ++b) {
        Eab_iajb(i, a, j, b) = 1.0 / (epsA_occ(i) - epsA_virt(a) + epsB_occ(j) - epsB_virt(b));
    }}}}

    E_MP2 += ((Iiajb_AB^2) * Eab_iajb).sum();
    }

    return E0 + E_MP2;
}


double calc_rhf_MP2_E_old(const Tensor& Imo, const Vector& eps, int n_occ, double E0) {
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

        const double direct = Imo(i, a, j, b);
        const double exchange = Imo(i, b, j, a);

        os_E += direct * direct * denominator;
        ss_E += direct * (direct - exchange) * denominator;
    }}}}

    return E0 + os_E + ss_E;
}


double calc_uhf_MP2_old(const Tensor& Iaa_mo, const Tensor& Ibb_mo, const Tensor& Iab_mo,
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
    }}}}

    for (int i = 0; i < n_beta; ++i) {
    for (int a = 0; a < n_bvirt; ++a) {
    for (int j = 0; j < n_beta; ++j) {
    for (int b = 0; b < n_bvirt; ++b) {
        const double denominator = 1.0 / (epsb_occ(i) - epsb_virt(a)
                                         + epsb_occ(j)  - epsb_virt(b));

        const double direct = Ibb_mo(i, a, j, b);
        const double exchange = Ibb_mo(i, b, j, a);

        E_bb += 0.25 * (direct - exchange) * (direct - exchange) * denominator;
    }}}}

    for (int i = 0; i < n_alpha; ++i) {
    for (int a = 0; a < n_avirt; ++a) {
    for (int j = 0; j < n_beta; ++j) {
    for (int b = 0; b < n_bvirt; ++b) {
        const double denominator = 1.0 / (epsa_occ(i) - epsa_virt(a)
                                         + epsb_occ(j)  - epsb_virt(b));

        const double direct = Iab_mo(i, a, j, b);

        E_ab += direct * direct * denominator;
    }}}}

    return E0 + E_aa + E_bb + E_ab;
}
