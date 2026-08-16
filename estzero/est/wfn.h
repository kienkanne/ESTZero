#ifndef LOADER
#define LOADER

#include <Eigen/Core>
#include <vector>
#include <Eigen/Dense>
#include <libint2.hpp>
#include <tensor.h>
#include "integrals.h"

struct SCFResult {
    double E0;
    Matrix CA;
    Matrix CB;
    Vector epsA;
    Vector epsB;
};

class WaveFunction {
private:
    libint2::BasisSet obs;
    std::vector<libint2::Atom> atoms;
    int n_docc{};
    int n_alpha{};
    int n_beta{};
    int nbf;

    Matrix S, T, V;
    Tensor eri_tensor;
    std::vector<double> eri_vector;
    ERIScheme eri_scheme_{ERIScheme::Tensor};

    void loader(const std::string xyz, const std::string basis,
                const int mult, const int charge);

public:
    explicit WaveFunction(const std::string xyz, const std::string basis, const int mult, const int charge) {
        loader(xyz, basis, mult, charge);
    }

    void get_integrals(std::string eri_scheme_str = "tensor");
    double get_Vnn();
    double get_energy(const std::string reference, const std::string method,
                      const double econv, const int max_iter, const int verbose);

    // Read-only accessors for tests: expose the AO integrals and electron
    // counts computed by loader()/get_integrals() without duplicating the
    // libint2 setup logic in test fixtures.
    const Matrix& get_S() const { return S; }
    const Matrix& get_T() const { return T; }
    const Matrix& get_V() const { return V; }
    const std::vector<double>& get_eri_vector() const { return eri_vector; }
    const Tensor& get_eri_tensor() const { return eri_tensor; }
    const std::vector<libint2::Atom>& get_atoms() const { return atoms; }
    int get_n_docc() const { return n_docc; }
    int get_n_alpha() const { return n_alpha; }
    int get_n_beta() const { return n_beta; }
};

#endif