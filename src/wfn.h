#ifndef LOADER
#define LOADER

#include <Eigen/Core>
#include <vector>
#include <Eigen/Dense>
#include <libint2.hpp>


class WaveFunction {
private:
    libint2::BasisSet obs;
    std::vector<libint2::Atom> atoms;
    int n_docc{};
    int n_alpha{};
    int n_beta{};

    Eigen::MatrixXd S; Eigen::MatrixXd T; Eigen::MatrixXd V;
    std::vector<double> eri_vector;

    void loader(const std::string xyz, const std::string basis,
                const int mult, const int charge);
    
public:
    explicit WaveFunction(const std::string xyz, const std::string basis, const int mult, const int charge) {
        loader(xyz, basis, mult, charge);
    }
    
    void get_integrals();
    double get_energy(const std::string reference, const std::string method, 
                      const double econv, const int max_iter, const int verbose);
};

#endif