#ifndef LOADER
#define LOADER

#include <Eigen/Core>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <libint2.hpp>


class WaveFunction {
private:
    libint2::BasisSet obs;
    std::vector<libint2::Atom> atoms;
    int ndocc{};

    Eigen::MatrixXd S; Eigen::MatrixXd T; Eigen::MatrixXd V;
    std::vector<double> eri_vector;

    void loader(const std::string xyz, const std::string basis);
    
public:
    explicit WaveFunction(std::string xyz, std::string basis) {
        loader(xyz, basis);
    }
    
    void get_integrals();
    double get_energy(std::string type, double tol, int max_iter, int verbose);
};

#endif