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
    int ndocc{};

    Eigen::MatrixXd S; Eigen::MatrixXd T; Eigen::MatrixXd V;
    std::vector<double> eri_vector;

    void loader(const std::string xyz, const std::string basis);
    
public:
    explicit WaveFunction(const std::string xyz, const std::string basis) {
        loader(xyz, basis);
    }
    
    void get_integrals();
    double get_energy(const std::string type, const double tol, 
                      const int max_iter, const int verbose);
};

#endif