#ifndef LOADER
#define LOADER

#include <vector>
#include <string>
#include <Eigen/Dense>
#include <libint2.hpp>

struct Mol {
    libint2::BasisSet obs;
    std::vector<libint2::Atom> atoms;
    int ndocc{};
};

Mol loader(const std::string& xyz, const std::string& basis);

#endif