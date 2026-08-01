#ifndef DIIS
#define DIIS

#include <vector>
#include <Eigen/Dense>
#include <libint2.hpp>

Eigen::VectorXd diis(const std::vector<Eigen::MatrixXd>& err_list);
#endif