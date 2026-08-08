#ifndef BUILD_JK
#define BUILD_JK

#include <vector>
#include <Eigen/Dense>
#include <libint2.hpp>

struct JK {Eigen::MatrixXd J; Eigen::MatrixXd K;};
JK build_JK(const std::vector<double>& eri_vector, const Eigen::MatrixXd& D);

#endif