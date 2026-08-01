#include <vector>
#include <Eigen/Dense>
#include <libint2.hpp>
#include <utils.h>
#include "build_JK.h"


JK build_JK (const std::vector<double> &ERI, const Eigen::MatrixXd &D) {
	const int nbf = D.rows();
    
	Eigen::MatrixXd J(nbf, nbf);
	Eigen::MatrixXd K(nbf, nbf);

    J.setZero();
    K.setZero();

	for (int p=0; p != nbf; ++p) {
	for (int q=0; q != nbf; ++q) {
	for (int r=0; r != nbf; ++r) {
	for (int s=0; s != nbf; ++s) {
        J(p, q) += ERI[eri_index(p, q, r, s)] * D(r, s);
        K(p, q) += ERI[eri_index(p, r, q, s)] * D(r, s);
	}
	}
	}
	}

	return {J, K};
}