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

	for (size_t i=0; i != nbf; ++i) {
	for (size_t j=i; j != nbf; ++j) {
	for (size_t k=i; k != nbf; ++k) {
		size_t l_min = (i == k) ? j : k;
	for (size_t l=l_min; l != nbf; ++l) {
		// Scattering scheme for J and K
		double factor {1.0};
		// Account for cases when filling the same block
		if (i == j) {factor *= 0.5;}
		if (k == l) {factor *= 0.5;}
		if ((i == k) && (j == l)) {factor *= 0.5;}

		double val {factor * ERI[eri_index(i, j, k, l)]};		

		// J scatter half as much as K
        J(i, j) += 2 * val * D(k, l);
		J(k, l) += 2 * val * D(i, j);

		K(i, l) += val * D(j, k);
        K(i, k) += val * D(j, l);
		K(j, l) += val * D(i, k);
		K(j, k) += val * D(i, l);

		// Symmetry
        J(j, i) += 2 * val * D(k, l);
        J(l, k) += 2 * val * D(i, j);

        K(k, i) += val * D(j, l);
        K(l, i) += val * D(j, k);
        K(k, j) += val * D(i, l);
        K(l, j) += val * D(i, k);
	}
	}
	}
	}

	return {J, K};
}