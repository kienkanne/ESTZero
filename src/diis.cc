#include <vector>
#include <Eigen/Dense>

Eigen::VectorXd diis(const std::vector<Eigen::MatrixXd>& err_list) {
    int size = err_list.size();

    Eigen::MatrixXd B = Eigen::MatrixXd::Constant(size + 1, size + 1, -1.0);
    B(size, size) = 0.0;

    int n = err_list[0].size();
    Eigen::MatrixXd E(size, n);

    for(int i = 0; i != size; ++i) {
        Eigen::Map<const Eigen::VectorXd> err_vec(err_list[i].data(), n); // n for vector size
        E.row(i) = err_vec;
    }

    B(Eigen::seq(0, size - 1), Eigen::seq(0, size - 1)) = E * E.transpose();

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(size + 1);
    rhs(size) = -1.0;

    Eigen::VectorXd coeffs = B.colPivHouseholderQr().solve(rhs);

    // Return only the coefficients (drop the final Lagrange multiplier element)
    return coeffs.head(size);
}