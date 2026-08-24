#include "doctest.h"
#include <cblas.h>

TEST_CASE("OpenBLAS CBLAS include and link") {
    const double left[] = {1.0, 2.0, 3.0, 4.0};
    const double right[] = {5.0, 6.0, 7.0, 8.0};
    double result[4] = {};

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                2, 2, 2, 1.0, left, 2, right, 2, 0.0, result, 2);

    CHECK(result[0] == 19.0);
    CHECK(result[1] == 22.0);
    CHECK(result[2] == 43.0);
    CHECK(result[3] == 50.0);
}