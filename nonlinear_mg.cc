#include "nonlinear_mg.h"

using namespace std;
using namespace Eigen;

namespace nlmg {

#define INDEX(i, j, N) ((i-1)*(N-1)+(j-1))

grid_func::grid_func(const size_t resolution, const double gamma)
    : N_(resolution), h_(1.0/resolution), gamma_(gamma) {}

size_t grid_func::nx() const {
    return (N_-1)*(N_-1);
}

size_t grid_func::nf() const {
    return (N_-1)*(N_-1);
}

int grid_func::eval_val(const double *u, double *val) const {
#pragma omp parallel for
    for (size_t i = 1; i <= N_-1; ++i) {
        for (size_t j = 1; j <= N_-1; ++j) {
            const size_t idx = INDEX(i, j, N_);
            double fx = 4*u[idx];
            i-1 == 0  ? (fx -= 0) : (fx -= u[INDEX(i-1, j, N_)]);
            i+1 == N_ ? (fx -= 0) : (fx -= u[INDEX(i+1, j, N_)]);
            j-1 == 0  ? (fx -= 0) : (fx -= u[INDEX(i, j-1, N_)]);
            j+1 == N_ ? (fx -= 0) : (fx -= u[INDEX(i, j+1, N_)]);
            val[idx] = fx/(h_*h_) + gamma_*u[idx]*std::exp(u[idx]);
        }
    }
    return 0;
}

int grid_func::eval_jac(const double *u, Eigen::SparseMatrix<double> *jac) const {
    return 0;
}

}
