#include "nonlinear_mg.h"

#include <iostream>
#include <iomanip>

using namespace std;
using namespace Eigen;

namespace nlmg {

#define INDEX(i, j, N) ((i-1)*(N-1)+(j-1))
#define VALID(i, j, N) ((i) > 0 && (i) < N && (j) > 0 && (j) < N)

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
            if ( i-1 > 0 )
                fx -= u[INDEX(i-1, j, N_)];
            if ( i+1 < N_ )
                fx -= u[INDEX(i+1, j, N_)];
            if ( j-1 > 0 )
                fx -= u[INDEX(i, j-1, N_)];
            if ( j+1 < N_ )
                fx -= u[INDEX(i, j+1, N_)];
            val[idx] = fx/(h_*h_) + gamma_*u[idx]*std::exp(u[idx]);
        }
    }
    return 0;
}

int grid_func::eval_jac(const double *u, Eigen::SparseMatrix<double> *jac) const {
    vector<Triplet<double>> trips;
    for (size_t i = 1; i <= N_-1; ++i) {
        for (size_t j = 1; j <= N_-1; ++j) {
            const size_t idx = INDEX(i, j, N_);
            const double dia = 4.0/(h_*h_) + gamma_*(1+u[idx])*std::exp(u[idx]);
            const double off = -1.0/(h_*h_);
            trips.push_back(Triplet<double>(idx, idx, dia));
            if ( i-1 > 0 )
                trips.push_back(Triplet<double>(idx, INDEX(i-1, j, N_), off));
            if ( i+1 < N_ )
                trips.push_back(Triplet<double>(idx, INDEX(i+1, j, N_), off));
            if ( j-1 > 0 )
                trips.push_back(Triplet<double>(idx, INDEX(i, j-1, N_), off));
            if ( j+1 < N_ )
                trips.push_back(Triplet<double>(idx, INDEX(i, j+1, N_), off));
        }
    }
    jac->resize(nf(), nx());
    jac->reserve(trips.size());
    jac->setFromTriplets(trips.begin(), trips.end());
    return 0;
}

int grid_func::smooth(double *u, const double *rhs, const size_t times) {
    Map<VectorXd> U(u, nx());
    for (size_t iter = 1; iter <= times; ++iter) {
        VectorXd prev = U;
        for (size_t i = 1; i <= N_-1; ++i) {
            for (size_t j = 1; j <= N_-1; ++j) {
                const size_t idx = INDEX(i, j, N_);
                double fx = 4*u[idx];
                if ( i-1 > 0 )
                    fx -= u[INDEX(i-1, j, N_)];
                if ( i+1 < N_ )
                    fx -= u[INDEX(i+1, j, N_)];
                if ( j-1 > 0 )
                    fx -= u[INDEX(i, j-1, N_)];
                if ( j+1 < N_ )
                    fx -= u[INDEX(i, j+1, N_)];
                u[idx] -= (fx/(h_*h_) + gamma_*u[idx]*std::exp(u[idx]) - rhs[idx])
                        /(4.0/(h_*h_) + gamma_*(1+u[idx])*std::exp(u[idx]));
            }
        }
        cout << "relative error: " << (U-prev).norm()/prev.norm() << endl;
        if ( (U-prev).norm() <= 1e-12 * prev.norm() ) {
            cout << "\t@converged after " << iter << " iterations\n";
            break;
        }
    }
    return 0;
}

int grid_func::solve(double *u, const double *rhs, const size_t times) {
    Map<VectorXd> x(u, nx());
    Map<const VectorXd> f(rhs, nf());

    SimplicialCholesky<SparseMatrix<double>> sol;
    for (size_t iter = 1; iter <= times; ++iter) {
        /// assemble lhs
        SparseMatrix<double> J(nf(), nx());
        eval_jac(&x[0], &J);

        /// assemble rhs
        VectorXd Au(nf());
        eval_val(&x[0], &Au[0]);
        VectorXd b = f-Au;

        sol.compute(J);
        if ( sol.info() != Success ) {
            cerr << "\t@factorization failed\n";
            return __LINE__;
        }
        VectorXd dx = sol.solve(b);
        if ( sol.info() != Success ) {
            cerr << "\t@solve failed\n";
            return __LINE__;
        }
        double x_norm = x.norm();
        x += dx;
        cout << "relative error: " << dx.norm()/x_norm << endl;
        if ( dx.norm() <= 1e-12 * x_norm ) {
            cout << "\t@converged after " << iter << " iterations\n";
            break;
        }
    }
    return 0;
}
//==============================================================================
nlmg_solver::nlmg_solver()
    : nbr_levels_(3),
      nbr_inner_cycle_(1),
      nbr_outer_cycle_(10),
      nbr_prev_smooth_(3),
      nbr_post_smooth_(3),
      tolerance_(1e-8),
      gamma_(1000) {}

nlmg_solver::nlmg_solver(const boost::property_tree::ptree &pt) {
    nbr_levels_      = pt.get<size_t>("level_number");
    nbr_inner_cycle_ = pt.get<size_t>("inner_iters");
    nbr_outer_cycle_ = pt.get<size_t>("outer_iters");
    nbr_prev_smooth_ = pt.get<size_t>("prev_smooth");
    nbr_post_smooth_ = pt.get<size_t>("post_smooth");
    tolerance_       = pt.get<double>("tolerance");
    gamma_           = pt.get<double>("gamma");
}

void nlmg_solver::build_levels(const size_t fine_res) {
    ptrspmat_t P, R;
    size_t N = fine_res;
    for (size_t i = 0; i < nbr_levels_-1; ++i) {
        levels_.push_back(level(N, gamma_));
        N /= 2;
        std::tie(P, R) = coarsen(--levels_.end());
        levels_.rbegin()->P_ = P;
        levels_.rbegin()->R_ = R;
    }
    levels_.push_back(level(N, gamma_));

    /// debug
    cout << "----------------------------------------------\n";
    cout << setw(6) << "level" << setw(20) << "resolution\n";
    size_t cnt = 0;
    for (level_iterator it = levels_.begin(); it != levels_.end(); ++it) {
        cout << setw(6) << ++cnt << setw(20) << it->get_res() << endl;
    }
    cout << "----------------------------------------------\n";
}

nlmg_solver::transfer_t nlmg_solver::coarsen(level_iterator curr) {
    const size_t Nf = curr->get_res();
    const size_t Nc = Nf/2;
    const size_t dimf = (Nf-1)*(Nf-1);
    const size_t dimc = (Nc-1)*(Nc-1);
    ptrspmat_t P, R;

    /// restriction
    {
        vector<Triplet<double>> trips;
        for (size_t i = 1; i <= Nc-1; ++i) {
            for (size_t j = 1; j <= Nc-1; ++j) {
                const size_t idx = INDEX(i, j, Nc);
                trips.push_back(Triplet<double>(idx, INDEX(2*i, 2*j, Nf), 4.0/16));
                trips.push_back(Triplet<double>(idx, INDEX(2*i, 2*j-1, Nf), 2.0/16));
                trips.push_back(Triplet<double>(idx, INDEX(2*i, 2*j+1, Nf), 2.0/16));
                trips.push_back(Triplet<double>(idx, INDEX(2*i-1, 2*j, Nf), 2.0/16));
                trips.push_back(Triplet<double>(idx, INDEX(2*i+1, 2*j, Nf), 2.0/16));
                trips.push_back(Triplet<double>(idx, INDEX(2*i-1, 2*j-1, Nf), 1.0/16));
                trips.push_back(Triplet<double>(idx, INDEX(2*i-1, 2*j+1, Nf), 1.0/16));
                trips.push_back(Triplet<double>(idx, INDEX(2*i+1, 2*j-1, Nf), 1.0/16));
                trips.push_back(Triplet<double>(idx, INDEX(2*i+1, 2*j+1, Nf), 1.0/16));
            }
        }
        spmat_t res(dimc, dimf);
        res.reserve(trips.size());
        res.setFromTriplets(trips.begin(), trips.end());
        R = std::make_shared<spmat_t>(res);
    }
    /// prolongation
    {
        vector<Triplet<double>> trips;
        for (size_t i = 0; i <= Nc-1; ++i) {
            for (size_t j = 0; j <= Nc-1; ++j) {
                if ( VALID(2*i, 2*j, Nf) ) {
                    size_t idx = INDEX(2*i, 2*j, Nf);
                    if ( VALID(i, j, Nc) )
                        trips.push_back(Triplet<double>(idx, INDEX(i, j, Nc), 1.0));
                }
                if ( VALID(2*i+1, 2*j, Nf) ) {
                    size_t idx = INDEX(2*i+1, 2*j, Nf);
                    if ( VALID(i, j, Nc) )
                        trips.push_back(Triplet<double>(idx, INDEX(i, j, Nc), 0.5));
                    if ( VALID(i+1, j, Nc) )
                        trips.push_back(Triplet<double>(idx, INDEX(i+1, j, Nc), 0.5));
                }
                if ( VALID(2*i, 2*j+1, Nf) ) {
                    size_t idx = INDEX(2*i, 2*j+1, Nf);
                    if ( VALID(i, j, Nc) )
                        trips.push_back(Triplet<double>(idx, INDEX(i, j, Nc), 0.5));
                    if ( VALID(i, j+1, Nc) )
                        trips.push_back(Triplet<double>(idx, INDEX(i, j+1, Nc), 0.5));
                }
                if ( VALID(2*i+1, 2*j+1, Nf) ) {
                    size_t idx = INDEX(2*i+1, 2*j+1, Nf);
                    if ( VALID(i, j, Nc) )
                        trips.push_back(Triplet<double>(idx, INDEX(i, j, Nc), 0.25));
                    if ( VALID(i+1, j, Nc) )
                        trips.push_back(Triplet<double>(idx, INDEX(i+1, j, Nc), 0.25));
                    if ( VALID(i, j+1, Nc) )
                        trips.push_back(Triplet<double>(idx, INDEX(i, j+1, Nc), 0.25));
                    if ( VALID(i+1, j+1, Nc) )
                        trips.push_back(Triplet<double>(idx, INDEX(i+1, j+1, Nc), 0.25));
                }
            }
        }
        spmat_t pro(dimf, dimc);
        pro.reserve(trips.size());
        pro.setFromTriplets(trips.begin(), trips.end());
        P = std::make_shared<spmat_t>(pro);
    }
    return std::make_tuple(P, R);
}

}
