#include <iostream>
#include <boost/property_tree/ptree.hpp>
#include <zjucad/ptree/ptree.h>

#include "nonlinear_mg.h"

using namespace std;
using namespace Eigen;
using namespace nlmg;
using boost::property_tree::ptree;

#define CALL_SUB_PROG(prog)                       \
    int prog(ptree &pt);                          \
    if ( pt.get<string>("prog.value") == #prog )  \
        return prog(pt);

#define INDEX(i, j, N) ((i-1)*(N-1)+(j-1))

int test_grid_func(ptree &pt) {
    /// resolution and gamma
    grid_func A(128, 1000);
    const size_t N = A.get_res();
    const double h = A.get_spa();

    VectorXd u(A.nx());
    for (size_t i = 1; i <= N-1; ++i) {
        for (size_t j = 1; j <= N-1; ++j) {
            u[INDEX(i, j, N)] = [](const double x, const double y)->double {
                return (x-x*x)*(y-y*y);
            }(i*h, j*h);
        }
    }
    VectorXd rhs(A.nf());
    A.eval_val(u.data(), rhs.data());

    shared_ptr<source_func> fun = std::make_shared<func1>(A.get_gamma());
    VectorXd f(A.nf());
    for (size_t i = 1; i <= N-1; ++i) {
        for (size_t j = 1; j <= N-1; ++j) {
            f[INDEX(i, j, N)] = (*fun)(i*h, j*h);
        }
    }
    cout << "residual norm: " << (f-rhs).norm() << endl;
    cout << "done\n";
    return 0;
}

int test_direct_solver(ptree &pt) {
    grid_func A(128, 1000);
    const size_t N = A.get_res();
    const double h = A.get_spa();

    /// exact solution
    VectorXd u(A.nx());
    for (size_t i = 1; i <= N-1; ++i) {
        for (size_t j = 1; j <= N-1; ++j) {
            u[INDEX(i, j, N)] = [](const double x, const double y)->double {
                return (x-x*x)*(y-y*y);
            }(i*h, j*h);
        }
    }
    /// rhs
    shared_ptr<source_func> fun = std::make_shared<func1>(A.get_gamma());
    VectorXd f(A.nf());
    for (size_t i = 1; i <= N-1; ++i) {
        for (size_t j = 1; j <= N-1; ++j) {
            f[INDEX(i, j, N)] = (*fun)(i*h, j*h);
        }
    }
    /// approximation
    srand(time(NULL));
    VectorXd x = VectorXd::Random(A.nx());
    x += 5*VectorXd::Random(A.nx());
    A.solve(&x[0], &f[0], 1000);

    cout << "error norm: " << (u-x).norm() << endl;
    cout << "done\n";
    return 0;
}

int test_smooth(ptree &pt) {
    grid_func A(128, 1000);
    const size_t N = A.get_res();
    const double h = A.get_spa();

    /// exact solution
    VectorXd u(A.nx());
    for (size_t i = 1; i <= N-1; ++i) {
        for (size_t j = 1; j <= N-1; ++j) {
            u[INDEX(i, j, N)] = [](const double x, const double y)->double {
                return (x-x*x)*(y-y*y);
            }(i*h, j*h);
        }
    }
    /// rhs
    shared_ptr<source_func> fun = std::make_shared<func1>(A.get_gamma());
    VectorXd f(A.nf());
    for (size_t i = 1; i <= N-1; ++i) {
        for (size_t j = 1; j <= N-1; ++j) {
            f[INDEX(i, j, N)] = (*fun)(i*h, j*h);
        }
    }
    /// approximation
    srand(time(NULL));
    VectorXd x = VectorXd::Random(A.nx());
    x += 5*VectorXd::Random(A.nx());
    A.smooth(&x[0], &f[0], 1000);

    cout << "error norm: " << (u-x).norm() << endl;
    cout << "done\n";
    return 0;
}

int test_levels(ptree &pt) {
    ptree opts;
    opts.put("level_number", 3);
    opts.put("inner_iters", 1);
    opts.put("outer_iters", 1);
    opts.put("prev_smooth", 2);
    opts.put("post_smooth", 2);
    opts.put("tolerance", 1e-8);
    opts.put("gamma", 1e3);

    nlmg_solver sol(opts);
    sol.build_levels(128);

    cout << "done\n";
    return 0;
}

int main(int argc, char *argv[])
{
    ptree pt;
    try {
        zjucad::read_cmdline(argc, argv, pt);
        CALL_SUB_PROG(test_grid_func);
        CALL_SUB_PROG(test_direct_solver);
        CALL_SUB_PROG(test_smooth);
        CALL_SUB_PROG(test_levels);
    } catch (const boost::property_tree::ptree_error &e) {
        cerr << "Usage: " << endl;
        zjucad::show_usage_info(std::cerr, pt);
    } catch (const std::exception &e) {
        cerr << "# " << e.what() << endl;
    }
    return 0;
}
