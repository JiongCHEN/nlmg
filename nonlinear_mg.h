#ifndef NONLINEAR_MG_H
#define NONLINEAR_MG_H

#include <memory>
#include <Eigen/Sparse>

namespace nlmg {

class source_func
{
public:
    virtual double operator ()(const double x, const double y) const = 0;
};

class f1 : public source_func
{
public:
    f1(const double gamma) : gamma_(gamma) {}
    double operator ()(const double x, const double y) const {
        const double x_xx = x - x*x;
        const double y_yy = y - y*y;
        return 2*(x_xx + y_yy) + gamma_*x_xx*y_yy*std::exp(x_xx*y_yy);
    }
private:
    const double gamma_;
};

class f2 : public source_func
{
public:
    f2(const double gamma) : gamma_(gamma) {}
    double operator ()(const double x, const double y) const {
        const double xx_xxx = x*x - x*x*x;
        const double sin3piy = std::sin(3*M_PI*y);
        return ((9*M_PI*M_PI+gamma_*std::exp(xx_xxx*sin3piy))*xx_xxx+6*x-2)*sin3piy;
    }
private:
    const double gamma_;
};

/// -\triangle u + \gamma u * e^u = f
class grid_func
{
public:
    grid_func(const size_t resolution, const double gamma);
    virtual size_t nx() const;
    virtual size_t nf() const;
    virtual int eval_val(const double *u, double *val) const;
    virtual int eval_jac(const double *u, Eigen::SparseMatrix<double> *jac) const;
private:
    const size_t N_;
    const double h_;
    const double gamma_;
    std::shared_ptr<source_func> src_;
    Eigen::VectorXd f_;
};

class nlmg_solver
{
public:
    struct level {
        level();
    };
};

}

#endif
