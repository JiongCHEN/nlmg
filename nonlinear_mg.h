#ifndef NONLINEAR_MG_H
#define NONLINEAR_MG_H

#include <memory>
#include <Eigen/Sparse>
#include <boost/property_tree/ptree.hpp>

namespace nlmg {

class source_func
{
public:
    virtual double operator ()(const double x, const double y) const = 0;
};

class func1 : public source_func
{
public:
    func1(const double gamma) : gamma_(gamma) {}
    double operator ()(const double x, const double y) const {
        const double x_xx = x - x*x;
        const double y_yy = y - y*y;
        return 2*(x_xx + y_yy) + gamma_*x_xx*y_yy*std::exp(x_xx*y_yy);
    }
private:
    const double gamma_;
};

class func2 : public source_func
{
public:
    func2(const double gamma) : gamma_(gamma) {}
    double operator ()(const double x, const double y) const {
        const double xx_xxx = x*x - x*x*x;
        const double sin3piy = std::sin(3*M_PI*y);
        return ((9*M_PI*M_PI+gamma_*std::exp(xx_xxx*sin3piy))*xx_xxx+6*x-2)*sin3piy;
    }
private:
    const double gamma_;
};

/// nonlinear operator: $-\triangle u + \gamma u * e^u$
class grid_func
{
public:
    grid_func(const size_t resolution, const double gamma);
    virtual size_t nx() const;
    virtual size_t nf() const;
    virtual int eval_val(const double *u, double *val) const;
    virtual int eval_jac(const double *u, Eigen::SparseMatrix<double> *jac) const;
    virtual int smooth(double *u, const double *rhs, const size_t times);
    virtual int solve(double *u, const double *rhs, const size_t times);
    virtual size_t get_res() const { return N_; }
    virtual double get_spa() const { return h_; }
    virtual double get_gamma() const { return gamma_; }
private:
    const size_t N_;
    const double h_;
    const double gamma_;
};

class nlmg_solver
{
public:
    typedef Eigen::SparseMatrix<double> spmat_t;
    typedef Eigen::VectorXd vec_t;
    typedef std::shared_ptr<spmat_t> ptrspmat_t;
    typedef std::shared_ptr<vec_t> ptrvec_t;
    typedef std::tuple<ptrspmat_t, ptrspmat_t> transfer_t;
    struct level {
        std::shared_ptr<grid_func> A_;
        ptrvec_t u_;
        ptrvec_t f_;
        ptrspmat_t P_;
        ptrspmat_t R_;
        level(const size_t res, const double gamma) {
            A_ = std::make_shared<grid_func>(res, gamma);
            u_ = std::make_shared<vec_t>(A_->nx());
            f_ = std::make_shared<vec_t>(A_->nf());
        }
        size_t get_res() const { return A_->get_res(); }
        double get_spa() const { return A_->get_spa(); }
        size_t get_nx() const { return A_->nx(); }
        size_t get_nf() const { return A_->nf(); }
    };
    typedef std::vector<level>::const_iterator level_iterator;
    nlmg_solver();
    nlmg_solver(const boost::property_tree::ptree &pt);
    void build_levels(const size_t fine_res);
    int solveFAS(vec_t &x, const vec_t &rhs);
    int solveFMG(vec_t &x, const vec_t &rhs);
    int solveNewton(vec_t &x, const vec_t &rhs);
    size_t get_domain_dim() const { return levels_.begin()->get_nx(); }
    size_t get_range_dim() const { return levels_.begin()->get_nf(); }
private:
    transfer_t coarsen(level_iterator curr);
    void cycle(level_iterator curr, const vec_t &rhs, vec_t &x);
    void fmg_cycle(level_iterator curr, const vec_t &rhs, vec_t &x);

    size_t nbr_levels_;
    size_t nbr_inner_cycle_;
    size_t nbr_outer_cycle_;
    size_t nbr_prev_smooth_;
    size_t nbr_post_smooth_;
    double tolerance_;
    double gamma_;
    std::vector<level> levels_;
};

}

#endif
