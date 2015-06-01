#include <iostream>
#include <boost/property_tree/ptree.hpp>
#include <zjucad/ptree/ptree.h>

#include "nonlinear_mg.h"

using namespace std;
using namespace Eigen;
using namespace nlmg;
using boost::property_tree::ptree;

#define CALL_SUB_PROG(prog)                     \
  int prog(ptree &pt);                          \
  if ( pt.get<string>("prog.value") == #prog )  \
  return prog(pt);

#define INDEX(i, j, N) ((i-1)*(N-1)+(j-1))

int write_image(const char *file, const Eigen::VectorXd &u) {
  const int N = static_cast<int>(std::sqrt(u.size())) + 1;
  const double min_u = u.minCoeff();
  const double max_u = u.maxCoeff();
  FILE *fp = fopen(file, "wb");
  fprintf(fp, "P6\n%d %d\n255\n", N-1, N-1);
  for (int i = 1; i <= N-1; ++i) {
    for(int j = 1; j <= N-1; ++j) {
      static unsigned char color[3];
      color[0] = ((u[INDEX(i, j, N)]-min_u)/(max_u-min_u))*255;
      color[1] = 0;
      color[2] = 0;
      fwrite(color, 1, 3, fp);
    }
  }
  fclose(fp);
  return 0;
}

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

int test_fas(ptree &pt) {
  ptree opts;
  opts.put("level_number", 7);
  opts.put("inner_iters", 1);
  opts.put("outer_iters", 100);
  opts.put("prev_smooth", 3);
  opts.put("post_smooth", 3);
  opts.put("tolerance", 1e-10);
  opts.put("gamma", 1000);

  const size_t N = 128;
  const double h = 1.0/N;

  nlmg_solver sol(opts);
  sol.build_levels(N);

  const size_t NX = sol.get_domain_dim();
  const size_t NF = sol.get_range_dim();

  /// sampling exact solution
  VectorXd u(NX);
  for (size_t i = 1; i <= N-1; ++i) {
    for (size_t j = 1; j <= N-1; ++j) {
      u[INDEX(i, j, N)] = [](const double x, const double y)->double {
        return (x-x*x)*(y-y*y);
      }(i*h, j*h);
    }
  }
  write_image("./solution.ppm", u);

  /// sampling rhs
  shared_ptr<source_func> fun = std::make_shared<func1>(opts.get<size_t>("gamma"));
  VectorXd f(NF);
  for (size_t i = 1; i <= N-1; ++i) {
    for (size_t j = 1; j <= N-1; ++j) {
      f[INDEX(i, j, N)] = (*fun)(i*h, j*h);
    }
  }
  /// solve using FAS
  VectorXd x = VectorXd::Zero(NX);
  sol.solveFAS(x, f);
  write_image("./approximation.ppm", x);
  cout << "# error norm: " << (u-x).norm() << endl;

  /// solve by Newton
  x = VectorXd::Zero(NX);
  sol.solveNewton(x, f);
  cout << "# error norm: " << (u-x).norm() << endl;

  cout << "# done\n";
  return 0;
}

int test_fmg_fas(ptree &pt) {
  ptree opts;
  opts.put("level_number", 7);
  opts.put("inner_iters", 1);
  opts.put("outer_iters", 100);
  opts.put("prev_smooth", 2);
  opts.put("post_smooth", 2);
  opts.put("tolerance", 1e-10);
  opts.put("gamma", 10000);

  const size_t N = 128;
  const double h = 1.0/N;

  nlmg_solver sol(opts);
  sol.build_levels(N);

  const size_t NX = sol.get_domain_dim();
  const size_t NF = sol.get_range_dim();

  /// sampling exact solution
  VectorXd u(NX);
  for (size_t i = 1; i <= N-1; ++i) {
    for (size_t j = 1; j <= N-1; ++j) {
      u[INDEX(i, j, N)] = [](const double x, const double y)->double {
        return (x-x*x)*(y-y*y);
      }(i*h, j*h);
    }
  }
  /// sampling rhs
  shared_ptr<source_func> fun = std::make_shared<func1>(opts.get<size_t>("gamma"));
  VectorXd f(NF);
  for (size_t i = 1; i <= N-1; ++i) {
    for (size_t j = 1; j <= N-1; ++j) {
      f[INDEX(i, j, N)] = (*fun)(i*h, j*h);
    }
  }
  /// solve
  srand(time(NULL));
  VectorXd x = VectorXd::Random(NX);
  sol.solveFMG(x, f);

  cout << "# error norm: " << (u-x).norm() << endl;
  cout << "# done\n";
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
    CALL_SUB_PROG(test_fas);
    CALL_SUB_PROG(test_fmg_fas);
  } catch (const boost::property_tree::ptree_error &e) {
    cerr << "Usage: " << endl;
    zjucad::show_usage_info(std::cerr, pt);
  } catch (const std::exception &e) {
    cerr << "# " << e.what() << endl;
  }
  return 0;
}
