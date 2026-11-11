/*
 * spline.h
 *
 * simple cubic spline interpolation library without external
 * dependencies
 *
 * ---------------------------------------------------------------------
 * Copyright (C) 2011, 2014 Tino Kluge (ttk448 at gmail.com)
 *
 *  This program is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License
 *  as published by the Free Software Foundation; either version 2
 *  of the License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * ---------------------------------------------------------------------
 *
 */

#ifndef TK_SPLINE_H
#define TK_SPLINE_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <type_traits>
#include <utility>
#include <vector>
#include "vec2d.h"

// unnamed namespace only because the implementation is in this
// header file, and we don't want to export symbols to the obj files

// band matrix solver
class band_matrix final {
 private:
  double *upper_;
  double *lower_;
  int n_upper_;
  int n_lower_;
  int dim_;
  band_matrix(const band_matrix &) = delete;
  void operator=(const band_matrix &) = delete;

 public:
  band_matrix()
      : upper_(nullptr),
        lower_(nullptr),
        n_upper_(0),
        n_lower_(0),
        dim_(0) {}                         // constructor
  band_matrix(int dim, int n_u, int n_l);  // constructor
  ~band_matrix();
  void init(int dim, int n_u, int n_l);       // init with dim,n_u,n_l
  int dim() const { return dim_; };           // matrix dimensio n
  int num_upper() const { return n_upper_; }  // m_upper.size() - 1; }
  int num_lower() const { return n_lower_; }  // m_lower.size() - 1; }
  // access operator
  double &operator()(int i, int j);       // write
  double operator()(int i, int j) const;  // read
  // we can store an additional diogonal (in m_lower)
  double &saved_diag(int i);
  double saved_diag(int i) const;
  void lu_decompose();
  int index(int i, int j) const { return i * dim_ + j; }
  std::vector<double> r_solve(const std::vector<double> &b) const;
  std::vector<double> l_solve(const std::vector<double> &b) const;
  std::vector<double> lu_solve(const std::vector<double> &b,
                               bool is_lu_decomposed = false);
};

const int DISCRETIZATION = 50;
// spline interpolation
class spline {
    // 用之前需要把相邻很近的点删除掉，避免求解时失败
public:
    // first_deriv: clamped spline f'(x_0) = A, f'(x_n) = B
    // second_deriv: natural spline f''(x_0) = f''(x_n) = 0
    // details in https://zhuanlan.zhihu.com/p/62860859
    enum bd_type { first_deriv = 1, second_deriv = 2 };

private:
    std::vector<double> m_x, m_y;  // x,y coordinates of points
    // interpolation parameters
    // f(x) = a*(x-x_i)^3 + b*(x-x_i)^2 + c*(x-x_i) + y_i
    std::vector<double> m_a, m_b, m_c;  // spline coefficients
    double m_b0, m_c0;                  // for left extrapol
    bd_type m_left, m_right;
    double m_left_value, m_right_value;
    bool m_force_linear_extrapolation;
    double cached_inverse_;

public:
    // set default boundary condition to be zero curvature at both ends
    spline()
      : m_left(second_deriv),
        m_right(second_deriv),
        m_left_value(0.0),
        m_right_value(0.0),
        m_force_linear_extrapolation(false),
        cached_inverse_(0) {
    ;
    }

    spline(const spline &) = delete;
    void operator=(const spline &) = delete;
    spline(spline &&rhs)
      : m_x(std::move(rhs.m_x)),
        m_y(std::move(rhs.m_y)),
        m_a(std::move(rhs.m_a)),
        m_b(std::move(rhs.m_b)),
        m_c(std::move(rhs.m_c)),
        m_b0(rhs.m_b0),
        m_c0(rhs.m_c0),
        m_left(rhs.m_left),
        m_right(rhs.m_right),
        m_left_value(rhs.m_left_value),
        m_force_linear_extrapolation(rhs.m_force_linear_extrapolation),
        cached_inverse_(rhs.cached_inverse_) {}
    spline &operator=(spline &&rhs) {
    if (this == &rhs) {
      return *this;
    }
    this->~spline();
    new (this) spline(std::move(rhs));
    return *this;
    }

    // optional, but if called it has to come be before set_points()
    void set_boundary(bd_type left, double left_value, bd_type right,
                    double right_value,
                    bool force_linear_extrapolation = false);

    void set_points(const std::vector<double> &x, const std::vector<double> &y,
                  bool cubic_spline = true);
    void set_points(const std::vector<Vec2d>& vec, bool cubic_spline = true);
    double extrapolate(double x) const;

    double calc_old(const std::vector<double> &m_x,
                  const std::vector<double> &m_y, double x) const;
    double calc(double x) const;

    template <int order>
    double deriv(double x) const;

private:
    void set_points_old(const std::vector<double> &x,
                  const std::vector<double> &y, bool cubic_spline = true);
};

namespace detail {

template <int N, bool is_large>
struct static_mul_helper;

template <int N>
struct static_mul_helper<N, true> {
  template <typename T>
  inline T operator()(T x) const {
    static_assert(std::is_arithmetic<T>::value, "");
    return x * N;
  }
};

template <>
struct static_mul_helper<0, false> {
  template <typename T>
  inline T operator()(T x) const {
    static_assert(std::is_arithmetic<T>::value, "");
    return 0;
  }
};

template <int N>
struct static_mul_helper<N, false> {
  template <typename T>
  inline T operator()(T x) const {
    static_assert(std::is_arithmetic<T>::value, "");
    return x + static_mul_helper<N - 1, false>()(x);
  }
};

}  // namespace detail

template <int N>
struct static_mul {
  template <typename T>
  inline T operator()(T x) const {
    return detail::static_mul_helper<N, (N >= 0 && N <= 4)>()(x);
  }
};

__attribute__((always_inline)) inline double spline::extrapolate(
    double x) const {
  size_t n = m_x.size();
  // find the closest point m_x[idx] < x, idx=0 even if x<m_x[0]
  if (x < m_x[0]) {
    const double h = x - m_x[0];
    // extrapolation to the left
    return (m_b0 * h + m_c0) * h + m_y[0];
  } else if (x > m_x[n - 1]) {
    // extrapolation to the right
    const int idx = std::floor((x - m_x[0]) * cached_inverse_);
    const double h = x - m_x[idx];
    return (m_b[n - 1] * h + m_c[n - 1]) * h + m_y[n - 1];
  } else {
    const int idx = std::floor((x - m_x[0]) * cached_inverse_);
    const double h = x - m_x[idx];
    // interpolation
    return ((m_a[idx] * h + m_b[idx]) * h + m_c[idx]) * h + m_y[idx];
  }
}

template <int order>
__attribute__((always_inline)) inline double spline::deriv(double x) const {
  static_assert(order > 0, "");
  const size_t n = m_x.size();
  // find the closest point m_x[idx] < x, idx=0 even if x<m_x[0]
  if (x < m_x[0]) {
    double h = x - m_x[0];
    // extrapolation to the left
    // compiler will remove dead branches
    if (order == 1) {
      return static_mul<2>()(m_b0 * h) + m_c0;
    } else if (order == 2) {
      return static_mul<2>()(m_b0);
    }
  } else if (x > m_x[n - 1]) {
    const int idx = std::floor((x - m_x[0]) * cached_inverse_);
    double h = x - m_x[idx];
    // extrapolation to the right
    // compiler will remove dead branches
    if (order == 1) {
      return static_mul<2>()(m_b[n - 1] * h) + m_c[n - 1];
    } else if (order == 2) {
      return static_mul<2>()(m_b[n - 1]);
    }
  } else {
    // interpolation
    const int idx = std::floor((x - m_x[0]) * cached_inverse_);
    double h = x - m_x[idx];
    // compiler will remove dead branches
    if (order == 1) {
      return (static_mul<3>()(m_a[idx] * h) + static_mul<2>()(m_b[idx])) * h +
             m_c[idx];
    } else if (order == 2) {
      return static_mul<6>()(m_a[idx] * h) + static_mul<2>()(m_b[idx]);
    } else if (order == 3) {
      return static_mul<6>()(m_a[idx]);
    }
  }
  return 0.0;
}


#endif /* TK_SPLINE_H */
