//
// Small math helpers shared by the models, costs, constraints and the Frenet frame.
//

#ifndef CILQR_COMMON_MATH_UTILS_H
#define CILQR_COMMON_MATH_UTILS_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include <Eigen/Core>

#include "common/trig_table.h"

template<typename T>
__attribute__((always_inline)) inline T sqr(const T &value) {
    return value * value;
}

template<typename T>
__attribute((always_inline)) inline T cube(const T &value) {
    return sqr<T>(value) * value;
}

template<typename T>
__attribute__((always_inline)) inline T doppel(const T &value) {
    return value + value;
}

template<typename T>
__attribute__((always_inline)) inline int sgn(T &value) {
    return (T(0) < value) - (value < T(0));
}

template<typename T = Eigen::Vector2d>
__attribute__((always_inline)) inline double cross_prod(const T &w, const T &v) {
    return (w(0) * v(1) - w(1) * v(0));
}

inline constexpr double kPi = trig_table::kTwoPi / 2.0;

// Wraps an angle in the caller's supported numeric range to [-pi, pi).
inline double NormalizeAngle(const double angle) {
    if (angle >= -kPi && angle < kPi) {
        return angle;
    }
    const double principal = std::remainder(angle, trig_table::kTwoPi);
    return principal == kPi ? -kPi : principal;
}

inline double AngleDifference(double angle, double reference) {
    return NormalizeAngle(angle - reference);
}

// Piecewise-linear interpolation of (xs, ys) at x, held constant beyond the ends. xs must be
// sorted and non-empty, and ys the same size. Where two knots (almost) coincide the smaller
// value is used. If slope is given, it receives dy/dx at x (0 beyond the ends and at a
// coincident pair).
inline double InterpolatePiecewiseLinear(const std::vector<double> &xs,
                                         const std::vector<double> &ys,
                                         const double x,
                                         double *slope = nullptr) {
    assert(!xs.empty() && xs.size() == ys.size());
    double dy_dx = 0.0;
    double y;
    if (x <= xs.front()) {
        y = ys.front();
    } else if (x >= xs.back()) {
        y = ys.back();
    } else {
        const size_t i = std::upper_bound(xs.begin(), xs.end(), x) - xs.begin();
        const double dx = xs[i] - xs[i - 1];
        if (dx <= 1e-9) {
            y = std::min(ys[i - 1], ys[i]);
        } else {
            dy_dx = (ys[i] - ys[i - 1]) / dx;
            y = ys[i - 1] + dy_dx * (x - xs[i - 1]);
        }
    }
    if (slope != nullptr) {
        *slope = dy_dx;
    }
    return y;
}

#endif // CILQR_COMMON_MATH_UTILS_H
