//
// Created by LTsommer on 2024/6/13.
//

#ifndef CILQR_CALCULUS_H
#define CILQR_CALCULUS_H

#include "frenet_coordinate/spline.h"
#include <iostream>
#include <type_traits>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <algorithm>
#include <array>

using Eigen::Matrix;

template<typename T>
struct function {
    virtual T operator()(const T x) const = 0;
};

template<typename T>
struct demo final : function<T> {
    T operator()(const T x) const override { return static_cast<T>(1) / (1 + std::sqrt(x)); }
};

// Nodes and weights of the 10-point Gauss-Legendre rule on [-1, 1].
namespace gauss_legendre {
inline constexpr std::array<double, 10> kNodes{-0.9739065285171717,
                                               -0.8650633666889845,
                                               -0.6794095682990244,
                                               -0.4333953941292472,
                                               -0.1488743389816312,
                                               0.1488743389816312,
                                               0.4333953941292472,
                                               0.6794095682990244,
                                               0.8650633666889845,
                                               0.9739065285171717};
inline constexpr std::array<double, 10> kWeights{0.0666713443086881,
                                                 0.1494513491505806,
                                                 0.2190863625159820,
                                                 0.2692667193099963,
                                                 0.2955242247147529,
                                                 0.2955242247147529,
                                                 0.2692667193099963,
                                                 0.2190863625159820,
                                                 0.1494513491505806,
                                                 0.0666713443086881};
} // namespace gauss_legendre

template<template<typename> class __func,
         typename T,
         typename std::enable_if<std::is_base_of_v<function<T>, __func<T>>, int>::type = 0>
inline T GaussLegendreIntegrate(const T lb, const T ub) {
    Eigen::Matrix<T, 10, 1> weights;
    weights << 0.0666713443086881, 0.1494513491505806, 0.2190863625159820, 0.2692667193099963, 0.2955242247147529,
            0.2955242247147529, 0.2692667193099963, 0.2190863625159820, 0.1494513491505806, 0.0666713443086881;
    // Fixed-size arrays only: nothing here may touch the heap.
    const std::array<double, 10> &x = gauss_legendre::kNodes;
    T coef_m = (ub - lb) / 2;
    T coef_p = (ub + lb) / 2;
    std::array<T, 10> fx;
    for (size_t i = 0; i < x.size(); ++i)
        fx[i] = __func<T>()((coef_m * static_cast<T>(x[i]) + coef_p));
    Eigen::Map<Eigen::Matrix<T, 10, 1>> xi(fx.data());
    T integrates = (weights.array() * xi.array()).sum();
    return coef_m * integrates;
}

template<typename T>
struct delta_x {
    T operator()(const T theta, const T vel) const { return std::cos(theta) * vel; }
};

template<typename T>
struct delta_y {
    T operator()(const T theta, const T vel) const { return std::sin(theta) * vel; }
};

template<template<typename> class __func, typename T>
inline T GaussLegendreIntegrate(const T l_theta, const T u_theta, const T l_vel, const T u_vel, const T delta_t) {
    constexpr static int knots = 10;
    Eigen::Matrix<T, knots, 1> weights;
    weights << 0.0666713443086881, 0.1494513491505806, 0.2190863625159820, 0.2692667193099963, 0.2955242247147529,
            0.2955242247147529, 0.2692667193099963, 0.2190863625159820, 0.1494513491505806, 0.0666713443086881;
    // Fixed-size arrays only: nothing here may touch the heap.
    const std::array<double, 10> &x = gauss_legendre::kNodes;
    T coef_m = delta_t / 2.0;
    T coef_p = delta_t / 2.0;
    T t_rate = (u_theta - l_theta) / delta_t;
    T v_rate = (u_vel - l_vel) / delta_t;
    std::array<T, knots> fx;
    for (size_t i = 0; i < x.size(); ++i) {
        T dt = static_cast<T>(x[i]) * coef_m + coef_p;
        T theta = l_theta + t_rate * dt;
        T vel = l_vel + v_rate * dt;
        fx[i] = __func<T>()(theta, vel);
    }
    Eigen::Map<Matrix<T, 10, 1>> xi(fx.data());
    T integrates = (weights.array() * xi.array()).sum();
    return coef_m * integrates;
}

// Position increment of a step along which heading and speed vary linearly in time:
//   [dx, dy] = int_0^dt [cos(theta(t)), sin(theta(t))] * v(t) dt   (10-point Gauss-Legendre).
// Same quadrature as GaussLegendreIntegrate<delta_x / delta_y>, evaluated in the frame of the
// midpoint heading: there the nodes pair up symmetrically, the odd terms cancel, and only 5
// cos + 5 sin of the half nodes plus one sin / cos of the midpoint heading are needed, instead
// of 10 sin + 10 cos. Equal to the unpaired sums up to round-off.
inline Eigen::Vector2d IntegratePlanarPosition(
        double theta0, double theta1, double velocity0, double velocity1, double dt) {
    const double half_dt = dt / 2.0;
    const double half_theta_change = ((theta1 - theta0) / dt) * half_dt;
    const double half_velocity_change = ((velocity1 - velocity0) / dt) * half_dt;
    const double mid_heading = theta0 + half_theta_change;
    const double mid_velocity = velocity0 + half_velocity_change;
    double weighted_cos = 0.0;
    double weighted_node_sin = 0.0;
    for (size_t i = gauss_legendre::kNodes.size() / 2; i < gauss_legendre::kNodes.size(); ++i) {
        const double node = gauss_legendre::kNodes[i];
        const double angle = node * half_theta_change;
        const double weight = gauss_legendre::kWeights[i];
        weighted_cos += weight * std::cos(angle);
        weighted_node_sin += weight * node * std::sin(angle);
    }
    // Even terms use the sum of the paired velocities, odd terms use their difference.
    const double longitudinal = dt * mid_velocity * weighted_cos;
    const double lateral = dt * half_velocity_change * weighted_node_sin;
    const double cos_heading = std::cos(mid_heading);
    const double sin_heading = std::sin(mid_heading);
    return {cos_heading * longitudinal - sin_heading * lateral, sin_heading * longitudinal + cos_heading * lateral};
}

//template<template<typename> class __func, typename T>
//inline T GaussLegendreIntegrate(const T l_theta, const T u_theta,
//                                const T l_vel, const T u_vel,
//                                const T delta_t) {
//    constexpr static int knots = 10;
//    Eigen::Matrix<T, knots, 1> weights;
//    weights <<  0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
//                0.2692667193099963, 0.2955242247147529, 0.2955242247147529,
//                0.2692667193099963, 0.2190863625159820, 0.1494513491505806,
//                0.0666713443086881;
//    Eigen::Matrix<T, knots, 1> x;
//    x << -0.9739065285171717, -0.8650633666889845,
//         -0.6794095682990244, -0.4333953941292472,
//         -0.1488743389816312, 0.1488743389816312,
//         0.4333953941292472, 0.6794095682990244,
//         0.8650633666889845, 0.9739065285171717;
//    T coef_m = delta_t / 2.0;
//    T coef_p = delta_t / 2.0;
//    Eigen::Matrix<T, knots, 1> Coef_p;
//    Coef_p.setConstant(coef_p);
//    Eigen::Matrix<T, knots, 1> dt = coef_m * x + Coef_p;
//    Eigen::Matrix<T, knots, 1> LB_T, LB_V;
//    LB_T.setConstant(l_theta);
//    LB_V.setConstant(l_vel);
//    T t_rate = (u_theta - l_theta) / delta_t;
//    T v_rate = (u_vel - l_vel) / delta_t;
//    Eigen::Matrix<T, knots, 1> Theta = LB_T + t_rate * dt;
//    Eigen::Matrix<T, knots, 1> Vel = LB_V + v_rate * dt;
//    std::vector<T> fx(knots, (T)0.0);
//    int row = 0;
//    std::for_each(std::begin(fx), std::end(fx), [&](T &tmp_f){
//        tmp_f = __func<T>()(Theta(row), Vel(row));
//        ++row;
//    });
//    Eigen::Map<Matrix<T, 10, 1>> xi(fx.data());
//    T integrates = (weights.array() * xi.array()).sum();
//    return coef_m * integrates;
//}

//template<template<typename> class __func, typename T>
//inline T GaussLegendreIntegrate(const spline &sy,
//                                const spline &ts,
//                                const spline &tv,
//                                const T t0,
//                                const T delta_T) {
//    Eigen::Matrix<T, 10, 1> weights;
//    weights <<  0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
//                0.2692667193099963, 0.2955242247147529, 0.2955242247147529,
//                0.2692667193099963, 0.2190863625159820, 0.1494513491505806,
//                0.0666713443086881;
//    std::vector<T> x{-0.9739065285171717, -0.8650633666889845,
//                     -0.6794095682990244, -0.4333953941292472,
//                     -0.1488743389816312, 0.1488743389816312,
//                     0.4333953941292472, 0.6794095682990244,
//                     0.8650633666889845, 0.9739065285171717};
//    T coef_m = delta_T / 2;
//    T coef_p = delta_T / 2;
//    std::vector<T> fx;
//    std::for_each(std::begin(x), std::end(x), [&](const T x_){
//        T t = t0 + coef_m * x_ + coef_p;
//        T s = static_cast<T>(ts.extrapolate(t));
//        T v = static_cast<T>(tv.extrapolate(t));
//        T theta = static_cast<T>(sy.extrapolate(s));
//        fx.emplace_back(__func<T>()(theta, v));
//    });
//    Eigen::Map<Matrix<T, 10, 1>> xi(fx.data());
//    T integrates = (weights.array() * xi.array()).sum();
//    return coef_m * integrates;
//}

#endif //CILQR_CALCULUS_H
