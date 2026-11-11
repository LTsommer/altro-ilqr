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

using Eigen::Matrix;

template<typename T>
struct function {
    virtual T operator() (const T x) const = 0;
};

template<typename T>
struct demo final : function<T> {
    T operator() (const T x) const override {
        return static_cast<T>(1) / (1 + std::sqrt(x));
    }
};

template<template<typename> class __func, typename T, 
        typename std::enable_if<std::is_base_of_v<function<T>, __func<T>>, int>::type = 0>
inline T GaussLegendreIntegrate(const T lb, const T ub) {
    Eigen::Matrix<T, 10, 1> weights;
    weights << 0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
            0.2692667193099963, 0.2955242247147529, 0.2955242247147529,
            0.2692667193099963, 0.2190863625159820, 0.1494513491505806,
            0.0666713443086881;
    std::vector<T> x{-0.9739065285171717, -0.8650633666889845,
                     -0.6794095682990244, -0.4333953941292472,
                     -0.1488743389816312, 0.1488743389816312,
                     0.4333953941292472, 0.6794095682990244,
                     0.8650633666889845, 0.9739065285171717};
    T coef_m = (ub - lb) / 2;
    T coef_p = (ub + lb) / 2;
    std::vector<T> fx;
    std::for_each(std::begin(x), std::end(x), [&](const T x_){
        fx.emplace_back(__func<T>()((coef_m * x_ + coef_p)));
    });
    Eigen::Map<Eigen::Matrix<T, 10, 1>> xi(fx.data());
    T integrates = (weights.array() * xi.array()).sum();
    return coef_m * integrates;
}

template<typename T>
struct delta_x {
    T operator()(const T theta, const T vel) const {
        return std::cos(theta) * vel;
    }
};

template<typename T>
struct delta_y {
    T operator()(const T theta, const T vel) const{
        return std::sin(theta) * vel;
    }
};

template<template<typename> class __func, typename T>
inline T GaussLegendreIntegrate(const spline &sy,
                                const spline &ts,
                                const spline &tv,
                                const T t0,
                                const T delta_T) {
    Eigen::Matrix<T, 10, 1> weights;
    weights <<  0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
                0.2692667193099963, 0.2955242247147529, 0.2955242247147529,
                0.2692667193099963, 0.2190863625159820, 0.1494513491505806,
                0.0666713443086881;
    std::vector<T> x{-0.9739065285171717, -0.8650633666889845,
                     -0.6794095682990244, -0.4333953941292472,
                     -0.1488743389816312, 0.1488743389816312,
                     0.4333953941292472, 0.6794095682990244,
                     0.8650633666889845, 0.9739065285171717};
    T coef_m = delta_T / 2;
    T coef_p = delta_T / 2;
    std::vector<T> fx;
    std::for_each(std::begin(x), std::end(x), [&](const T x_){
        T t = t0 + coef_m * x_ + coef_p;
        T s = static_cast<T>(ts.extrapolate(t));
        T v = static_cast<T>(tv.extrapolate(t));
        T theta = static_cast<T>(sy.extrapolate(s));
        fx.emplace_back(__func<T>()(theta, v));
    });
    Eigen::Map<Matrix<T, 10, 1>> xi(fx.data());
    T integrates = (weights.array() * xi.array()).sum();
    return coef_m * integrates;
}

#endif //CILQR_CALCULUS_H
