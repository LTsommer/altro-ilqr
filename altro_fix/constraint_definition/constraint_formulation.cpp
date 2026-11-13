//
// Created by LTsommer on 2024/6/20.
//

#include "constraint_formulation.h"

void OneSideCost::Hessian(const double &x, double &val) const {
    val = 0;
}

double OneSideQuadraticCost::Evaluate(const double &x) const {
    return x < kInequalityTol ? 0.0 : 0.5 * sqr(x);
}

void OneSideQuadraticCost::Gradient(const double &x, double &val) const {
    val = x < kInequalityTol ? 0.0 : x;
}

void OneSideQuadraticCost::Hessian(const double &x, double &val) const {
    val = x < kInequalityTol ? 0.0 : 1.0;
}

double OneSideEqualityCost::Evaluate(const double &x) const {
    return std::fabs(x) < kEqualityTol ? 0.0 : x;
}

void OneSideEqualityCost::Gradient(const double &x, double &val) const {
    val = (std::fabs(x) < kEqualityTol) ? 0.0 : 1.0;
}

void OneSideEqualityCost::Hessian(const double &x, double &val) const {
    val = 0.0;
}
