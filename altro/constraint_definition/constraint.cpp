//
// Created by LTsommer on 2024/6/20.
//

#include "constraint_definition/constraint.hpp"

bool Equality::IsActive(const double c_val, const double lambda) {
    return true;
}

void Equality::UpdateDual(const double c_val,
                          const double penalty,
                          double &lambda) {
    lambda += penalty * c_val;
}

bool Inequality::IsActive(const double c_val, const double lambda) {
    return c_val > 0.0 || lambda > 0.0;
}

void Inequality::UpdateDual(const double c_val,
                            const double penalty,
                            double &lambda) {
    lambda = std::max(0.0, lambda + penalty * c_val);
}
