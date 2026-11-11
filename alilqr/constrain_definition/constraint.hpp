//
// Created by LTsommer on 2024/6/20.
//

#ifndef ALILQR_CONSTRAINT_HPP
#define ALILQR_CONSTRAINT_HPP

#include "../cost_function/cost_function.hpp"
#include <cmath>
#include <iostream>
#include <memory>
#include <type_traits>

class Equality {
public:
    static constexpr double kDefaultTol = 1e-2;
    Equality() = delete;
    virtual ~Equality() = default;

    using Name = Equality;

    static void UpdateDual(const double c_val, const double penalty, double &lambda) {
        lambda += penalty * c_val;
    }

    static void UpdatePenalty(const double c_val, const double lambda,
                              const double penalty_factor, double &penalty) {
        if (std::fabs(lambda) < kDefaultTol and c_val < 0.0)
            penalty = 0.0;
        else penalty *= penalty_factor;
    }
};

class Inequality {
public:
    Inequality() = delete;
    virtual ~Inequality() = default;

    using Name = Inequality;

    static void UpdateDual(const double c_val, const double penalty, double &lambda) {
        lambda = std::max(0.0, lambda + penalty * c_val);
    }

    static void UpdatePenalty(const double c_val, const double lambda,
                              const double penalty_factor, double &penalty) {
        penalty *= penalty_factor;
    }
};

template<typename T, unsigned M, unsigned N, class ConsType>
class Constraint : public FunctionBase<T, M, N> {
public:
    OCP_VARIABLES(T, M, N)
    Constraint() = default;

    virtual std::type_index get_type_index() {
        return std::type_index(typeid(Constraint));
    }



    virtual ~Constraint() = default;

    virtual bool Evaluate(const int step, const State &x, const Control &u, double &val) const = 0;

    virtual bool Gradient(const int step, const State &x, const Control &u,
                          Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const = 0;

    virtual bool Hessian(const int step, const State &x, const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const = 0;

    bool HasHessian() { return false;}

    void GetConstraintType() const {
        if (std::is_same_v<ConsType, Equality>) {
            std::cout << "Equality Constraint\n";
        } else if (std::is_same_v<ConsType, Inequality>) {
            std::cout << "Inequality Constraint\n";
        } else {
            std::cout << "Unknown Constraint Type\n";
        }
    }

    void SetHorizon(const int horizon) {horizon_ = horizon;}

    const int Horizon() const {return horizon_;}

protected:
    int horizon_;
};

template<typename T, unsigned M, unsigned N, class ConsType>
using ConstraintPtr = std::unique_ptr<Constraint<T, M, N, ConsType>>;


#endif //ALILQR_CONSTRAINT_HPP
