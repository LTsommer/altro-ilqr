//
// Created by LTsommer on 2024/6/20.
//

#ifndef ALILQR_CONSTRAINT_VALUES_HPP
#define ALILQR_CONSTRAINT_VALUES_HPP

#include "constraint.hpp"
#include "../ilqr_system_declaration.hpp"
#include <iostream>

template <typename T, unsigned M, unsigned N, class ConsType>
class ConstraintValue {
public:
    static constexpr double kDefaultPenalty = 1.0;
    static constexpr double kDefaultPenaltyFactor = 10.0;
    static constexpr double kDefaultLambda = 1.0e-2;
public:
    OCP_VARIABLES(T, M, N)
    ConstraintValue() = default;
    ~ConstraintValue() = default;

    void UpdateDual() {
        ConsType::Name::UpdateDual(cons_val_, penalty_, lambda_);
    }

    void UpdatePenalty() {
        ConsType::Name::UpdatePenalty(cons_val_, lambda_, penalty_factor_, penalty_);
    }

    void SetPenalty(const double penalty) {penalty_ = penalty;}

    void SetDual(const double lambda) {lambda_ = lambda;}

    double GetLambda() const {return lambda_;}

    double GetPenalty() const {return penalty_;}

    void LoadConstraint(ConstraintPtr<T, M, N, ConsType> && cons) {
        cons_ptr_.swap(cons);
    };

    double ConsVal() { return cons_val_; }

    double AugLagVal() { return ctg_val_; }

    VecX GradientX() { return grad_x_; }

    VecU GradientU() { return grad_u_; }

    bool Evaluate(const int step, const State &x, const Control &u) {
//        cons_ptr_->GetConstraintType();
        bool success = cons_ptr_->Evaluate(step, x, u, cons_val_);
        if (success) {
            ctg_val_ = (lambda_ + 0.5 * cons_val_ * penalty_) * cons_val_;
//            violations_.emplace_back(std::fabs(cons_val_));
            return success;
        }
        ctg_val_ = 0.0;
        return false;
    }

//    double MaxViolation() {
//        auto iter = std::max_element(violations_.begin(), violations_.end());
//        double max_val = violations_[std::distance(violations_.begin(), iter)];
//        violations_.clear();
//        return max_val;
//    }

    void Gradient(const int step, const State &x, const Control &u) {
        grad_x_.setZero();
        grad_u_.setZero();
        cons_ptr_->Gradient(step, x, u, grad_x_, grad_u_);
    }

    void Hessian(const int step, const State &x, const Control &u) {
        hessian_xx_.setZero();
        hessian_uu_.setZero();
        hessian_xu_.setZero();
        cons_ptr_->Hessian(step, x, u, hessian_xx_, hessian_uu_, hessian_xu_);
    }

private:
    double lambda_;   // penalty scale
    double penalty_factor_ = kDefaultPenaltyFactor;
    double cons_val_;  // c(x)
    double ctg_val_;  // (lambda + 0.5 * penalty * c(x)) * c(x)
    double penalty_ = kDefaultPenalty;
    MatrixLXX hessian_xx_;
    MatrixLUU hessian_uu_;
    MatrixLXU hessian_xu_;
    VecX grad_x_;
    VecU grad_u_;
    ConstraintPtr<T, M, N, ConsType> cons_ptr_;
    vector<double> violations_;
};

template<typename T, unsigned M, unsigned N, class ConsType>
using ConstraintValuePtr = std::unique_ptr<ConstraintValue<T, M, N, ConsType>>;


#endif //ALILQR_CONSTRAINT_VALUES_HPP
