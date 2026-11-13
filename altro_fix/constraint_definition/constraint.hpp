//
// Created by LTsommer on 2024/6/20.
//

#ifndef ALILQR_CONSTRAINT_HPP
#define ALILQR_CONSTRAINT_HPP

#include "../cost_function/cost_function.hpp"
#include "../cost_function/cost_term_config.hpp"
#include <cmath>
#include <iostream>
#include <memory>
#include <type_traits>

class Equality {
public:
    static constexpr double kDefaultTol = 1e-8;
    Equality() = delete;
    virtual ~Equality() = default;

    using Name = Equality;

    // An equality constraint c = 0 is always active.
    static bool IsActive(const double c_val, const double lambda);

    static void UpdateDual(const double c_val, const double penalty, double &lambda);
};

class Inequality {
public:
    Inequality() = delete;
    virtual ~Inequality() = default;

    using Name = Inequality;

    // For c <= 0, the penalty term only applies when the constraint is violated
    // or its multiplier is still positive (the I_mu mask in AL-iLQR).
    static bool IsActive(const double c_val, const double lambda);

    static void UpdateDual(const double c_val, const double penalty, double &lambda);
};

// One smooth sub-constraint g(x, u) <= 0 and its gradient.
template<typename T, unsigned M, unsigned N>
struct RawConstraintRow {
    double g = 0.0;
    Eigen::Matrix<T, M, 1> gx = Eigen::Matrix<T, M, 1>::Zero();
    Eigen::Matrix<T, N, 1> gu = Eigen::Matrix<T, N, 1>::Zero();
};

template<typename T, unsigned M, unsigned N>
using RawConstraintRows = std::vector<RawConstraintRow<T, M, N>, Eigen::aligned_allocator<RawConstraintRow<T, M, N>>>;

// Appends the value and gradient of a sub-constraint (anything with Evaluate / Gradient).
template<typename T, unsigned M, unsigned N, class Sub>
void AppendRawConstraint(const Sub &sub,
                         const int step,
                         const Eigen::Matrix<T, M, 1> &x,
                         const Eigen::Matrix<T, N, 1> &u,
                         RawConstraintRows<T, M, N> &out) {
    RawConstraintRow<T, M, N> row;
    if (!sub.Evaluate(step, x, u, row.g)) {
        return;
    }
    sub.Gradient(step, x, u, row.gx, row.gu);
    out.push_back(row);
}

template<typename T, unsigned M, unsigned N, class ConsType>
class Constraint : public FunctionBase<T, M, N> {
public:
    OCP_VARIABLES(T, M, N)
    Constraint() = default;

    virtual std::type_index GetTypeIndex() const override { return std::type_index(typeid(Constraint)); }

    virtual ~Constraint() = default;

    virtual void UpdateConfig(const CostTermConfig &config) { return; }

    virtual std::string GetName() const override = 0;

    virtual bool Evaluate(const int step, const State &x, const Control &u, double &val) const override = 0;

    virtual bool Gradient(const int step,
                          const State &x,
                          const Control &u,
                          Eigen::Ref<VecX> lx,
                          Eigen::Ref<VecU> lu) const override = 0;

    virtual bool Hessian(const int step,
                         const State &x,
                         const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const override = 0;

    bool HasHessian() { return false; }

    // The smooth sub-constraints g_i(x, u) <= 0 this constraint is built from.
    // Composite constraints of the form w_lb * max(g_lb, 0) + w_ub * max(g_ub, 0) have a zero
    // gradient on the feasible side, so ALTRO polishing needs the raw g_i to also hold
    // nearly-active constraints in place. Returns false if the constraint does not provide them.
    virtual bool RawConstraints(const int step,
                                const State &x,
                                const Control &u,
                                RawConstraintRows<T, M, N> &out) const {
        return false;
    }

    void GetConstraintType() const {
        if (std::is_same_v<ConsType, Equality>) {
            std::cout << "Equality Constraint\n";
        } else if (std::is_same_v<ConsType, Inequality>) {
            std::cout << "Inequality Constraint\n";
        } else {
            std::cout << "Unknown Constraint Type\n";
        }
    }

    void SetHorizon(const int horizon) { horizon_ = horizon; }

    const int Horizon() const { return horizon_; }

protected:
    int horizon_;
};

template<typename T, unsigned M, unsigned N, class ConsType>
using ConstraintPtr = std::unique_ptr<Constraint<T, M, N, ConsType>>;

#endif //ALILQR_CONSTRAINT_HPP
