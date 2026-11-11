//
// Created by LTsommer on 2024/6/20.
//

#ifndef ALILQR_CONSTRAINT_FORMULATION_H
#define ALILQR_CONSTRAINT_FORMULATION_H

#include "../ilqr_system_declaration.hpp"
#include "../cost_function/cost_function.hpp"
#include "../cost_function/cost_term_config.hpp"

constexpr static double kInequalityTol = -1.0e-6;
constexpr static double kEqualityTol = 1.0e-6;

/*
  f(g(x, u)), where g(x, u) <= 0 represent the constraint
 */
class ConstraintPenaltyCostFunc {
public:
    ConstraintPenaltyCostFunc() = default;
    virtual ~ConstraintPenaltyCostFunc() = default;

    /**
     * @param x The LHS in constraint x < 0
     * @return the cost Evaluate f(x)
     */
    virtual double Evaluate(const double &x) const = 0;

    /**
     * @param x The LHS in constraint x < 0
     * @param res d(f(x))/dx: the computed gradient
     */
    virtual void Gradient(const double &x, double &val) const = 0;

    /**
     * @param x The LHS in constraint x < 0
     * @param res d^2(f(x))/d^2x: the computed hessian
     */
    virtual void Hessian(const double &x, double &val) const = 0;
};

/*
  f(x) = x, if x > 0
         0, otherwise
  where x is in R
 */
class OneSideCost final : public ConstraintPenaltyCostFunc {
public:
    /**
     * @param x The LHS in constraint x > 0
     * @return the cost Evaluate f(x)
     */

    double Evaluate(const double &x) const override { return x < kInequalityTol ? 0.0 : x; }
    /**
     * @param x The LHS in constraint x < 0
     *
     * @param res d(f(x))/dx: the computed gradient
     */
    void Gradient(const double &x, double &val) const override { val = x < kInequalityTol ? 0.0 : 1.0; }

    /**
     * @param x The LHS in constraint x > 0
     * @param res d^2(f(x))/d^2x: the computed hessian
     */
    void Hessian(const double &x, double &val) const override { val = 0; }
};

/*
  f(x) = 1/2 * x^2, if x > 0
         0, otherwise
  where x is in R
 */
class OneSideQuadraticCost final : public ConstraintPenaltyCostFunc {
public:
    /**
     * @param x The LHS in constraint x > 0
     * @return the cost Evaluate f(x)
     */

    double Evaluate(const double &x) const override { return x < kInequalityTol ? 0.0 : 0.5 * sqr(x); }
    /**
     * @param x The LHS in constraint x > 0
     *
     * @param res d(f(x))/dx: the computed gradient
     */
    void Gradient(const double &x, double &val) const override { val = x < kInequalityTol ? 0.0 : x; }

    /**
     * @param x The LHS in constraint x > 0
     * @param res d^2(f(x))/d^2x: the computed hessian
     */
    void Hessian(const double &x, double &val) const override { val = x < kInequalityTol ? 0.0 : 1.0; }
};

class OneSideEqualityCost final : ConstraintPenaltyCostFunc {
public:
    OneSideEqualityCost() = default;
    virtual ~OneSideEqualityCost() = default;

    /**
     * @param x The LHS in constraint x = 0
     * @return the cost Evaluate f(x)
     */
    double Evaluate(const double &x) const override { return std::fabs(x) < kEqualityTol ? 0.0 : x; };

    /**
     * @param x The LHS in constraint x = 0
     * @param res d(f(x))/dx: the computed gradient
     */
    void Gradient(const double &x, double &val) const override { val = (std::fabs(x) < kEqualityTol) ? 0.0 : 1.0; };

    /**
     * @param x The LHS in constraint x > 0
     * @param res d^2(f(x))/d^2x: the computed hessian
     */
    void Hessian(const double &x, double &val) const override { val = 0.0; };
};

// TODO: 抽象出一个constraint class用来处理形如 x_l <= x <= x_u 这一类的约束，
//  Evaluate, Gradient和Hessian的计算是一致的，不必每次都重写，从父类继承即可

#endif //ALILQR_CONSTRAINT_FORMULATION_H
