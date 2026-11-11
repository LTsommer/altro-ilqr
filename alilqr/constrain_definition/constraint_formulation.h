//
// Created by LTsommer on 2024/6/20.
//

#ifndef ALILQR_CONSTRAINT_FORMULATION_H
#define ALILQR_CONSTRAINT_FORMULATION_H


#include "../ilqr_system_declaration.hpp"
#include "../cost_function/cost_function.hpp"
#include "../cost_function/cost_term_config.hpp"


/*
  f(g(x, u)), where g(x, u) > 0 represent the constraint
 */
class ConstraintPenaltyCostFunc {
public:
    ConstraintPenaltyCostFunc() = default;
    virtual ~ConstraintPenaltyCostFunc() = default;

    /**
     * @param x The LHS in constraint x > 0
     * @return the cost Evaluate f(x)
     */
    virtual double Evaluate(const double &x) const = 0;

    /**
     * @param x The LHS in constraint x > 0
     * @param res d(f(x))/dx: the computed gradient
     */
    virtual void Gradient(const double &x, double &val) const = 0;

    /**
     * @param x The LHS in constraint x > 0
     * @param res d^2(f(x))/d^2x: the computed hessian
     */
    virtual void Hessian(const double &x, double &val) const = 0;
};

/*
  f(x) = x, if x < 0
         0, otherwise
  where x is in R
 */
class OneSideCost final : public ConstraintPenaltyCostFunc {
public:
    /**
     * @param x The LHS in constraint x > 0
     * @return the cost Evaluate f(x)
     */

    double Evaluate(const double &x) const override{
        return x > 0.0 ? 0.0 : x;
    }
    /**
     * @param x The LHS in constraint x > 0
     *
     * @param res d(f(x))/dx: the computed gradient
     */
    void Gradient(const double &x, double &val) const override {
        val = x > 0.0 ? 0.0 : 1.0;
    }

    /**
     * @param x The LHS in constraint x > 0
     * @param res d^2(f(x))/d^2x: the computed hessian
     */
    void Hessian(const double &x, double &val) const override {
        val = 0;
    }
};

/*
  f(x) = 1/2 * x^2, if x < 0
         0, otherwise
  where x is in R
 */
class OneSideQuadraticCost final: public ConstraintPenaltyCostFunc {
public:
    /**
     * @param x The LHS in constraint x > 0
     * @return the cost Evaluate f(x)
     */

    double Evaluate(const double &x) const override{
        return x > 0.0 ? 0.0 : 0.5 * sqr(x);
    }
    /**
     * @param x The LHS in constraint x > 0
     *
     * @param res d(f(x))/dx: the computed gradient
     */
    void Gradient(const double &x, double &val) const override {
        val = x > 0.0 ? 0.0 : x;
    }

    /**
     * @param x The LHS in constraint x > 0
     * @param res d^2(f(x))/d^2x: the computed hessian
     */
    void Hessian(const double &x, double &val) const override {
        val = x > 0.0 ? 0.0 : 1.0;
    }
};

class OneSideEqualityCost final : ConstraintPenaltyCostFunc {
public:
    constexpr static double kEpsilon = 1.0e-6;
    OneSideEqualityCost() = default;
    virtual ~OneSideEqualityCost() = default;

    /**
     * @param x The LHS in constraint x = 0
     * @return the cost Evaluate f(x)
     */
    double Evaluate(const double &x) const override {
        return std::fabs(x) < kEpsilon ? 0.0 : x;
    };

    /**
     * @param x The LHS in constraint x = 0
     * @param res d(f(x))/dx: the computed gradient
     */
    void Gradient(const double &x, double &val) const override {
        val = (std::fabs(x) < kEpsilon) ? 0.0 : 1.0;
    };

    /**
     * @param x The LHS in constraint x > 0
     * @param res d^2(f(x))/d^2x: the computed hessian
     */
    void Hessian(const double &x, double &val) const override {
        val = 0.0;
    };
};




#endif //ALILQR_CONSTRAINT_FORMULATION_H
