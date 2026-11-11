//
// Created by LTsommer on 2024/5/30.
//

#ifndef CILQR_COST_CALC_H
#define CILQR_COST_CALC_H

#include "cost_function.hpp"
#include "cost_term_config.hpp"
#include "constraint_values.hpp"

template<typename T, unsigned int M, unsigned int N>
class CostUnion {
public:
    OCP_VARIABLES(T, M, N)
    using CostFuncs = std::vector<CostFuncPtr<T, M, N>>;

    template<class ConsType>
    using ConstraintValues = std::vector<ConstraintValuePtr<T, M, N, ConsType>>;

    static constexpr int dim = Eigen::Dynamic;

    template<int size>
    using VectorNd = Eigen::Matrix<T, size, 1>;

public:
    explicit CostUnion(const CostTermConfig & config): config_(config),
    horizon_(config.steps.size())
    {};

    CostUnion &operator=(const CostUnion&) = delete;
    CostUnion(const CostUnion&) = delete;

    virtual ~CostUnion() = default;

    void SetHorizon(const int horizon) {horizon_ = horizon;}

    double Evaluate(const int step,
                    const State &x,
                    const Control &u) const;

    void CalcCostGradient(const int step,
                          const State &x,
                          const Control &u,
                          Eigen::Ref<VecX> lx,
                          Eigen::Ref<VecU> lu) const;

    void CalcCostHessian(const int step, const State &x, const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const;

    void CalcAugLagConstraintGradient(const int step, const State &x, const Control &u,
                                      Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const;

    void CalcAugLagConstraintHessian(const int step, const State &x, const Control &u,
                                     Eigen::Ref<MatrixLXX> cxx,
                                     Eigen::Ref<MatrixLUU> cuu,
                                     Eigen::Ref<MatrixLXU> cxu) const;

    void CalcTerminalCostToGo(const State &x);

    double GetMaxViolation() const {return max_violation_;};

    void EvaluateConstraints(const States &x, const Controls &u);

    VecX GetFinalCostToGoGradient() const {return pN;}

    MatrixLXX GetFinalCostToGoHessian() const {return PN;}

    void AddCostTerm(CostFuncPtr<T, M, N> &&cost_term) {
        cost_terms_.emplace_back(std::move(cost_term));
    }

    void AddEqConstraint(ConstraintValuePtr<T, M, N, Equality> &&constraint_value) {
        eqs_.emplace_back(std::move(constraint_value));
    }

    void AddIneqConstraint(ConstraintValuePtr<T, M, N, Inequality> &&constraint_value) {
        ineqs_.emplace_back(std::move(constraint_value));
    }

    ConstraintValues<Equality> &GetEqConstraints() {return eqs_;}

    ConstraintValues<Inequality> &GetIneqConstraints() {return ineqs_;}

    const ConstraintValues<Equality> &GetImmutableEqConstraints() const {return eqs_;}

    const ConstraintValues<Inequality> &GetImmutableIneqConstraints() const {return ineqs_;}

    const CostFuncs &GetCostTerms() const {return cost_terms_;}
private:
    CostTermConfig config_;
    CostFuncs cost_terms_;
    ConstraintValues<Equality> eqs_;
    ConstraintValues<Inequality> ineqs_;
    VectorNd<dim> eqc_violations_;
    VectorNd<dim> ineqc_violations_;
    VecX pN;
    MatrixLXX PN;
    int horizon_;
    double max_violation_ = std::numeric_limits<double>::max();
};

template<typename T, unsigned M, unsigned N>
using CostUnionPtr = std::unique_ptr<CostUnion<T, M, N>>;

#endif //CILQR_COST_CALC_H
