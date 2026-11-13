//
// Created by LTsommer on 2024/5/30.
//

#ifndef ALTRO_OCP_PROBLEM_HPP
#define ALTRO_OCP_PROBLEM_HPP

#include "model/model.h"
#include "augmented_ilqr/cost_calc.h"
#include <cassert>
#include <cstdlib>
#include <memory>

template<typename T, unsigned int M, unsigned int N>
class OCPInterface {
public:
    OCP_VARIABLES(T, M, N)
    OCPInterface() = default;
    virtual ~OCPInterface() = default;

    double Evaluate(const int step, const State &x, const Control &u) const {
        if (cost_union_ == nullptr) {
            return 0.0;
        }
        return cost_union_->Evaluate(step, x, u);
    };

    const CostUnionPtr<T, M, N> &GetCostUnionPtr() const { return cost_union_; };

    const CostUnion<T, M, N> &GetCostUnion() const {
        assert(cost_union_ != nullptr);
        if (!cost_union_) {
            std::abort();
        }
        return *cost_union_;
    }

    void SetModel(std::unique_ptr<ModelBase<T, M, N>> &&model) { model_.swap(model); }

    void SetCostUnion(std::unique_ptr<CostUnion<T, M, N>> &&cost_union) {
        cost_union_.swap(cost_union);
        cost_union_->SetHorizon(config_.steps.size());
    }

    const std::unique_ptr<ModelBase<T, M, N>> &GetModel() const { return model_; };

    void SetConfig(const CostTermConfig &config) { config_ = config; }

    template<template<typename, unsigned int, unsigned int> class CT,
             typename std::enable_if<std::is_base_of<CostFunc<T, M, N>, CT<T, M, N>>::value, int>::type = 0>
    void AddCostFunc() {
        cost_union_->AddCostTerm(std::make_unique<CT<T, M, N>>(config_));
    }

    template<typename CT, typename std::enable_if<std::is_base_of_v<CostFunc<T, M, N>, CT>, int>::type = 0>
    void AddCostFunc() {
        cost_union_->AddCostTerm(std::make_unique<CT>(config_));
    }

    void AddEqConstraint(ConstraintValuePtr<T, M, N, Equality> &&constraint_value) {
        cost_union_->AddEqConstraint(std::move(constraint_value));
    }

    void AddIneqConstraint(ConstraintValuePtr<T, M, N, Inequality> &&constraint_value) {
        cost_union_->AddIneqConstraint(std::move(constraint_value));
    }

protected:
    std::unique_ptr<ModelBase<T, M, N>> model_;
    std::unique_ptr<CostUnion<T, M, N>> cost_union_;
    CostTermConfig config_;
};

#endif //ALTRO_OCP_PROBLEM_HPP
