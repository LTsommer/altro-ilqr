//
// Created by LTsommer on 2024/5/30.
//

#ifndef CILQR_COST_CALC_H
#define CILQR_COST_CALC_H

#include "cost_function.hpp"
#include "cost_term_config.hpp"
#include "constraint_values.hpp"
#include "../frenet_coordinate/frenet_coordinate_system.h"
#include <iostream>

using std::cout;
using std::endl;

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
    explicit CostUnion(const CostTermConfig &config) : config_(config), horizon_(config.steps.size()) {
        FrenetCoordinateSystemParameters fcs_params;
        fcs_params.Init();
        fcs_.reset(new FrenetCoordinateSystem(config.ref_x, config.ref_y, fcs_params));
    };

    CostUnion &operator=(const CostUnion &) = delete;
    CostUnion(const CostUnion &) = delete;

    virtual ~CostUnion() = default;

    void SetHorizon(const int horizon) { horizon_ = horizon; }

    double Evaluate(const int step, const State &x, const Control &u) const;

    void CalcCostGradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const;

    void CalcCostHessian(const int step,
                         const State &x,
                         const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const;

    void CalcAugLagConstraintGradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const;

    void CalcAugLagConstraintHessian(const int step,
                                     const State &x,
                                     const Control &u,
                                     Eigen::Ref<MatrixLXX> cxx,
                                     Eigen::Ref<MatrixLUU> cuu,
                                     Eigen::Ref<MatrixLXU> cxu) const;

    void CalcTerminalCostToGo(const State &x);

    double GetMaxViolation() const { return max_violation_; };

    void EvaluateConstraints(const States &x, const Controls &u);

    VecX GetFinalCostToGoGradient() const { return pN; }

    MatrixLXX GetFinalCostToGoHessian() const { return PN; }

    void AddCostTerm(CostFuncPtr<T, M, N> &&cost_term) {
        type_names_[cost_term->GetTypeIndex()] = cost_term->GetName();
        std::cout << cost_term->GetName() << " is loaded\n";
        cost_terms_.emplace_back(std::move(cost_term));
    }

    void AddEqConstraint(ConstraintValuePtr<T, M, N, Equality> &&constraint_value) {
        type_names_[constraint_value->GetTypeIndex()] = constraint_value->GetName();
        std::cout << constraint_value->GetName() << " is loaded\n";
        eqs_.emplace_back(std::move(constraint_value));
    }

    void AddIneqConstraint(ConstraintValuePtr<T, M, N, Inequality> &&constraint_value) {
        type_names_[constraint_value->GetTypeIndex()] = constraint_value->GetName();
        std::cout << constraint_value->GetName() << " is loaded\n";
        ineqs_.emplace_back(std::move(constraint_value));
    }

    TypeNames GetTypeNames() const { return type_names_; }

    // Re-anchors the time-indexed reference data to where the trajectory actually is: for each
    // step, project the state onto the reference line and take the reference point / heading
    // and the boundary segment at that s. Without this, step t keeps using the boundary next
    // to the t-th reference point even when the vehicle is several meters behind or ahead of it.
    // Needs config.bound_s / bound_segments / bound_*_safe_dist; does nothing without them.
    void UpdateConfig(const States &x_res_seq) {
        const size_t num_bounds = config_.bound_s.size();
        if (!fcs_ or num_bounds < 2 or config_.bound_segments.size() != num_bounds or
            config_.bound_left_safe_dist.size() != num_bounds or config_.bound_right_safe_dist.size() != num_bounds)
            return;
        const int size = x_res_seq.size();
        config_.p_bound.assign(size, {});
        config_.p_ref.assign(size, Vector2d::Zero());
        config_.yaw.assign(size, 0.0);
        config_.ref_x.assign(size, 0.0);
        config_.ref_y.assign(size, 0.0);
        config_.left_safe_dist.assign(size, 0.0);
        config_.right_safe_dist.assign(size, 0.0);
        for (int t = 0; t < size; ++t) {
            Vec2d frenet_pt;
            (void)fcs_->CartCoord2FrenetCoord(Vec2d(x_res_seq[t](0), x_res_seq[t](1)), frenet_pt);
            const double s = frenet_pt.x();
            const Point2D ref_point = fcs_->GetRefCurvePoint(s);
            config_.p_ref[t] << ref_point.x, ref_point.y;
            config_.ref_x[t] = ref_point.x;
            config_.ref_y[t] = ref_point.y;
            config_.yaw[t] = fcs_->GetRefCurveHeading(s);

            // Boundary point nearest to s (bound_s[i - 1] <= s < bound_s[i], clamped at the ends).
            // Its boundary pieces run parallel to the road, so the line constraint built from
            // them stays valid around the vehicle.
            const auto it = std::upper_bound(config_.bound_s.begin(), config_.bound_s.end(), s);
            const size_t i = std::min(std::max<size_t>(std::distance(config_.bound_s.begin(), it), 1), num_bounds - 1);
            const size_t k = (s - config_.bound_s[i - 1] < config_.bound_s[i] - s) ? i - 1 : i;
            config_.p_bound[t] = config_.bound_segments[k];
            config_.left_safe_dist[t] = config_.bound_left_safe_dist[k];
            config_.right_safe_dist[t] = config_.bound_right_safe_dist[k];
        }

        for (const auto &ct : cost_terms_) {
            ct->UpdateConfig(config_);
        }

        for (const auto &cons_ptr : eqs_) {
            cons_ptr->UpdateConfig(config_);
        }

        for (const auto &cons_ptr : ineqs_) {
            cons_ptr->UpdateConfig(config_);
        }
    };

    ConstraintValues<Equality> &GetEqConstraints() { return eqs_; }

    ConstraintValues<Inequality> &GetIneqConstraints() { return ineqs_; }

    const ConstraintValues<Equality> &GetImmutableEqConstraints() const { return eqs_; }

    const ConstraintValues<Inequality> &GetImmutableIneqConstraints() const { return ineqs_; }

    const CostFuncs &GetCostTerms() const { return cost_terms_; }

    const CostTermConfig &GetConfig() const { return config_; }

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
    TypeNames type_names_;
    vector<double> max_penalty_;
    std::unique_ptr<FrenetCoordinateSystem> fcs_;
};

template<typename T, unsigned M, unsigned N>
using CostUnionPtr = std::unique_ptr<CostUnion<T, M, N>>;

#endif //CILQR_COST_CALC_H
