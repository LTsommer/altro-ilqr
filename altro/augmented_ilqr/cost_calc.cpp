//
// Created by LTsommer on 2024/5/30.
//

#include "augmented_ilqr/cost_calc.h"

template<typename T, unsigned M, unsigned int N>
double CostUnion<T, M, N>::Evaluate(const int step,
                                    const State &x,
                                    const Control &u) const {
    // AL-iLQR-tutorial 33
    double total_val = 0.0;
    for (const auto &ct : cost_terms_) {
        double cost_val = 0.0;
        bool success = ct->Evaluate(step, x, u, cost_val);
        if (success) {
            total_val += cost_val;
        }
    }

    for (const auto &cons_ptr : eqs_) {
        bool success = cons_ptr->Evaluate(step, x, u);
        if (success) {
            double cost_val = cons_ptr->AugLagVal();
            total_val += cost_val;
        }
    }

    for (const auto &cons_ptr : ineqs_) {
        bool success = cons_ptr->Evaluate(step, x, u);
        double cost_val = cons_ptr->AugLagVal();
        if (success) {
            total_val += cost_val;
        }
    }

    return total_val;
}

template<typename T, unsigned M, unsigned int N>
void CostUnion<T, M, N>::CalcCostGradient(const int step,
                                          const State &x,
                                          const Control &u,
                                          Eigen::Ref<VecX> lx,
                                          Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    // AL-iLQR-tutorial 41-43
    for (const auto &ct : cost_terms_) {
        VecX cur_lx;
        VecU cur_lu;
        ct->Gradient(step, x, u, cur_lx, cur_lu);
        lx += cur_lx;
        lu += cur_lu;
    }
}

template<typename T, unsigned M, unsigned N>
void CostUnion<T, M, N>::CalcCostHessian(const int step,
                                         const State &x,
                                         const Control &u,
                                         Eigen::Ref<MatrixLXX> lxx,
                                         Eigen::Ref<MatrixLUU> luu,
                                         Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    // AL-iLQR-tutorial 41-43
    for (const auto &ct : cost_terms_) {
        MatrixLXX cur_lxx;
        MatrixLUU cur_luu;
        MatrixLXU cur_lxu;
        ct->Hessian(step, x, u, cur_lxx, cur_luu, cur_lxu);
        lxx += cur_lxx;
        luu += cur_luu;
        lxu += cur_lxu;
    }
}

template<typename T, unsigned M, unsigned N>
void CostUnion<T, M, N>::CalcAugLagConstraintGradient(
        const int step,
        const State &x,
        const Control &u,
        Eigen::Ref<VecX> lx,
        Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    // AL-iLQR-tutorial 44-45
    for (const auto &cons_ptr : eqs_) {
        cons_ptr->Gradient(step, x, u);
        VecX cur_cx = cons_ptr->GradientX();
        VecU cur_cu = cons_ptr->GradientU();
        // lambda + penalty * c(x)
        double lag_cons_val =
                cons_ptr->GetLambda(step) +
                cons_ptr->ActivePenalty(step) * cons_ptr->ConsVal();
        lx += cur_cx * lag_cons_val;
        lu += cur_cu * lag_cons_val;
    }

    for (const auto &cons_ptr : ineqs_) {
        cons_ptr->Gradient(step, x, u);
        VecX cur_cx = cons_ptr->GradientX();
        VecU cur_cu = cons_ptr->GradientU();
        // lambda + penalty * c(x)
        double lag_cons_val =
                cons_ptr->GetLambda(step) +
                cons_ptr->ActivePenalty(step) * cons_ptr->ConsVal();
        lx += cur_cx * lag_cons_val;
        lu += cur_cu * lag_cons_val;
    }
}

template<typename T, unsigned M, unsigned N>
void CostUnion<T, M, N>::CalcAugLagConstraintHessian(
        const int step,
        const State &x,
        const Control &u,
        Eigen::Ref<MatrixLXX> cxx,
        Eigen::Ref<MatrixLUU> cuu,
        Eigen::Ref<MatrixLXU> cxu) const {
    cxx.setZero();
    cuu.setZero();
    cxu.setZero();
    // Gauss-Newton term I_mu * mu * dc * dc^T for both equality and inequality constraints
    // (inequalities were previously missing here).
    auto add_gn_term = [&](auto &cons_ptr) {
        cons_ptr->Gradient(step, x, u);
        const double mu = cons_ptr->ActivePenalty(step);
        VecX cur_cx = cons_ptr->GradientX();
        VecU cur_cu = cons_ptr->GradientU();
        cxx += cur_cx * mu * cur_cx.transpose();
        cuu += cur_cu * mu * cur_cu.transpose();
        cxu += cur_cx * mu * cur_cu.transpose();
    };
    for (const auto &cons_ptr : eqs_) {
        add_gn_term(cons_ptr);
    }
    for (const auto &cons_ptr : ineqs_) {
        add_gn_term(cons_ptr);
    }
}

template<typename T, unsigned M, unsigned N>
void CostUnion<T, M, N>::CalcTerminalCostToGo(const State &x) {
    pN.setZero();
    PN.setZero();
    for (const auto &ct : cost_terms_) {
        VecX cur_lx;
        VecU cur_lu;
        bool success =
                ct->Gradient(horizon_, x, Control::Zero(), cur_lx, cur_lu);
        // AL-iLQR-tutorial 38
        if (success) {
            pN += cur_lx;
        }

        MatrixLXX lxx;
        MatrixLUU luu;
        MatrixLXU lxu;
        success = ct->Hessian(horizon_, x, Control::Zero(), lxx, luu, lxu);
        // AL-iLQR-tutorial 39
        if (success) {
            PN += lxx;
        }
    }

    for (const auto &cons_ptr : eqs_) {
        cons_ptr->Evaluate(horizon_, x, Control::Zero());
        cons_ptr->Gradient(horizon_, x, Control::Zero());
        // AL-iLQR-tutorial 38-39
        double lag_cons_val =
                cons_ptr->GetLambda(horizon_) +
                cons_ptr->ActivePenalty(horizon_) * cons_ptr->ConsVal();
        pN += cons_ptr->GradientX() * lag_cons_val;
        PN += cons_ptr->GradientX() * cons_ptr->ActivePenalty(horizon_) *
              cons_ptr->GradientX().transpose();
    }

    for (const auto &cons_ptr : ineqs_) {
        cons_ptr->Evaluate(horizon_, x, Control::Zero());
        cons_ptr->Gradient(horizon_, x, Control::Zero());
        // AL-iLQR-tutorial 38-39
        double lag_cons_val =
                cons_ptr->GetLambda(horizon_) +
                cons_ptr->ActivePenalty(horizon_) * cons_ptr->ConsVal();
        pN += cons_ptr->GradientX() * lag_cons_val;
        PN += cons_ptr->GradientX() * cons_ptr->ActivePenalty(horizon_) *
              cons_ptr->GradientX().transpose();
    }
}

template<typename T, unsigned M, unsigned N>
void CostUnion<T, M, N>::EvaluateConstraints(const States &x,
                                             const Controls &u) {
    // Also refreshes every constraint value (including the terminal step), so dual
    // updates afterwards use the accepted trajectory rather than a rejected line-search trial.
    max_violation_ = 0.0;
    for (int t = 0; t <= horizon_; ++t) {
        const Control u_t = t < horizon_ ? u[t] : Control::Zero();
        for (const auto &cons_ptr : eqs_) {
            bool success = cons_ptr->Evaluate(t, x[t], u_t);
            if (success) {
                // equality violation is |c|
                max_violation_ = std::max(max_violation_,
                                          std::fabs(cons_ptr->ConsVal()));
            }
        }

        for (const auto &cons_ptr : ineqs_) {
            bool success = cons_ptr->Evaluate(t, x[t], u_t);
            if (success) {
                // inequality violation is max(c, 0)
                max_violation_ = std::max(max_violation_, cons_ptr->ConsVal());
            }
        }
    }
}

template<typename T, unsigned M, unsigned N>
CostUnion<T, M, N>::CostUnion(const CostTermConfig &config)
    : config_(config), horizon_(config.steps.size()) {
    FrenetCoordinateSystemParameters fcs_params;
    fcs_params.Init();
    fcs_.reset(
            new FrenetCoordinateSystem(config.ref_x, config.ref_y, fcs_params));
}

template<typename T, unsigned M, unsigned N>
void CostUnion<T, M, N>::AddCostTerm(CostFuncPtr<T, M, N> &&cost_term) {
    type_names_[cost_term->GetTypeIndex()] = cost_term->GetName();
    std::cout << cost_term->GetName() << " is loaded\n";
    cost_terms_.emplace_back(std::move(cost_term));
}

template<typename T, unsigned M, unsigned N>
void CostUnion<T, M, N>::AddEqConstraint(
        ConstraintValuePtr<T, M, N, Equality> &&constraint_value) {
    type_names_[constraint_value->GetTypeIndex()] = constraint_value->GetName();
    std::cout << constraint_value->GetName() << " is loaded\n";
    eqs_.emplace_back(std::move(constraint_value));
}

template<typename T, unsigned M, unsigned N>
void CostUnion<T, M, N>::AddIneqConstraint(
        ConstraintValuePtr<T, M, N, Inequality> &&constraint_value) {
    type_names_[constraint_value->GetTypeIndex()] = constraint_value->GetName();
    std::cout << constraint_value->GetName() << " is loaded\n";
    ineqs_.emplace_back(std::move(constraint_value));
}

template<typename T, unsigned M, unsigned N>
void CostUnion<T, M, N>::UpdateConfig(const States &x_res_seq) {
    const size_t num_bounds = config_.bound_s.size();
    if (!fcs_ or num_bounds < 2 or
        config_.bound_segments.size() != num_bounds or
        config_.bound_left_safe_dist.size() != num_bounds or
        config_.bound_right_safe_dist.size() != num_bounds) {
        return;
    }
    const int size = x_res_seq.size();
    // resize, not assign(size, {}): keeps the buffers of the per-step boundary segments, so the
    // element-wise copies below do not allocate again on every update.
    config_.p_bound.resize(size);
    config_.p_ref.assign(size, Vector2d::Zero());
    config_.yaw.assign(size, 0.0);
    config_.ref_x.assign(size, 0.0);
    config_.ref_y.assign(size, 0.0);
    config_.left_safe_dist.assign(size, 0.0);
    config_.right_safe_dist.assign(size, 0.0);
    for (int t = 0; t < size; ++t) {
        Vec2d frenet_pt;
        (void)fcs_->CartCoord2FrenetCoord(
                Vec2d(x_res_seq[t](0), x_res_seq[t](1)), frenet_pt);
        const double s = frenet_pt.x();
        const Point2D ref_point = fcs_->GetRefCurvePoint(s);
        config_.p_ref[t] << ref_point.x, ref_point.y;
        config_.ref_x[t] = ref_point.x;
        config_.ref_y[t] = ref_point.y;
        config_.yaw[t] = fcs_->GetRefCurveHeading(s);

        // Boundary point nearest to s (bound_s[i - 1] <= s < bound_s[i], clamped at the ends).
        // Its boundary pieces run parallel to the road, so the line constraint built from
        // them stays valid around the vehicle.
        const auto it = std::upper_bound(
                config_.bound_s.begin(), config_.bound_s.end(), s);
        const size_t i = std::min(
                std::max<size_t>(std::distance(config_.bound_s.begin(), it), 1),
                num_bounds - 1);
        const size_t k = (s - config_.bound_s[i - 1] < config_.bound_s[i] - s)
                                 ? i - 1
                                 : i;
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
}

template class CostUnion<double, 5, 1>;

template class CostUnion<double, 4, 2>;

template class CostUnion<double, 6, 2>;
