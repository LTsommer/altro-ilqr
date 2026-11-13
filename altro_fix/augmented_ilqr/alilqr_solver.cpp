//
// Created by LTsommer on 2024/5/30.
//

#include "alilqr_solver.h"
#include "../clock.hpp"
#include "../common/math_utils.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>

using std::cout;

template<typename T, unsigned int M, unsigned int N>
void ALILQRSolver<T, M, N>::Solve(const State &x0, const Controls &u0, States &x_res_seq, Controls &u_res_seq) {
    std::unique_ptr<ILQRSolverState<T, M, N>> ilqr_state =
            std::make_unique<ILQRSolverState<T, M, N>>(step_sizes().size());
    // Everything after this works on the initial trajectory; if its rollout is not finite (NaN /
    // inf in x0, u0 or the resulting states / cost) there is nothing valid to return.
    const State initial_state = Problem().GetModel()->PrepareInitialState(x0);
    if (!std::isfinite(InitTraj(initial_state, u0, ilqr_state.get()))) {
        throw std::runtime_error("ALILQRSolver: initial rollout is not finite or violates the model input range");
    }
    GenerateTrajectory(ilqr_state.get());
    x_res_seq = ilqr_state->x_seq_;
    u_res_seq = ilqr_state->u_seq_;
}

template<typename T, unsigned int M, unsigned int N>
double ALILQRSolver<T, M, N>::InitTraj(const State &x_0,
                                       const Controls &u_0,
                                       ILQRSolverState<T, M, N> *ilqr_state) const {
    // Extra initialization
    ilqr_state->x0_ = x_0;
    ilqr_state->x_seq_[0] = x_0;
    ilqr_state->u_seq_ = u_0;
    ilqr_state->dV_ = Vector2d::Zero();

    // Call forward_pass to get xs, us, cost
    double cost_ini = RollOut(ilqr_state->x0_, ilqr_state->u_seq_, ilqr_state);
    ilqr_state->cost_ = cost_ini;

    return ilqr_state->cost_;
}

template<typename T, unsigned int M, unsigned int N>
void ALILQRSolver<T, M, N>::GenerateTrajectory(ILQRSolverState<T, M, N> *ilqr_state) {
    StopWatch stop_watch_total;
    stop_watch_total.start();
    uint32_t derivative_time_microseconds = 0U;
    uint32_t backward_time_microseconds = 0U;
    uint32_t forward_time_microseconds = 0U;
    uint32_t update_config_time_microseconds = 0U;

    const auto &cost_union = Problem().GetCostUnionPtr();
    max_penalty_ = 0.0;
    ilqr_state->status_ = SolverStatus::OCPUnsolved;
    int total_iters = 0;
    double prev_max_violation = std::numeric_limits<double>::infinity();

    // Outer loop: augmented Lagrangian. Each outer iteration solves the unconstrained
    // problem for fixed multipliers / penalties, then updates them once.
    for (int outer = 0; outer < kMaxOuterIter_; ++outer) {
        bool need_to_recompute = true; // if to recompute derivatives
        bool inner_failed = false;

        // Re-anchor reference points / heading / boundaries to the current trajectory once per
        // outer iteration, so every inner iLQR solve minimizes one fixed objective (changing it
        // after each accepted step would make the line search compare costs of different
        // problems). The cost of the current trajectory changes with it.
        if (update_config_) {
            StopWatch stop_watch;
            stop_watch.start();
            cost_union->UpdateConfig(ilqr_state->x_seq());
            ilqr_state->cost_ = EvaluateTotalCost(ilqr_state->x_seq(), ilqr_state->u_seq());
            update_config_time_microseconds += stop_watch.elapsed_microseconds();
        }

        // Inner loop: plain iLQR on the current augmented Lagrangian.
        for (int iter = 0; iter < kMaxIter_; ++iter, ++total_iters) {
            const States x_old = ilqr_state->x_seq_;
            const Controls u_old = ilqr_state->u_seq_;

            //--------------------------------------------------------------------------
            // STEP 1: Differentiate dynamics and cost along new trajectory
            StopWatch stop_watch;
            stop_watch.start();
            if (need_to_recompute) {
                InitializeILQRSolverState(ilqr_state);
                need_to_recompute = false;
            }
            derivative_time_microseconds += stop_watch.elapsed_microseconds();

            //--------------------------------------------------------------------------
            // STEP 2: Backward pass, compute optimal control law and cost-to-go
            stop_watch.start();
            const bool backward_ok = BackwardProcess(ilqr_state);
            backward_time_microseconds += stop_watch.elapsed_microseconds();
            if (!backward_ok) {
                if (verbose_) {
                    cout << "EXIT inner loop: regularization failed\n";
                }
                inner_failed = true;
                break;
            }

            // Inner convergence, part 1: the quadratic model predicts a negligible decrease
            // for a full step, relative to the current cost. Measured in cost units, so it does
            // not depend on how large the controls happen to be (the old test on |k| / (|u| + 1)
            // stopped almost immediately when the controls are tiny, e.g. ddkappa ~ 1e-2).
            const double cost_scale = std::max(1.0, std::fabs(ilqr_state->cost_));
            // A negative prediction means the local model is not trustworthy here (e.g. an
            // indefinite cost Hessian), not that we have converged; let the line search decide.
            const double expected_full = -(ilqr_state->dV_(0) + ilqr_state->dV_(1));
            if (expected_full >= 0.0 and expected_full < kTolExpRel_ * cost_scale) {
                break;
            }

            //--------------------------------------------------------------------------
            // STEP 3: Forward pass / line-search to find new control sequence,
            // trajectory, cost
            stop_watch.start();
            double dcost = 0.0;    // cost decrease
            double ratio = 0.0;    // ratio of actual vs expected cost reduction
            double expected = 0.0; // expected cost reduction
            double new_cost = 0.0; // cost after forward pass
            const bool is_fwd_pass_done =
                    ForwardProcess(x_old, u_old, &new_cost, &dcost, &expected, &ratio, ilqr_state);
            forward_time_microseconds += stop_watch.elapsed_microseconds();

            ilqr_state->g_norm_ = GetGradientNorm(ilqr_state->k_, ilqr_state->u_seq_);

            //--------------------------------------------------------------------------
            // STEP 4: accept step (or not), check inner convergence
            if (is_fwd_pass_done) {
                need_to_recompute = true;
                // Inner convergence, part 2: the accepted step barely reduced the cost.
                if (dcost < kTolFunRel_ * cost_scale) {
                    break;
                }
            } else {
                IncreaseRho(ilqr_state->drho_, ilqr_state->rho_);
                if (ilqr_state->rho_ >= krhoMax_) {
                    if (verbose_) {
                        cout << "EXIT inner loop: rho > rhoMax\n";
                    }
                    inner_failed = true;
                    break;
                }
            }
        }

        // Evaluate constraints on the accepted trajectory (this also refreshes the values
        // the dual update below uses, instead of the last rejected line-search trial).
        cost_union->EvaluateConstraints(ilqr_state->x_seq(), ilqr_state->u_seq());
        const double max_violation = cost_union->GetMaxViolation();
        if (verbose_) {
            cout << "AL outer iter " << outer << ": max_violation = " << max_violation
                 << ", max_penalty = " << max_penalty_ << "\n";
        }

        if (max_violation < kViolationTol_) {
            ilqr_state->status_ = SolverStatus::OCPSolved;
            break;
        }
        if (inner_failed) {
            ilqr_state->status_ = SolverStatus::RegularizationFailed;
            break;
        }

        // Penalties are capped (ConstraintValue::kMaxPenalty), so reaching the cap is not a
        // failure by itself: the multipliers keep improving. Stop only when some penalty is
        // already at the cap and the violation has stopped decreasing.
        if (AnyPenaltyAtMax() and max_violation > kStallRatio_ * prev_max_violation) {
            if (verbose_) {
                cout << "EXIT: penalty at its cap and no further progress\n";
            }
            ilqr_state->status_ = SolverStatus::MaxPenalty;
            break;
        }
        prev_max_violation = max_violation;

        UpdateDualsAndPenalties();
        UpdateConvergenceState();

        // Multipliers and penalties changed, so the AL cost of the current trajectory changed.
        ilqr_state->cost_ = EvaluateTotalCost(ilqr_state->x_seq(), ilqr_state->u_seq());

        if (outer == kMaxOuterIter_ - 1) {
            if (verbose_) {
                cout << "EXIT: Maximum outer iterations reached\n";
            }
            ilqr_state->status_ = SolverStatus::MaxIterReached;
        }
    } // end AL outer loop

    status_ = ilqr_state->status_;
    // Report violations against the boundaries at the final positions.
    if (update_config_) {
        cost_union->UpdateConfig(ilqr_state->x_seq());
    }
    cost_union->EvaluateConstraints(ilqr_state->x_seq(), ilqr_state->u_seq());
    double max_violation = cost_union->GetMaxViolation();
    // Stop the clock before printing, so console output is not counted as solve time.
    uint32_t total_time_microseconds = stop_watch_total.elapsed_microseconds();
    total_iters_ = total_iters;
    solve_time_ms_ = total_time_microseconds / 1.0e3;
    if (!print_summary_) {
        return;
    }
    cout << "Solver Status : ";
    ShowSolverState();
    cout << "total iLQR iterations : " << total_iters << "\n";
    cout << "max_violation : " << max_violation << "\n";
    for (const auto &cons_ptr : cost_union->GetImmutableEqConstraints()) {
        cout << "  " << cons_ptr->GetName() << " max violation : " << cons_ptr->GetMaxViolation() << "\n";
    }
    for (const auto &cons_ptr : cost_union->GetImmutableIneqConstraints()) {
        cout << "  " << cons_ptr->GetName() << " max violation : " << cons_ptr->GetMaxViolation() << "\n";
    }
    cout << "AL-iLQR total time : " << total_time_microseconds / 1.0e3 << "ms\n"
         << "calc Gradient and Hessian : " << derivative_time_microseconds / 1.0e3 << "ms\n"
         << "backward process : " << backward_time_microseconds / 1.0e3 << "ms\n"
         << "forward process : " << forward_time_microseconds / 1.0e3 << "ms\n"
         << "update boundary, ref_points, ref_angle etc :" << update_config_time_microseconds / 1.0e3 << "ms\n"
         << "unknown consumption : "
         << (total_time_microseconds - (derivative_time_microseconds + backward_time_microseconds +
                                        forward_time_microseconds + update_config_time_microseconds)) /
                    1.0e3
         << "ms\n";
}

template<typename T, unsigned int M, unsigned int N>
double ALILQRSolver<T, M, N>::EvaluateTotalCost(const States &x, const Controls &u) const {
    double total_cost = 0.0;
    const uint32_t step_nums = step_sizes_.size();
    for (int t = 0; t < static_cast<int>(step_nums); ++t) {
        total_cost += Problem().Evaluate(t, x[t], u[t]);
    }
    total_cost += Problem().Evaluate(step_nums, x[step_nums], Control::Zero());
    return total_cost;
}

template<typename T, unsigned M, unsigned N>
void ALILQRSolver<T, M, N>::UpdateDualsAndPenalties() const {
    for (auto &cons_ptr : Problem().GetCostUnionPtr()->GetEqConstraints()) {
        cons_ptr->UpdateDual();
        cons_ptr->UpdatePenalty();
    }

    for (auto &cons_ptr : Problem().GetCostUnionPtr()->GetIneqConstraints()) {
        cons_ptr->UpdateDual();
        cons_ptr->UpdatePenalty();
    }
}

template<typename T, unsigned M, unsigned N>
bool ALILQRSolver<T, M, N>::AnyPenaltyAtMax() const {
    for (const auto &cons_ptr : Problem().GetCostUnionPtr()->GetImmutableEqConstraints()) {
        if (cons_ptr->AnyPenaltyAtMax()) {
            return true;
        }
    }
    for (const auto &cons_ptr : Problem().GetCostUnionPtr()->GetImmutableIneqConstraints()) {
        if (cons_ptr->AnyPenaltyAtMax()) {
            return true;
        }
    }
    return false;
}

template<typename T, unsigned M, unsigned N>
void ALILQRSolver<T, M, N>::UpdateConvergenceState() {
    for (const auto &cons_ptr : Problem().GetCostUnionPtr()->GetImmutableEqConstraints()) {
        max_penalty_ = std::max(max_penalty_, cons_ptr->GetMaxPenalty());
    }
    for (const auto &cons_ptr : Problem().GetCostUnionPtr()->GetImmutableIneqConstraints()) {
        max_penalty_ = std::max(max_penalty_, cons_ptr->GetMaxPenalty());
    }
}

template<typename T, unsigned int M, unsigned int N>
bool ALILQRSolver<T, M, N>::ForwardProcess(const States &x_old,
                                           const Controls &u_old,
                                           double *new_cost,
                                           double *dcost,
                                           double *expected,
                                           double *ratio,
                                           ILQRSolverState<T, M, N> *ilqr_state) const {
    bool is_fwd_pass_done = false;
    double alpha = 1.0;
    double gamma = 0.5;
    for (int i = 0; i < 20; ++i) {
        Controls u_plu_seq_feedforward = ilqr_state->u_seq_;
        for (int j = 0; j < static_cast<int>(ilqr_state->u_seq_.size()); j++) {
            u_plu_seq_feedforward[j] += ilqr_state->k_[j] * alpha;
        }

        *new_cost = RollOut(ilqr_state->x0_, u_plu_seq_feedforward, ilqr_state);
        // A trial whose state, control or cost is not finite is rejected explicitly (it used to
        // be rejected only because comparisons with NaN are false).
        if (!std::isfinite(*new_cost)) {
            if (verbose_) {
                cout << "Line search: alpha = " << alpha << " gives a non-finite trajectory, rejected\n";
            }
            ilqr_state->x_seq_ = x_old;
            ilqr_state->u_seq_ = u_old;
            alpha *= gamma;
            continue;
        }

        *dcost = ilqr_state->cost_ - *new_cost;
        *expected = -alpha * (ilqr_state->dV_(0) + alpha * ilqr_state->dV_(1));

        if (*expected > 0) {
            *ratio = *dcost / *expected;
        } else {
            *ratio = sgn(*dcost);
            ilqr_state->status_ = SolverStatus::CostIncrease;
            if (verbose_) {
                cout << "Warning: non-positive expected reduction\n";
            }
        }

        // Armijo-like acceptance as in [Tassa 2012]: only a lower bound on the ratio.
        // (Previously compared against krhoMax_, which was a typo for kRationMax_.)
        if (*ratio > kRatioMin_) {
            is_fwd_pass_done = true;
            DecreaseRho(ilqr_state->drho_, ilqr_state->rho_);
            ilqr_state->cost_ = *new_cost; // accept step
            break;
        }
        // Line search failed for this alpha, so reset x_seq
        // and u_seq since they are updated in roll_out()
        ilqr_state->x_seq_ = x_old;
        ilqr_state->u_seq_ = u_old;
        alpha *= gamma;
    }

    return is_fwd_pass_done;
}

/*
  Forward roll out of states using original dynamics
    INPUTS
      x0: M * 1, initial state
      u: N * 1 * T, control input used in roll out
    OUTPUTS
      total_cost: double, total cost value after roll out
*/
template<typename T, unsigned int M, unsigned int N>
double ALILQRSolver<T, M, N>::RollOut(const State &x0, const Controls &u, ILQRSolverState<T, M, N> *ilqr_state) const {
    double total_cost = 0;
    const auto &model = Problem().GetModel();

    State x_curr = x0;
    Control u_curr;

    const uint32_t step_nums = step_sizes_.size();
    States x_new(step_nums + 1);
    x_new[0] = x0;
    for (int t = 0; t < static_cast<int>(step_nums); t++) {
        u_curr = u[t];
        if (ilqr_state->K_.size() > 0) {
            // apply LQR control gains after first iteration
            u_curr += ilqr_state->K_[t] * model->StateDifference(x_new[t], ilqr_state->x_seq_[t]);
        }
        // The model is only defined for finite inputs: stop here and report the rollout as
        // invalid (infinite cost) instead of passing NaN / inf on.
        if (!x_curr.allFinite() or !u_curr.allFinite()) {
            return std::numeric_limits<double>::infinity();
        }

        ilqr_state->u_seq_[t] = u_curr; // no clamping since no control limit
        total_cost += Problem().Evaluate(t, x_curr, u_curr);
        x_curr = model->ForwardCalculation(x_curr, u_curr, step_sizes_[t]);
        x_new[t + 1] = x_curr;
    }

    if (!x_new[step_nums].allFinite()) {
        return std::numeric_limits<double>::infinity();
    }
    ilqr_state->x_seq_ = x_new;
    total_cost += Problem().Evaluate(step_nums, x_new[step_nums], Control::Zero());
    return total_cost;
}

/*
  Perform the Ricatti solution based backward pass
   INPUTS
      cx: M * (T+1)        cu: N * (T+1)
      cuu: N * N * (T+1)   cxx: M * M * (T+1)  cxu: M * N * (T+1)
      fx: M * M * (T+1)    fu: M * N * (T+1)   fxx: none
      fxu: None            fuu: none           u: N * T
    OUTPUTS
      Vx: M * (T+1)      Vxx: M * M * (T+1)      k: N * T
      K: N * M * T         dV: 2 * 1
*/
template<typename T, unsigned int M, unsigned int N>
bool ALILQRSolver<T, M, N>::BackwardProcess(ILQRSolverState<T, M, N> *ilqr_state) const {
    const uint32_t step_nums = step_sizes().size();
    if (step_nums == 0) {
        cout << "iLQR backward_pass: planning horizon length is zero\n";
        return false;
    }
    // Counted across retries (it used to be reset inside the loop, which could spin forever).
    int reg_count = 0;

    while (true) {
        bool is_cost_psd = true;
        ilqr_state->dV_.setZero();
        for (int i = static_cast<int>(step_nums - 1); i >= 0; i--) {
            // backward from the end
            ilqr_state->Qx_ =
                    ilqr_state->lx_[i] + (ilqr_state->fx_[i].transpose() * ilqr_state->Vx_[i + 1]) + ilqr_state->cx_[i];
            ilqr_state->Qu_ =
                    ilqr_state->lu_[i] + (ilqr_state->fu_[i].transpose() * ilqr_state->Vx_[i + 1]) + ilqr_state->cu_[i];
            ilqr_state->Qxx_ = ilqr_state->lxx_[i] +
                               (ilqr_state->fx_[i].transpose() * ilqr_state->Vxx_[i + 1] * ilqr_state->fx_[i]) +
                               ilqr_state->cxx_[i];
            ilqr_state->Qux_ = ilqr_state->lxu_[i].transpose() +
                               (ilqr_state->fu_[i].transpose() * ilqr_state->Vxx_[i + 1] * ilqr_state->fx_[i]) +
                               ilqr_state->cxu_[i].transpose();
            ilqr_state->Quu_ = ilqr_state->luu_[i] +
                               (ilqr_state->fu_[i].transpose() * ilqr_state->Vxx_[i + 1] * ilqr_state->fu_[i]) +
                               ilqr_state->cuu_[i];

            // Similar to equations 10a and 10b in [Tassa 2012]. Note that
            // regularization is different from the paper but consistent
            // with Tassa's Matlab implementation)
            // improved regularization QuuF = Quu + mu * I
            ilqr_state->QuuF_ = ilqr_state->Quu_ + ilqr_state->rho_ * MatrixLUU::Identity();

            // The following method is only efficient when n <= 4
            MatrixLUU QuuF_inv;
            double QuuF_det;
            bool QuuF_invertible;
            double tol = 1.0e-6;
            auto start = std::chrono::steady_clock::now();
            ilqr_state->QuuF_.computeInverseAndDetWithCheck(QuuF_inv, QuuF_det, QuuF_invertible, tol);
            if (!QuuF_invertible) {
                is_cost_psd = false;
                break; // QuuF is not positive definite, need to increase lambda
            }

            // Compute the control gain
            ilqr_state->k_i_ = -QuuF_inv * ilqr_state->Qu_;
            ilqr_state->K_i_ = -QuuF_inv * ilqr_state->Qux_;

            // Update cost-to-go approximation. Equations 11 in [Tassa 2012]
            ilqr_state->dV_(0) += ilqr_state->k_i_.transpose() * ilqr_state->Qu_;
            ilqr_state->dV_(1) += 0.5 * (ilqr_state->k_i_.transpose() * ilqr_state->Quu_ * ilqr_state->k_i_)[0];
            ilqr_state->Vx_[i] = ilqr_state->Qx_ + ilqr_state->K_i_.transpose() * ilqr_state->Quu_ * ilqr_state->k_i_ +
                                 ilqr_state->K_i_.transpose() * ilqr_state->Qu_ +
                                 ilqr_state->Qux_.transpose() * ilqr_state->k_i_;
            ilqr_state->Vxx_[i] =
                    ilqr_state->Qxx_ + ilqr_state->K_i_.transpose() * ilqr_state->Quu_ * ilqr_state->K_i_ +
                    ilqr_state->K_i_.transpose() * ilqr_state->Qux_ + ilqr_state->Qux_.transpose() * ilqr_state->K_i_;
            ilqr_state->Vxx_[i] = 0.5 * (ilqr_state->Vxx_[i] + ilqr_state->Vxx_[i].transpose());

            // save controls/gains
            ilqr_state->k_[i] = ilqr_state->k_i_;
            ilqr_state->K_[i] = ilqr_state->K_i_;
        }

        if (is_cost_psd) {
            return true;
        }
        // rho is clamped at krhoMax_, so once it is there retrying cannot help.
        const bool rho_at_max = ilqr_state->rho_ >= krhoMax_;
        IncreaseRho(ilqr_state->drho_, ilqr_state->rho_);
        if (rho_at_max or ++reg_count > kMaxRegCount_) {
            ilqr_state->status_ = SolverStatus::RegularizationFailed;
            return false;
        }
    }
}

template<typename T, unsigned int M, unsigned int N>
void ALILQRSolver<T, M, N>::IncreaseRho(double &drho, double &rho) const {
    drho = std::max(drho * krhoFactor_, krhoFactor_);
    rho = std::max(rho * drho, krhoMin_);
    rho = std::min(rho, krhoMax_);
}

template<typename T, unsigned int M, unsigned int N>
void ALILQRSolver<T, M, N>::DecreaseRho(double &drho, double &rho) const {
    drho = std::min(drho / krhoFactor_, 1 / krhoFactor_);
    rho = std::max(rho * drho, krhoMin_);
    rho = std::min(rho, krhoMax_);
}

template<typename T, unsigned int M, unsigned int N>
double ALILQRSolver<T, M, N>::GetGradientNorm(const VecUs &l, const Controls &u) const {
    // Fixed-size expression and a running sum: called once per iLQR iteration, no heap use.
    double sum = 0.0;
    for (int i = 0; i < static_cast<int>(l.size()); i++) {
        sum += (l[i].cwiseAbs().array() / (u[i].cwiseAbs().array() + 1)).maxCoeff();
    }
    return sum / l.size();
}

template<typename T, unsigned M, unsigned N>
double ALILQRSolver<T, M, N>::GetMaxViolation() const {
    return Problem().GetCostUnionPtr()->GetMaxViolation();
}

template<typename T, unsigned M, unsigned N>
void ALILQRSolver<T, M, N>::InitializeILQRSolverState(ILQRSolverState<T, M, N> *ilqr_state) const {
    const uint32_t step_nums = step_sizes().size();
    States x = ilqr_state->x_seq();
    Controls u = ilqr_state->u_seq();
    CalcKinematicsDerivatives(x, u, &(ilqr_state->fx_), &(ilqr_state->fu_));
    for (int t = 0; t < static_cast<int>(step_nums); ++t) {
        (void)Problem().GetCostUnionPtr()->Evaluate(t, x[t], u[t]);
        Problem().GetCostUnionPtr()->CalcCostGradient(t, x[t], u[t], ilqr_state->lx_[t], ilqr_state->lu_[t]);
        Problem().GetCostUnionPtr()->CalcAugLagConstraintGradient(
                t, x[t], u[t], ilqr_state->cx_[t], ilqr_state->cu_[t]);
        Problem().GetCostUnionPtr()->CalcCostHessian(
                t, x[t], u[t], ilqr_state->lxx_[t], ilqr_state->luu_[t], ilqr_state->lxu_[t]);
        Problem().GetCostUnionPtr()->CalcAugLagConstraintHessian(
                t, x[t], u[t], ilqr_state->cxx_[t], ilqr_state->cuu_[t], ilqr_state->cxu_[t]);
    }
    Problem().GetCostUnionPtr()->CalcTerminalCostToGo(x[step_nums]);
    ilqr_state->Vx_[step_nums] = Problem().GetCostUnionPtr()->GetFinalCostToGoGradient();
    ilqr_state->Vxx_[step_nums] = Problem().GetCostUnionPtr()->GetFinalCostToGoHessian();
}

template<typename T, unsigned int M, unsigned int N>
void ALILQRSolver<T, M, N>::CalcKinematicsDerivatives(const States &x,
                                                      const Controls &u,
                                                      MatrixLXXs *f_x,
                                                      MatrixLXUs *f_u) const {
    const uint32_t step_nums = step_sizes().size();
    for (int t = 0; t < static_cast<int>(step_nums); t++) {
        (*f_x)[t] = Problem().GetModel()->JacobianX(x[t], u[t], step_sizes_[t]);
        (*f_u)[t] = Problem().GetModel()->JacobianU(x[t], u[t], step_sizes_[t]);
    }
}

template<typename T, unsigned int M, unsigned int N>
const int ALILQRSolver<T, M, N>::kMaxIter_ = 20;

template<typename T, unsigned int M, unsigned int N>
const int ALILQRSolver<T, M, N>::kMaxOuterIter_ = 10;

template<typename T, unsigned int M, unsigned int N>
const double ALILQRSolver<T, M, N>::kTolFunRel_ = 1e-6;

template<typename T, unsigned int M, unsigned int N>
const double ALILQRSolver<T, M, N>::kTolExpRel_ = 1e-7;

template<typename T, unsigned int M, unsigned int N>
const double ALILQRSolver<T, M, N>::kStallRatio_ = 0.99;

template<typename T, unsigned int M, unsigned int N>
const double ALILQRSolver<T, M, N>::krhoFactor_ = 1.6;

template<typename T, unsigned int M, unsigned int N>
const double ALILQRSolver<T, M, N>::krhoMax_ = 1e8;

template<typename T, unsigned int M, unsigned int N>
const double ALILQRSolver<T, M, N>::krhoMin_ = 1e-8;

template<typename T, unsigned int M, unsigned int N>
const double ALILQRSolver<T, M, N>::krhoMinGrad_ = 1e-5;

template<typename T, unsigned int M, unsigned int N>
const double ALILQRSolver<T, M, N>::kRatioMin_ = 1.0e-8;

template<typename T, unsigned M, unsigned N>
const double ALILQRSolver<T, M, N>::kRationMax_ = 10.0;

template<typename T, unsigned M, unsigned N>
const double ALILQRSolver<T, M, N>::kViolationTol_ = 1e-2;

template<typename T, unsigned M, unsigned N>
const int ALILQRSolver<T, M, N>::kMaxRegCount_ = 20;

template<typename T, unsigned int M, unsigned int N>
const std::array<double, 11> ALILQRSolver<T, M, N>::alpha_vec_{
        1.0000, 0.5012, 0.2512, 0.1259, 0.0631, 0.0316, 0.0158, 0.0079, 0.0040, 0.0020, 0.0010};

template class ALILQRSolver<double, 5, 1>;

template class ALILQRSolver<double, 4, 2>;

template class ALILQRSolver<double, 6, 2>;
