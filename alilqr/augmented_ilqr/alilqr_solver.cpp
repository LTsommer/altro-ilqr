//
// Created by LTsommer on 2024/5/30.
//

#include "alilqr_solver.h"
#include "../clock.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

using std::cout;

template<typename T, unsigned int M, unsigned int N>
void ILQRSolver<T, M, N>::Solve(const State &x0, const Controls &u0, States &x_res_seq, Controls &u_res_seq) {
    std::unique_ptr<ILQRSolverState<T, M, N>> ilqr_state =
            std::make_unique<ILQRSolverState<T, M, N>>(step_sizes().size());
    InitTraj(x0, u0, ilqr_state.get());
    GenerateTrajectory(ilqr_state.get());
    x_res_seq = ilqr_state->x_seq_;
    u_res_seq = ilqr_state->u_seq_;
}

template<typename T, unsigned int M, unsigned int N>
double ILQRSolver<T, M, N>::InitTraj(const State &x_0, const Controls &u_0, ILQRSolverState<T, M, N> *ilqr_state) const {
    // Extra initialization
    ilqr_state->x0_ = x_0;
    ilqr_state->x_seq_[0] = x_0;
    ilqr_state->u_seq_ = u_0;
    ilqr_state->dV_ = Vector2d::Zero();

    // Call forward_pass to get xs, us, cost
    double cost_ini = RollOut(ilqr_state->x0_, ilqr_state->u_seq_, ilqr_state);
    //  NLOGD("Initial cost: %.3g", cost_ini);
    ilqr_state->cost_ = cost_ini;

    return ilqr_state->cost_;
}

template<typename T, unsigned int M, unsigned int N>
void ILQRSolver<T, M, N>::GenerateTrajectory(ILQRSolverState<T, M, N> *ilqr_state)  {
    States x_old;
    Controls u_old;

    bool reinit_solver_state = true; // if to recompute derivatives
    StopWatch stop_watch_total;
    stop_watch_total.start();
    uint32_t derivative_time_microseconds = 0U;
    uint32_t backward_time_microseconds = 0U;
    uint32_t forward_time_microseconds = 0U;
    for (int iter = 0; iter < kMaxIter_; iter++) {
        x_old = ilqr_state->x_seq_;
        u_old = ilqr_state->u_seq_;

        //--------------------------------------------------------------------------
        // STEP 1: Differentiate dynamics and cost along new trajectory

        StopWatch stop_watch;
        stop_watch.start();
        if (reinit_solver_state) {
            InitializeILQRSolverState(ilqr_state);
            reinit_solver_state = false;
        }
        derivative_time_microseconds += stop_watch.elapsed_microseconds();
        //--------------------------------------------------------------------------
        // STEP 2: Backward pass, compute optimal control law and cost-to-go

        stop_watch.start();
        // update Vx, Vxx, l, L, dV with backward_pass
        BackwardProcess(ilqr_state);
        backward_time_microseconds += stop_watch.elapsed_microseconds();
        //--------------------------------------------------------------------------
        // STEP 3: Forward pass / line-search to find new control sequence,
        // trajectory, cost

        
        stop_watch.start();
        double dcost = 0.0;    // cost decrease
        double ratio = 0.0;    // ratio of actual vs expected cost reduction
        double expected = 0.0; // expected cost reduction
        double new_cost = 0.0; // cost after forward pass
        bool is_fwd_pass_done =
                ForwardProcess(x_old, u_old, &new_cost, &dcost, &expected, &ratio,
                               ilqr_state);

        ilqr_state->g_norm_ = GetGradientNorm(ilqr_state->k_, ilqr_state->u_seq_);
        if (ilqr_state->g_norm_ < kTolGrad_ and dcost > 0.0) {
            ilqr_state->status_ = SolverStatus::OCPSolved;
            cout << "SUCCESS: gradient norm < tolGrad\n";
            break;
        }
        forward_time_microseconds += stop_watch.elapsed_microseconds();

        //--------------------------------------------------------------------------
        // STEP 4: accept step (or not), log status
        if (iter == 0) {

            if (ilqr_state->status_ != SolverStatus::OCPSolved)
                ilqr_state->status_ = SolverStatus::MaxIterReached;
            cout << "AL-iLQR is terminated\n";
        }

        if (is_fwd_pass_done) {
            reinit_solver_state = true;
            // terminate if cost reduction is small enough
            if (dcost < kTolFun_) {
                //        NLOGD("SUCCESS: cost change < tolFun");
                ilqr_state->status_ = SolverStatus::OCPSolved;
                cout << "SUCCESS: cost change < tolFun\n";
                break;
            }
        } else {
//            cout << "Iteration " << iter << ": Forward pass failed when line searching\n";

            IncreaseRho(ilqr_state->drho_, ilqr_state->rho_);
            // terminate if lambda reaches maximum
            if (ilqr_state->rho_ > krhoMax_) {
                cout << "EXIT: rho > rhoMax\n";
                break;
            }
        }

        UpdateDualsAndPenalties(ilqr_state);

        if (iter == kMaxIter_) {
            if (ilqr_state->status_ != SolverStatus::OCPSolved)
                ilqr_state->status_ = SolverStatus::MaxIterReached;
            cout << "EXIT: Maximum iterations reached\n";
        }
    } // end top-level for-loop

    Problem().GetCostUnionPtr()->EvaluateConstraints(ilqr_state->x_seq(), ilqr_state->u_seq());
//    cout << "Solver Status : ";
    status_ = ilqr_state->status_;
//    ShowSolverState();
//    cout << "max_violation : " << max_violation << "\n";
    uint32_t total_time_microseconds = stop_watch_total.elapsed_microseconds();
    cout << "AL-iLQR total time : " << total_time_microseconds / 1.0e3 << "ms\n"
         << "calc Gradient and Hessian : " << derivative_time_microseconds / 1.0e3 << "ms\n"
         << "backward process : " << backward_time_microseconds / 1.0e3 << "ms\n"
         << "forward process : " << forward_time_microseconds / 1.0e3 << "ms\n"
         << "unknown consumption : " << (total_time_microseconds -
                                                 (derivative_time_microseconds + backward_time_microseconds
                                                 + forward_time_microseconds)) / 1.0e3 << "ms\n";

   return;
}

template<typename T, unsigned M, unsigned N>
void ILQRSolver<T, M, N>::UpdateDualsAndPenalties(ILQRSolverState<T, M, N> *ilqr_state) const {
    States x = ilqr_state->x_seq();
    Controls u = ilqr_state->u_seq();
    for (auto &cons_ptr : Problem().GetCostUnionPtr()->GetEqConstraints()) {
        cons_ptr->UpdateDual();
        cons_ptr->UpdatePenalty();
    }

    for (auto &cons_ptr : Problem().GetCostUnionPtr()->GetIneqConstraints()) {
        cons_ptr->UpdateDual();
        cons_ptr->UpdatePenalty();
    }
}

template<typename T, unsigned int M, unsigned int N>
bool ILQRSolver<T, M, N>::ForwardProcess(const States &x_old, const Controls &u_old, double *new_cost, double *dcost,
                                         double *expected, double *ratio, ILQRSolverState<T, M, N> *ilqr_state) const {
    bool is_fwd_pass_done = false;
    for (const auto &alpha : alpha_vec_) {
        Controls u_plu_seq_feedforward = ilqr_state->u_seq_;
        for (size_t j = 0; j < ilqr_state->u_seq_.size(); j++) {
            u_plu_seq_feedforward[j] += ilqr_state->k_[j] * alpha;
        }

        *new_cost = RollOut(ilqr_state->x0_, u_plu_seq_feedforward, ilqr_state);
//        double f_t = ftimer.elapsed_microseconds();
//        cout << "roll out : " << f_t << "ms\n";

        *dcost = ilqr_state->cost_ - *new_cost;
        *expected = -alpha * (ilqr_state->dV_(0) + alpha * ilqr_state->dV_(1));

        if (*expected > 0) {
            *ratio = *dcost / *expected;
        } else {
            *ratio = sgn(*dcost);
            ilqr_state->status_ = SolverStatus::CostIncrease;
            cout << "Warning: non-positive expected reduction\n";
        }

        if (*ratio > kRatioMin_ and *ratio < krhoMax_) {
            is_fwd_pass_done = true;
            DecreaseRho(ilqr_state->drho_, ilqr_state->rho_);
            ilqr_state->cost_ = *new_cost; // accept step
            break;
        }
//        double duration = timer.elapsed_microseconds();
//        cout << "line search cost : " << duration << "ms\n";
        // Line search failed for this alpha, so reset x_seq
        // and u_seq since they are updated in roll_out()
        ilqr_state->x_seq_ = x_old;
        ilqr_state->u_seq_ = u_old;
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
double ILQRSolver<T, M, N>::RollOut(const State &x0, const Controls &u, ILQRSolverState<T, M, N> *ilqr_state) const {
    double total_cost = 0;

    State x_curr = x0;
    Control u_curr;

    const uint32_t step_nums = step_sizes_.size();
    States x_new(step_nums + 1);
    x_new[0] = x0;
    double timer = 0.0;
    Problem().GetModel()->SetTimer(timer);
    for (uint32_t t = 0; t < step_nums + 1; t++) {
        u_curr = u[t];
        if (ilqr_state->K_.size() > 0)
            // apply LQR control gains after first iteration
            u_curr += ilqr_state->K_[t] * (x_new[t] - ilqr_state->x_seq_[t]);

        ilqr_state->u_seq_[t] = u_curr; // no clamping since no control limit
        total_cost += Problem().Evaluate(t, x_curr, u_curr);
        x_curr = Problem().GetModel()->ForwardCalculation(x_curr, u_curr, step_sizes_[t]);
        if (use_timer_)
            Problem().GetModel()->UpdateTimer(step_sizes_[t]);
        if (t < step_nums)
            x_new[t + 1] = x_curr;
    }

    ilqr_state->x_seq_ = x_new;
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
void ILQRSolver<T, M, N>::BackwardProcess(ILQRSolverState<T, M, N> *ilqr_state) const {
    const uint32_t step_nums = step_sizes().size();
    if (step_nums == 0) {
        //    NLOGW("iLQR backward_pass: planning horizon length is zero");
        cout << "iLQR backward_pass: planning horizon length is zero\n";
        return;
    }

    while (true) {
        bool is_cost_psd = true;
        ilqr_state->dV_.setZero();
        int max_reg_count = 0;
        for (int i = static_cast<int>(step_nums - 1); i >= 0; i--) {
            // backward from the end
            ilqr_state->Qx_ = ilqr_state->lx_[i] + (ilqr_state->fx_[i].transpose() *
                                                    ilqr_state->Vx_[i + 1]) + ilqr_state->cx_[i];
            ilqr_state->Qu_ = ilqr_state->lu_[i] + (ilqr_state->fu_[i].transpose() *
                                                    ilqr_state->Vx_[i + 1]) + ilqr_state->cu_[i];
            ilqr_state->Qxx_ =
                    ilqr_state->lxx_[i] + (ilqr_state->fx_[i].transpose() *
                                           ilqr_state->Vxx_[i + 1] * ilqr_state->fx_[i])
                                           + ilqr_state->cxx_[i];
            ilqr_state->Qux_ = ilqr_state->lxu_[i].transpose() +
                               (ilqr_state->fu_[i].transpose() *
                                ilqr_state->Vxx_[i + 1] * ilqr_state->fx_[i])
                                + ilqr_state->cxu_[i].transpose();
            ilqr_state->Quu_ =
                    ilqr_state->luu_[i] + (ilqr_state->fu_[i].transpose() *
                                           ilqr_state->Vxx_[i + 1] * ilqr_state->fu_[i])
                                           + ilqr_state->cuu_[i];

            // Similar to equations 10a and 10b in [Tassa 2012]. Note that
            // regularization is different from the paper but consistent
            // with Tassa's Matlab implementation)
            // improved regularization QuuF = Quu + mu * I
            ilqr_state->QuuF_ = ilqr_state->luu_[i] +
                                ilqr_state->fu_[i].transpose() *
                                (ilqr_state->Vxx_[i + 1] + (ilqr_state->rho_ * MatrixLXX::Identity()))
                                * ilqr_state->fu_[i];
//            ilqr_state->QuuF_ = ilqr_state->Quu_ +
//                                ilqr_state->lambda_ * MatrixLUU::Identity();
//            total_update_derivative += update_derivative.elapsed_microseconds();
//            cout << "update derivatives in backward process : " << update_derivative.elapsed_microseconds() << "\n";


            // The following method is only efficient when n <= 4
            MatrixLUU QuuF_inv;
            double QuuF_det;
            bool QuuF_invertible;
            double tol = 1.0e-6;
            auto start = std::chrono::steady_clock::now();
            ilqr_state->QuuF_.computeInverseAndDetWithCheck(QuuF_inv, QuuF_det,
                                                            QuuF_invertible, tol);
//            auto end = std::chrono::steady_clock::now();
//            double d = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1.0e3;
//            cout << "Compute inverse : " << d << " ms" << "\n";
            if (!QuuF_invertible) {
                //        NLOGW("backward pass diverged at step %d", i);
                is_cost_psd = false;
                break; // QuuF is not positive definite, need to increase lambda
            }

            // Compute the control gain
//            update_derivative.start();
            ilqr_state->k_i_ = -QuuF_inv * ilqr_state->Qu_;
            ilqr_state->K_i_ = -QuuF_inv * ilqr_state->Qux_;

            // Update cost-to-go approximation. Equations 11 in [Tassa 2012]
            ilqr_state->dV_(0) += ilqr_state->k_i_.transpose() * ilqr_state->Qu_;
            ilqr_state->dV_(1) += 0.5 * (ilqr_state->k_i_.transpose() *
                                         ilqr_state->Quu_ * ilqr_state->k_i_)[0];
            ilqr_state->Vx_[i] =
                    ilqr_state->Qx_ +
                    ilqr_state->K_i_.transpose() * ilqr_state->Quu_ * ilqr_state->k_i_ +
                    ilqr_state->K_i_.transpose() * ilqr_state->Qu_ +
                    ilqr_state->Qux_.transpose() * ilqr_state->k_i_;
            ilqr_state->Vxx_[i] =
                    ilqr_state->Qxx_ +
                    ilqr_state->K_i_.transpose() * ilqr_state->Quu_ * ilqr_state->K_i_ +
                    ilqr_state->K_i_.transpose() * ilqr_state->Qux_ +
                    ilqr_state->Qux_.transpose() * ilqr_state->K_i_;
            ilqr_state->Vxx_[i] =
                    0.5 * (ilqr_state->Vxx_[i] + ilqr_state->Vxx_[i].transpose());

            // save controls/gains
            ilqr_state->k_[i] = ilqr_state->k_i_;
            ilqr_state->K_[i] = ilqr_state->K_i_;
//            total_calc_V += update_derivative.elapsed_microseconds();
        }

        if (is_cost_psd) {
            break;
        } else {
            IncreaseRho(ilqr_state->drho_, ilqr_state->rho_);
            if (ilqr_state->rho_ > krhoMax_) {
                ++max_reg_count;
                if (max_reg_count > kMaxRegCount_) {
                    ilqr_state->status_ = SolverStatus::RegularizationFailed;
                    break;
                }
            }
        }
//        DecreaseRho(ilqr_state->drho_, ilqr_state->rho_);
    }
//    cout << "update derivatives in backward process : " << total_update_derivative << "\n";
//    cout << "calc V function : " << total_calc_V << "\n";
}

template<typename T, unsigned int M, unsigned int N>
void ILQRSolver<T, M, N>::IncreaseRho(double &drho, double &rho) const {
    drho = std::max(drho * krhoFactor_, krhoFactor_);
    rho = std::max(rho * drho, krhoMin_);
    rho = std::min(rho, krhoMax_);
}

template<typename T, unsigned int M, unsigned int N>
void ILQRSolver<T, M, N>::DecreaseRho(double &drho, double &rho) const {
    drho = std::min(drho / krhoFactor_, 1 / krhoFactor_);
    rho = std::max(rho * drho, krhoMin_);
    rho = std::min(rho, krhoMax_);
}

template<typename T, unsigned int M, unsigned int N>
double ILQRSolver<T, M, N>::GetGradientNorm(const VecUs &l, const Controls &u) const {
    std::vector<double> vals(l.size());
    for (uint32_t i = 0; i < l.size(); i++) {
        VectorXd v = l[i].cwiseAbs().array() / (u[i].cwiseAbs().array() + 1);
        vals[i] = v.maxCoeff();
    }
    return std::accumulate(vals.begin(), vals.end(), 0.0) / vals.size();
}

template<typename T, unsigned M, unsigned N>
double ILQRSolver<T, M, N>::GetMaxViolation() const {
    return Problem().GetCostUnionPtr()->GetMaxViolation();
}

template<typename T, unsigned M, unsigned N>
void ILQRSolver<T, M, N>::InitializeILQRSolverState(ILQRSolverState<T, M, N> *ilqr_state) const {
    const uint32_t step_nums = step_sizes().size();
    States x = ilqr_state->x_seq();
    Controls u = ilqr_state->u_seq();
    CalcKinematicsDerivatives(x, u, &(ilqr_state->fx_), &(ilqr_state->fu_));
    for (uint32_t t = 0; t < step_nums; ++t) {
        (void) Problem().GetCostUnionPtr()->Evaluate(t, x[t], u[t]);
        Problem().GetCostUnionPtr()->CalcCostGradient(t, x[t], u[t],
                                                      ilqr_state->lx_[t], ilqr_state->lu_[t]);
        Problem().GetCostUnionPtr()->CalcAugLagConstraintGradient(t, x[t], u[t],
                                                                  ilqr_state->cx_[t], ilqr_state->cu_[t]);
        Problem().GetCostUnionPtr()->CalcCostHessian(t, x[t], u[t],
                                                     ilqr_state->lxx_[t],
                                                     ilqr_state->luu_[t],
                                                     ilqr_state->lxu_[t]);
        Problem().GetCostUnionPtr()->CalcAugLagConstraintHessian(t, x[t], u[t],
                                                                 ilqr_state->cxx_[t],
                                                                 ilqr_state->cuu_[t],
                                                                 ilqr_state->cxu_[t]);

    }
    Problem().GetCostUnionPtr()->CalcTerminalCostToGo(x[step_nums]);
    ilqr_state->Vx_[step_nums] = Problem().GetCostUnionPtr()->GetFinalCostToGoGradient();
    ilqr_state->Vxx_[step_nums] = Problem().GetCostUnionPtr()->GetFinalCostToGoHessian();
}

template<typename T, unsigned int M, unsigned int N>
void ILQRSolver<T, M, N>::CalcKinematicsDerivatives(const States &x, const Controls &u, MatrixLXXs *f_x,
                                                    MatrixLXUs *f_u) const {
    const uint32_t step_nums = step_sizes().size();
    for (uint32_t t = 0; t < step_nums; t++) {
        (*f_x)[t] =
                Problem().GetModel()->JacobianX(x[t], u[t], step_sizes_[t]);
        (*f_u)[t] =
                Problem().GetModel()->JacobianU(x[t], u[t], step_sizes_[t]);
    }
}

template<typename T, unsigned int M, unsigned int N>
const int ILQRSolver<T, M, N>::kMaxIter_ = 20;

template<typename T, unsigned int M, unsigned int N>
const double ILQRSolver<T, M, N>::kTolFun_ = 1e-6;

template<typename T, unsigned int M, unsigned int N>
const double ILQRSolver<T, M, N>::kTolGrad_ = 1e-6;

template<typename T, unsigned int M, unsigned int N>
const double ILQRSolver<T, M, N>::krhoFactor_ = 1.6;

template<typename T, unsigned int M, unsigned int N>
const double ILQRSolver<T, M, N>::krhoMax_ = 1e8;

template<typename T, unsigned int M, unsigned int N>
const double ILQRSolver<T, M, N>::krhoMin_ = 1e-8;

template<typename T, unsigned int M, unsigned int N>
const double ILQRSolver<T, M, N>::krhoMinGrad_ = 1e-5;

template<typename T, unsigned int M, unsigned int N>
const double ILQRSolver<T, M, N>::kRatioMin_ = 1.0e-8;

template<typename T, unsigned M, unsigned N>
const double ILQRSolver<T, M, N>::kRationMax_ = 10.0;

template<typename T, unsigned M, unsigned N>
const double ILQRSolver<T, M, N>::kViolationTol_ = 1e-2;

template<typename T, unsigned M, unsigned N>
const int ILQRSolver<T, M, N>::kMaxRegCount_ = 20;

template<typename T, unsigned int M, unsigned int N>
const std::array<double, 11> ILQRSolver<T, M, N>::alpha_vec_{
        1.0000, 0.5012, 0.2512, 0.1259, 0.0631, 0.0316,
        0.0158, 0.0079, 0.0040, 0.0020, 0.0010};

template class ILQRSolver<double, 5, 1>;

template class ILQRSolver<double, 4, 2>;

template class ILQRSolver<double, 6, 2>;