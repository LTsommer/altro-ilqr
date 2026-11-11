//
// Created by LTsommer on 2024/6/25.
//

#ifndef ALILQR_ALTRO_SOLVER_HPP
#define ALILQR_ALTRO_SOLVER_HPP

#include "alilqr_solver.h"
#include "cost_calc.h"
#include "../clock.hpp"
#include <iostream>
#include <algorithm>
#include <vector>

using std::cout;
using std::endl;

enum class JacSize { None, Gradient, Jacobian };

template<typename... Args>
void print(const Args &...args) {
    using arr = int[];
    (void)arr{0, (std::cout << args << ", ", 0)...};
    cout << "\n";
    return;
}

template<typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
    for (const auto &element : vec)
        os << element << ' ';
    os << "\n";
    return os;
}

template<typename T, unsigned M, unsigned N>
class ALTROSolver {
public:
    using CostUnionType = CostUnion<T, M, N>;
    template<class ConsType>
    using ConstraintValues = typename CostUnionType::template ConstraintValues<ConsType>;
    constexpr static double kDefaultConstraintEpsilon = 1.0e-3;
    constexpr static double kDefaultProjectionTrigger = 0.1;
    constexpr static double kTol = 1.0e-6;
    constexpr static double kConvRateTol = 0.9;
    constexpr static int kMaxLineSearchIter = 20;
    constexpr static int kMaxInnerIter = 10;
    constexpr static int kMaxOuterIter = 10;

public:
    OCP_VARIABLES(T, M, N)
    explicit ALTROSolver(std::unique_ptr<OCPInterface<T, M, N>> &&ocp_interface, const vector<double> &steps)
        : step_sizes_(steps) {
        //        problem_.reset(std::move(solver->ProblemPtr()));
        steps_num_ = steps.size();
        solver_.reset(new ALILQRSolver<T, M, N>(std::move(ocp_interface), steps));
    };

    virtual ~ALTROSolver() = default;

    void Solve(const State &x0, const Controls &u0, States &x_res_seq, Controls &u_res_seq) {
        // The AL-iLQR summary is printed below, after both clocks have stopped.
        solver_->SetPrintSummary(false);
        StopWatch timer;
        timer.start();
        States alilqr_x;
        Controls alilqr_u;
        solver_->Solve(x0, u0, alilqr_x, alilqr_u);
        const double alilqr_violation = solver_->GetMaxViolation();
        StopWatch proj_timer;
        proj_timer.start();
        // Polishing is a local correction: from a trajectory still far from feasible, projecting
        // onto the linearized constraints can change it a lot and raise the cost, so it only
        // runs once AL-iLQR is close.
        const bool project = alilqr_violation < projection_trigger_;
        const int proj_steps = project ? ProjectTrajectory(alilqr_x, alilqr_u) : 0;
        const double proj_ms = proj_timer.elapsed_microseconds() / 1.0e3;
        x_res_seq = alilqr_x;
        u_res_seq = alilqr_u;
        solver_->Problem().GetCostUnionPtr()->EvaluateConstraints(x_res_seq, u_res_seq);
        const double dyn_residual = MaxDynamicsResidual(x_res_seq, u_res_seq);
        const double total_ms = timer.elapsed_microseconds() / 1.0e3;

        cout << "AL-iLQR Solver Status : ";
        solver_->ShowSolverState();
        cout << "AL-iLQR iterations : " << solver_->TotalIterations() << ", time : " << solver_->SolveTimeMs()
             << " ms\n";
        cout << "AL-iLQR max_violation : " << alilqr_violation << "\n";
        if (project)
            cout << "ALTRO projection : " << proj_steps << " steps, " << proj_ms << " ms\n";
        else
            cout << "ALTRO projection skipped: AL-iLQR violation above " << projection_trigger_ << "\n";
        cout << "ALTRO solver max violation : " << GetMaxViolation() << "\n";
        for (const auto &cons_ptr : solver_->Problem().GetCostUnionPtr()->GetImmutableEqConstraints())
            cout << "  " << cons_ptr->GetName() << " max violation : " << cons_ptr->GetMaxViolation() << "\n";
        for (const auto &cons_ptr : solver_->Problem().GetCostUnionPtr()->GetImmutableIneqConstraints())
            cout << "  " << cons_ptr->GetName() << " max violation : " << cons_ptr->GetMaxViolation() << "\n";
        cout << "ALTRO dynamics residual : " << dyn_residual << "\n";
        cout << "altro solver consumes : " << total_ms << " ms\n";
    }

    void SetVerbose(const bool verbose) { solver_->SetVerbose(verbose); }

    // During polishing, satisfied constraints whose first-order distance to the boundary
    // (g / |grad g|, in z-space) is below this value are held in place.
    void SetActiveTolerance(const double dist) { active_dist_ = dist; }

    // Polishing runs only when the AL-iLQR violation is below this value.
    void SetProjectionTrigger(const double violation) { projection_trigger_ = violation; }

    void SetUpdateConfig(const bool update) { solver_->SetUpdateConfig(update); }

    MatrixXd CalcAllViolations(States &x,
                               Controls &u,
                               std::unordered_map<std::string, vector<double>> &violation_record) const {
        vector<double> violations;
        int rows = 0;
        for (const auto &cons_ptr : solver_->Problem().GetCostUnionPtr()->GetImmutableEqConstraints()) {
            vector<double> vio;
            cout << cons_ptr->GetName() << "\n";
            for (int t = 0; t < steps_num_; ++t) {
                bool success = cons_ptr->Evaluate(t, x[t], u[t]);
                if (success) {
                    double violation = cons_ptr->ConsVal();
                    violations.emplace_back(violation);
                    vio.emplace_back(violation);
                }
            }
            std::string name = solver_->Problem().GetCostUnionPtr()->GetTypeNames().at(cons_ptr->GetTypeIndex());
            violation_record[name] = vio;
            ++rows;
        }

        for (const auto &cons_ptr : solver_->Problem().GetCostUnionPtr()->GetImmutableIneqConstraints()) {
            vector<double> vio;
            cout << cons_ptr->GetName() << "\n";
            for (int t = 0; t < steps_num_; ++t) {
                bool success = cons_ptr->Evaluate(t, x[t], u[t]);
                if (success) {
                    double violation = cons_ptr->ConsVal();
                    violations.emplace_back(violation);
                    vio.emplace_back(violation);
                }
            }
            std::string name = solver_->Problem().GetCostUnionPtr()->GetTypeNames().at(cons_ptr->GetTypeIndex());
            violation_record[name] = vio;
            ++rows;
        }

        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Violations(
                violations.data(), rows, steps_num_);
        return Violations;
    }

private:
    // ------------------------------------------------------------------------------
    // Solution polishing (ALTRO, Howell et al. 2019, Sec. IV-C).
    // The trajectory is projected onto the active constraints, weighted by the cost Hessian H:
    //     min 1/2 dz^T H dz + 1/(2 delta) |C dz + c|^2
    //     s.t. dx_{t+1} = A_t dx_t + B_t du_t,   dx_0 = 0
    // where C dz + c = 0 are the linearized active constraint rows. The dynamics are kept
    // exactly; the constraints enter as a 1/delta penalty, which for small delta approaches the
    // exact projection and, when the rows cannot all be met (more active rows than controls, or
    // conflicting rows), gives a damped least-squares compromise. delta is adapted like
    // Levenberg-Marquardt: it shrinks after an accepted step and grows after a rejected one.
    //
    // The problem is a time-varying LQR, so it is solved by a Riccati recursion in O(T) with no
    // global matrix. The recursion also yields feedback gains K_t, and each step is applied by a
    // nonlinear rollout from x_0,
    //     u_t' = u_t + alpha k_t + K_t (x_t' - x_t),   x_{t+1}' = f(x_t', u_t'),
    // so every iterate (including the result) satisfies the dynamics exactly.
    // ------------------------------------------------------------------------------
    constexpr static double kHessianReg = 1.0e-6; // relative to the normalized H, keeps it positive definite
    constexpr static double kDampingInit = 1.0e-6;
    constexpr static double kDampingMin = 1.0e-10;
    constexpr static int kMaxDampingTries = 8;
    constexpr static int kMaxProjectionLineSearch = 6;
    constexpr static int kMaxProjectionIter = 30;
    constexpr static double kMinProjectionProgress = 1.0e-3; // relative decrease per step

    // Per-step blocks of the cost Hessian (cost terms only, no AL terms), scaled to a mean
    // diagonal of 1. The projection does not depend on a constant scale of H, but the
    // regularization and damping do: with raw cost weights (H ~ 1e5..1e7) a fixed damping would
    // dominate and the steps would barely reduce the violation.
    struct StageHessian {
        MatrixLXX xx;
        MatrixLUU uu;
        MatrixLXU xu;
    };

    std::vector<StageHessian> BuildCostHessian(const States &x, const Controls &u) const {
        std::vector<StageHessian> h(steps_num_ + 1);
        double diag_sum = 0.0;
        for (int t = 0; t <= steps_num_; ++t) {
            const Control u_t = t < steps_num_ ? u[t] : Control::Zero();
            h[t].xx.setZero();
            h[t].uu.setZero();
            h[t].xu.setZero();
            for (const auto &ct : solver_->Problem().GetCostUnionPtr()->GetCostTerms()) {
                MatrixLXX cur_lxx = MatrixLXX::Zero();
                MatrixLUU cur_luu = MatrixLUU::Zero();
                MatrixLXU cur_lxu = MatrixLXU::Zero();
                if (ct->Hessian(t, x[t], u_t, cur_lxx, cur_luu, cur_lxu)) {
                    h[t].xx += cur_lxx;
                    h[t].uu += cur_luu;
                    h[t].xu += cur_lxu;
                }
            }
            if (t >= 1) diag_sum += h[t].xx.diagonal().cwiseAbs().sum(); // x_0 is fixed
            if (t < steps_num_) diag_sum += h[t].uu.diagonal().cwiseAbs().sum();
        }
        const double scale = std::max(diag_sum / (steps_num_ * static_cast<double>(M + N)), 1e-12);
        for (auto &b : h) {
            b.xx = b.xx / scale + kHessianReg * MatrixLXX::Identity();
            b.uu = b.uu / scale + kHessianReg * MatrixLUU::Identity();
            b.xu /= scale;
        }
        return h;
    }

    // One linearized constraint row: residual + gx^T dx_t + gu^T du_t = 0.
    // distance = g / |grad g| is the first-order signed distance to the boundary in z-space
    // (> 0 means violated). It is only used to pick the active set, so the choice does not
    // depend on each constraint's units; the row itself stays in the constraint's own units,
    // matching the violation measure used by the line search.
    struct ActiveRow {
        int t;
        double residual;
        double distance;
        VecX gx;
        VecU gu;
    };

    // Largest violation of the original (hinge) constraints: |c| for equalities, max(c, 0)
    // for inequalities. This is what the solver reports and what the line search compares.
    double MaxConstraintViolation(const States &x, const Controls &u) const {
        auto &cost_union = solver_->Problem().GetCostUnionPtr();
        double v = 0.0;
        for (int t = 0; t <= steps_num_; ++t) {
            const Control u_t = t < steps_num_ ? u[t] : Control::Zero();
            for (auto &c : cost_union->GetEqConstraints())
                if (c->Evaluate(t, x[t], u_t)) v = std::max(v, std::fabs(c->ConsVal()));
            for (auto &c : cost_union->GetIneqConstraints())
                if (c->Evaluate(t, x[t], u_t)) v = std::max(v, c->ConsVal());
        }
        return v;
    }

    double MaxDynamicsResidual(const States &x, const Controls &u) const {
        const auto &model = solver_->Problem().GetModel();
        double v = 0.0;
        for (int t = 0; t < steps_num_; ++t)
            v = std::max(v,
                         (x[t + 1] - model->ForwardCalculation(x[t], u[t], step_sizes_[t]))
                                 .template lpNorm<Eigen::Infinity>());
        return v;
    }

    // Active set:
    //  - equalities: always;
    //  - violated inequalities (raw sub-constraint g > tol, or the hinge value c > tol when
    //    a constraint does not expose sub-constraints): residual g, i.e. pull back onto the boundary;
    //  - satisfied inequalities within active_dist_ of the boundary (distance g / |grad g|):
    //    residual 0, i.e. do not move them to first order, so that fixing one constraint
    //    does not push a neighbouring one out.
    std::vector<ActiveRow> CollectActiveRows(const States &x, const Controls &u) const {
        auto &cost_union = solver_->Problem().GetCostUnionPtr();
        std::vector<ActiveRow> rows;
        auto make_row = [](int t, double g, const VecX &gx, const VecU &gu, double residual) {
            const double norm = std::max(std::sqrt(gx.squaredNorm() + gu.squaredNorm()), 1e-12);
            return ActiveRow{t, residual, g / norm, gx, gu};
        };
        RawConstraintRows<T, M, N> raw;
        for (int t = 0; t <= steps_num_; ++t) {
            const Control u_t = t < steps_num_ ? u[t] : Control::Zero();
            for (auto &c : cost_union->GetEqConstraints()) {
                if (!c->Evaluate(t, x[t], u_t)) continue;
                c->Gradient(t, x[t], u_t);
                rows.push_back(make_row(t, c->ConsVal(), c->GradientX(), c->GradientU(), c->ConsVal()));
            }
            for (auto &c : cost_union->GetIneqConstraints()) {
                if (!c->Evaluate(t, x[t], u_t)) continue;
                raw.clear();
                if (c->RawConstraints(t, x[t], u_t, raw)) {
                    for (const auto &r : raw) {
                        ActiveRow row = make_row(t, r.g, r.gx, r.gu, std::max(r.g, 0.0));
                        if (r.g > kTol or row.distance > -active_dist_) rows.push_back(std::move(row));
                    }
                } else if (c->ConsVal() > kTol) {
                    c->Gradient(t, x[t], u_t);
                    rows.push_back(make_row(t, c->ConsVal(), c->GradientX(), c->GradientU(), c->ConsVal()));
                }
            }
        }
        return rows;
    }

    // Quadratic model of one step: 1/2 [dx; du]^T [Qxx Qux^T; Qux Quu] [dx; du] + qx^T dx + qu^T du.
    struct StageQuadratic {
        MatrixLXX Qxx;
        MatrixLUU Quu;
        Eigen::Matrix<T, N, M> Qux;
        VecX qx;
        VecU qu;
    };

    // Cost Hessian plus the 1/delta penalty of the active rows, per step.
    std::vector<StageQuadratic> BuildStageQuadratics(const std::vector<StageHessian> &h,
                                                     const std::vector<ActiveRow> &rows,
                                                     const double delta) const {
        std::vector<StageQuadratic> q(steps_num_ + 1);
        for (int t = 0; t <= steps_num_; ++t) {
            q[t].Qxx = h[t].xx;
            q[t].Quu = h[t].uu;
            q[t].Qux = h[t].xu.transpose();
            q[t].qx.setZero();
            q[t].qu.setZero();
        }
        const double w = 1.0 / delta;
        for (const auto &r : rows) {
            auto &s = q[r.t];
            s.Qxx += w * r.gx * r.gx.transpose();
            s.Quu += w * r.gu * r.gu.transpose();
            s.Qux += w * r.gu * r.gx.transpose();
            s.qx += w * r.residual * r.gx;
            s.qu += w * r.residual * r.gu;
        }
        return q;
    }

    // Riccati recursion of the time-varying LQR; du_t = k_t + K_t dx_t. Returns false if a
    // step's control Hessian is not positive definite.
    bool RiccatiBackward(const States &x,
                         const Controls &u,
                         const std::vector<StageQuadratic> &q,
                         std::vector<VecU> &k,
                         std::vector<Eigen::Matrix<T, N, M>> &K) const {
        const auto &model = solver_->Problem().GetModel();
        k.assign(steps_num_, VecU::Zero());
        K.assign(steps_num_, Eigen::Matrix<T, N, M>::Zero());
        MatrixLXX V = q[steps_num_].Qxx;
        VecX v = q[steps_num_].qx;
        for (int t = steps_num_ - 1; t >= 0; --t) {
            const auto A = model->JacobianX(x[t], u[t], step_sizes_[t]);
            const auto B = model->JacobianU(x[t], u[t], step_sizes_[t]);
            const MatrixLXX Sxx = q[t].Qxx + A.transpose() * V * A;
            const MatrixLUU Suu = q[t].Quu + B.transpose() * V * B;
            const Eigen::Matrix<T, N, M> Sux = q[t].Qux + B.transpose() * V * A;
            const VecX sx = q[t].qx + A.transpose() * v;
            const VecU su = q[t].qu + B.transpose() * v;
            const Eigen::LLT<MatrixLUU> llt(Suu);
            if (llt.info() != Eigen::Success) return false;
            K[t] = -llt.solve(Sux);
            k[t] = -llt.solve(su);
            V = Sxx + Sux.transpose() * K[t];
            V = 0.5 * (V + V.transpose());
            v = sx + Sux.transpose() * k[t];
        }
        return true;
    }

    // Applies the step by a nonlinear rollout from x_0 with the feedback gains.
    void Rollout(const States &x,
                 const Controls &u,
                 const std::vector<VecU> &k,
                 const std::vector<Eigen::Matrix<T, N, M>> &K,
                 const double alpha,
                 States &x_new,
                 Controls &u_new) const {
        const auto &model = solver_->Problem().GetModel();
        x_new.resize(steps_num_ + 1);
        u_new.resize(steps_num_);
        x_new[0] = x[0];
        for (int t = 0; t < steps_num_; ++t) {
            u_new[t] = u[t] + alpha * k[t] + K[t] * (x_new[t] - x[t]);
            x_new[t + 1] = model->ForwardCalculation(x_new[t], u_new[t], step_sizes_[t]);
        }
    }

    // Returns the number of accepted steps.
    int ProjectTrajectory(States &x, Controls &u) const {
        // Start from a dynamically consistent trajectory (AL-iLQR already gives one; this only
        // removes round-off) so the recursion can assume zero dynamics residuals.
        {
            const std::vector<VecU> k0(steps_num_, VecU::Zero());
            const std::vector<Eigen::Matrix<T, N, M>> K0(steps_num_, Eigen::Matrix<T, N, M>::Zero());
            States x_r;
            Controls u_r;
            Rollout(x, u, k0, K0, 0.0, x_r, u_r);
            x = std::move(x_r);
        }
        const std::vector<StageHessian> h = BuildCostHessian(x, u);
        double delta = kDampingInit;
        int accepted_steps = 0;
        double v_prev = std::numeric_limits<double>::infinity();

        for (int iter = 0; iter < kMaxProjectionIter; ++iter) {
            const double v = MaxConstraintViolation(x, u);
            if (solver_->Verbose()) cout << "ALTRO projection iter " << iter << ": max violation = " << v << "\n";
            if (v < kTol) break;
            // Conflicting rows: the steps converge to a least-squares compromise and stop
            // making progress.
            if (v > (1.0 - kMinProjectionProgress) * v_prev) {
                if (solver_->Verbose()) cout << "ALTRO projection: violation stalled, stop\n";
                break;
            }
            v_prev = v;

            const std::vector<ActiveRow> rows = CollectActiveRows(x, u);
            bool accepted = false;
            for (int tries = 0; tries < kMaxDampingTries and !accepted; ++tries) {
                std::vector<VecU> k;
                std::vector<Eigen::Matrix<T, N, M>> K;
                if (RiccatiBackward(x, u, BuildStageQuadratics(h, rows, delta), k, K)) {
                    // Backtracking on the max constraint violation.
                    double alpha = 1.0;
                    for (int ls = 0; ls < kMaxProjectionLineSearch; ++ls) {
                        States x_n;
                        Controls u_n;
                        Rollout(x, u, k, K, alpha, x_n, u_n);
                        if (MaxConstraintViolation(x_n, u_n) < v) {
                            x = std::move(x_n);
                            u = std::move(u_n);
                            accepted = true;
                            break;
                        }
                        alpha *= 0.5;
                    }
                }
                delta = accepted ? std::max(delta / 10.0, kDampingMin) : delta * 100.0;
            }
            if (!accepted) {
                if (solver_->Verbose()) cout << "ALTRO projection: no improving step found, stop\n";
                break;
            }
            ++accepted_steps;
        }
        return accepted_steps;
    }

    const double GetMaxViolation() const { return solver_->Problem().GetCostUnionPtr()->GetMaxViolation(); }

private:
    std::unique_ptr<ALILQRSolver<T, M, N>> solver_;
    std::vector<double> step_sizes_;
    int steps_num_;
    double active_dist_ = 0.05;
    double projection_trigger_ = kDefaultProjectionTrigger;
};

#endif //ALILQR_ALTRO_SOLVER_HPP
