//
// Created by LTsommer on 2024/6/25.
//

#ifndef ALILQR_ALTRO_SOLVER_HPP
#define ALILQR_ALTRO_SOLVER_HPP

#include "alilqr_solver.h"
#include "cost_calc.h"
#include "../clock.hpp"
#include <iostream>

using std::cout;
using std::endl;

enum class JacSize {
    None,
    Gradient,
    Jacobian
};

template<typename T, unsigned M, unsigned N>
class ALTROSolver {
public:
    using CostUnionType = CostUnion<T, M, N>;
    template<class ConsType>
    using ConstraintValues = typename CostUnionType::template ConstraintValues<ConsType>;
    constexpr static double kDefaultConstraintEpsilon = 1.0e-3;
    constexpr static double kTol = 1.0e-6;
    constexpr static double kConvRateTol = 1;
    constexpr static int kMaxLineSearchIter = 20;
    constexpr static int kMaxInnerIter = 10;
    constexpr static int kMaxOuterIter = 10;
public:
    OCP_VARIABLES(T, M, N)
    explicit ALTROSolver(std::unique_ptr<OCPInterface<T, M, N>> &&ocp_interface,
                        const vector<double> &steps)
            : step_sizes_(steps)
    {
//        problem_.reset(std::move(solver->ProblemPtr()));
        steps_num_ = steps.size();
        solver_.reset(new ILQRSolver<T, M, N>(std::move(ocp_interface), steps));
    };

    virtual ~ALTROSolver() = default;

    void Solve(const State &x0, const Controls &u0,
               States &x_res_seq, Controls &u_res_seq) {
        StopWatch timer;
        timer.start();
        States alilqr_x;
        Controls alilqr_u;
        solver_->Solve(x0, u0, alilqr_x, alilqr_u);
        States x_old = alilqr_x;
        Controls u_old = alilqr_u;
        cout << "AL-iLQR Solver Status : ";
        solver_->ShowSolverState();
        cout << "AL-iLQR max_violation : " << solver_->GetMaxViolation() << "\n";
        vector<vector<double>> all_violations;
        StopWatch proj_timer;
        proj_timer.start();
        double proj_consumption = 0.0;
        for (int t = 0; t < steps_num_; ++t) {
            State x = alilqr_x[t];
            Control u = alilqr_u[t];
            Projection(t, x, u, alilqr_x[t], alilqr_u[t]);
        }
        proj_consumption += proj_timer.elapsed_microseconds() / 1.0e3;
        x_res_seq = alilqr_x;
        u_res_seq = alilqr_u;
//        for (int t = 0; t < steps_num_; ++t) {
//            cout << (x_res_seq[t] - x_old[t]).template lpNorm<1>() << endl;
//            cout << (u_res_seq[t] - u_old[t]).template lpNorm<1>() << endl;
//        }
        solver_->Problem().GetCostUnionPtr()->EvaluateConstraints(x_res_seq, u_res_seq);
        cout << "ALTRO solver max violation : " << GetMaxViolation() << endl;
        double d = timer.elapsed_microseconds() / 1.0e3;
        cout << "altro Projection Process : " << proj_consumption << " ms\n";
        cout << "altro solver consumes : " << d << " ms\n";
    }

    MatrixXd CalcAllViolations(States &x, Controls &u) const {
        vector<double> violations;
        int rows = 0;
        for (const auto &cons_ptr : solver_->Problem().GetCostUnionPtr()->GetImmutableEqConstraints()) {
            for (int t = 0; t < steps_num_; ++t) {
                bool success = cons_ptr->Evaluate(t, x[t], u[t]);
                if (success) {
                    double violation = cons_ptr->ConsVal();
                    violations.emplace_back(violation);
                }
            }
            ++rows;
        }

        for (const auto &cons_ptr : solver_->Problem().GetCostUnionPtr()->GetImmutableIneqConstraints()) {
            for (int t = 0; t < steps_num_; ++t) {
                bool success = cons_ptr->Evaluate(t, x[t], u[t]);
                if (success) {
                    double violation = cons_ptr->ConsVal();
                    violations.emplace_back(violation);
                }
            }
            ++rows;
        }

        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        Violations(violations.data(), rows, steps_num_);
        return Violations;
    }

private:
    void GetCostHessianAt(const int step, const State &x, const Control &u) {
        MatrixLXX lxx;
        MatrixLUU luu;
        MatrixLXU lxu;
        t_cost_hessian_.setZero();
        for (const auto &ct : solver_->Problem().GetCostUnionPtr()->GetCostTerms()) {
            MatrixLXX cur_lxx;
            MatrixLUU cur_luu;
            MatrixLXU cur_lxu;
            ct->Hessian(step, x, u, cur_lxx, cur_luu, cur_lxu);
            lxx += cur_lxx;
            luu += cur_luu;
            lxu += cur_lxu;
        }
        t_cost_hessian_ << lxx,              lxu,
                           lxu.transpose(),  luu;
//        cout << "t_cost_hessian" << t_cost_hessian_ << endl;
    }

    /* in this ocp, inequality constraints are defined as g(x) > 0
     * active set is defined as g(x) < -kDefaultConstraintEpsilon
     * true means that the size of Jacobian Matrix is larger than 1
     * false means that it is a grdient vector
     * */
    JacSize GetActiveConstraintsGradientAt(const int step, const State &x, const Control &u,
                                           MatrixXd &D, MatrixXd &d) {
//        cout << step << endl;
        VecX lx;
        VecU lu;
        lx.setZero();
        lu.setZero();
        vector<double> violations;
        MatrixZs JacVec;
        int index = 0;
        eq_active_sets_.clear();
        ineq_active_sets_.clear();
        for (auto &cons_ptr : solver_->Problem().GetCostUnionPtr()->GetImmutableEqConstraints()) {
            bool success = cons_ptr->Evaluate(step, x, u);
            if (success) {
                double cons_val = cons_ptr->ConsVal();
                if (std::fabs(cons_val) > kDefaultConstraintEpsilon) {
                    violations.emplace_back(cons_val);
                    cons_ptr->Gradient(step, x, u);
                    VecX cur_lx = cons_ptr->GradientX();
                    VecU cur_lu = cons_ptr->GradientU();
                    MatrixZ gz;
                    gz << cur_lx, cur_lu;
                    JacVec.emplace_back(gz);
                    eq_active_sets_.emplace_back(index);
                    index++;
                }
            }
        }

        index = 0;
        for (auto &cons_ptr : solver_->Problem().GetCostUnionPtr()->GetImmutableIneqConstraints()) {
            bool success = cons_ptr->Evaluate(step, x, u);
            if (success) {
                double cons_val = cons_ptr->ConsVal();
                if (cons_val < -kDefaultConstraintEpsilon) {
                    violations.emplace_back(cons_val);
                    cons_ptr->Gradient(step, x, u);
                    VecX cur_lx = cons_ptr->GradientX();
                    VecU cur_lu = cons_ptr->GradientU();
                    MatrixZ gz;
                    gz << cur_lx, cur_lu;
                    JacVec.emplace_back(gz);
                    ineq_active_sets_.emplace_back(index);
                    index++;
                }
            }
        }
        int rows = JacVec.size();
        if (rows == 0)
            return JacSize::None;
        D = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, M + N, Eigen::RowMajor>>(
                reinterpret_cast<T*>(JacVec.data()), rows, M + N);

        d = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(violations.data(), rows, 1);
        if (D.rows() <= 1)
            return JacSize::Gradient;
        return JacSize::Jacobian;
    }

    void UpdateConstraintValues(const int step, const State &x, const Control &u,
                                MatrixXd& d) {
        vector<double> violations;
        for (const int &ind : eq_active_sets_) {
            bool success = solver_->Problem().GetCostUnionPtr()->GetImmutableEqConstraints()[ind]->Evaluate(step, x, u);
            if (success) {
                double cons_val = solver_->Problem().GetCostUnionPtr()->GetImmutableEqConstraints()[ind]->ConsVal();
                violations.emplace_back(cons_val);
            }
        }

        for (const int &ind : ineq_active_sets_) {
            bool success = solver_->Problem().GetCostUnionPtr()->GetImmutableIneqConstraints()[ind]->Evaluate(step, x, u);
            if (success) {
                double cons_val = solver_->Problem().GetCostUnionPtr()->GetImmutableIneqConstraints()[ind]->ConsVal();
                violations.emplace_back(cons_val);
            }
        }
        d = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(violations.data(), violations.size(), 1);
//        cout << "d=\n" << d << endl;
    }

    Eigen::MatrixXd PseudoInverse(const Eigen::MatrixXd& mat, double tolerance = 1e-6) {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXd singularValues = svd.singularValues();
        Eigen::MatrixXd singularValuesInv(mat.cols(), mat.rows());
        singularValuesInv.setZero();

        for (int i = 0; i < singularValues.size(); ++i) {
            if (singularValues(i) > tolerance) {
                singularValuesInv(i, i) = 1.0 / singularValues(i);
            }
        }

        return svd.matrixV() * singularValuesInv * svd.matrixU().transpose();
    }

    void Projection(const int step, State x, Control u,
                    State &polished_x, Control &polished_u) {
        GetCostHessianAt(step, x, u);
        Eigen::LLT<MatrixXd> llt(t_cost_hessian_);
        MatrixXd H_inv;
        if (llt.info() == Eigen::Success) {
            H_inv = t_cost_hessian_.inverse();
        }
        else {
//            H_inv = t_cost_hessian_.completeOrthogonalDecomposition().pseudoInverse();
            H_inv = PseudoInverse(t_cost_hessian_);
        }
        if (!H_inv.allFinite() or H_inv.array().isNaN().any() or H_inv.sum() < kTol) {
//            cout << "fail to calc Inverse\n";
            return;
        }
        int loop = 0;
        while (true) {
            MatrixXd D;
            MatrixXd d;
            JacSize linearized_active_cons = GetActiveConstraintsGradientAt(step, x, u, D, d);
//            cout << "H_inv : \n" << H_inv << endl;
            double v = std::numeric_limits<double>::epsilon();
            if (linearized_active_cons == JacSize::None) {
//                cout << "No active constraints at step " << step << " found\n";
                return;
            }
            MatrixXd mat = D * H_inv * D.transpose();
            if (mat.sum() < kTol)
                return;
            MatrixXd S;
            double S_ = 0.0;
            S.setZero();
            if (linearized_active_cons == JacSize::Jacobian) {
                S = mat.llt().matrixL();
            }
            else if (linearized_active_cons == JacSize::Gradient){
                S_ = std::sqrt(mat(0));
            }

            v = d.lpNorm<Eigen::Infinity>();
            if (v < kTol) {
                break;
            }
            if (loop > kMaxOuterIter)
                break;

            int inner_loop = 0;
            MatrixXd S_inv;
            double S_inv_;
            MatrixXd delta_z;
            if (linearized_active_cons == JacSize::Jacobian) {
                Eigen::LLT<MatrixXd> llt(S);
                if (llt.info() == Eigen::Success) {
                    S_inv = S.inverse();
                }
                else {
                    S_inv = PseudoInverse(S);
                }

                if (!S_inv.allFinite() or S_inv.array().isNaN().any()) {
//                        cout << "Failed to calc inverse of Matrix S\n";
                    return;
                }
            } else {
                S_inv_ = 1.0 / S_;
            }
            double r = std::numeric_limits<double>::infinity();
            while (true) {
                // line search
                double alpha = 1.0;
                double gamma = 0.5;
                if (linearized_active_cons == JacSize::Jacobian)
                    delta_z = H_inv * D.transpose() * (S_inv * S_inv.transpose() * d);
                else if (linearized_active_cons == JacSize::Gradient)
                    delta_z = H_inv * D.transpose() * (S_inv_ * S_inv_ * d);
//                cout << "delta_z: \n" << delta_z << endl;
                double v0;
                State x_n;
                Control u_n;
                bool flag = false;
                for (int i = 0; i < kMaxLineSearchIter; ++i) {
//                    cout << "line search " << i << endl;
                    x_n = x + alpha * delta_z.topRows(M);
                    u_n = u + alpha * delta_z.bottomRows(N);
                    UpdateConstraintValues(step, x_n, u_n, d);
                    v0 = d.lpNorm<Eigen::Infinity>();
//                    cout << "v0=" << v0 << " v=" << v << endl;
                    if (v0 < v) {
//                        cout << "line search successfully\n";
                        x = x_n;
                        u = u_n;
                        flag = true;
                        break;
                    }
                    alpha *= gamma;
                }
                // end line search
                r = std::log(v0) / std::log(v);
                if (flag)
                    v = v0;
                if (v < kTol or r < kConvRateTol) {
//                    cout << "line search successfully\n";
//                    cout << "r=" << r << endl;
//                    cout << "v=" << v << endl;
                    x = x_n;
                    u = u_n;
                    break;
                }
                inner_loop++;
                if (inner_loop > kMaxInnerIter)
                    break;
            }
            loop++;
        }
        polished_x = x;
        polished_u = u;
    }

    double GetMaxViolation() {return solver_->Problem().GetCostUnionPtr()->GetMaxViolation();}



private:
//    std::unique_ptr<OCPInterface<T, M, N>> problem_;
    std::unique_ptr<ILQRSolver<T, M, N>> solver_;
    std::vector<double> step_sizes_;
    MatrixHZ t_cost_hessian_;
    int steps_num_;
//    CostUnion<T, M, N> solver_->ProblemPtr()->GetCostUnionPtr();
    vector<int> eq_active_sets_;
    vector<int> ineq_active_sets_;
};


#endif //ALILQR_ALTRO_SOLVER_HPP
