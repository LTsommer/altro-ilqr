//
// Created by LTsommer on 2024/5/30.
//

#ifndef CILQR_ILQR_SOLVER_H
#define CILQR_ILQR_SOLVER_H

#include "ocp_problem.hpp"
#include "ilqr_process.hpp"

template<typename T, unsigned int M, unsigned int N>
class ILQRSolver{
public:
    OCP_VARIABLES(T, M, N)
    explicit ILQRSolver(std::unique_ptr<OCPInterface<T, M, N>> &&ocp_interface,
                        const vector<double> & steps)
    : problem_(std::move(ocp_interface)),
    step_sizes_(steps)
    {};

    ILQRSolver(const ILQRSolver &) = delete;
    ILQRSolver &operator=(const ILQRSolver &) = delete;

    const OCPInterface<T, M, N> &Problem() const {return *problem_;}

    static const Controls &GenerateNominalControlSeq(int steps) {
        return std::move(Controls(steps, 0.5 * Control::Ones()));
    }

    [[maybe_unused]]
    void AddCostFuncs(const CostTermConfig &config) {
        problem_->SetConfig(config);
//        problem_->add_cost_funcs();
    }

    void Solve(const State &x0, const Controls &u0,
               States &x_res_seq, Controls &u_res_seq);

    void SetTimer(const bool use_timer) { use_timer_ = use_timer;}

    double GetMaxViolation() const;

    SolverStatus GetSolverStatus() {return status_;}

    void ShowSolverState() const {
        switch(status_) {
            case SolverStatus::OCPSolved:
                std::cout << "Problem Solved\n";
                break;
            case SolverStatus::OCPUnsolved:
                std::cout << "Problem Unsolved\n";
                break;
            case SolverStatus::MaxIterReached:
                std::cout << "Maximum Iteration Reached\n";
                break;
            case SolverStatus::CostIncrease:
                std::cout << "Problem Not Converge\n";
                break;
            case SolverStatus::MaxPenalty:
                std::cout << "Maximum Penalty Reached\n";
                break;
            default:
                std::cout << "Unknown Solver Status\n";
        }
    }

private:
    double InitTraj(const State &x_0, const Controls &u_0,
                    ILQRSolverState<T, M, N> *ilqr_state) const;

    // The following methods implement the iLQR algorithm
    void GenerateTrajectory(ILQRSolverState<T, M, N> *ilqr_state);

    bool ForwardProcess(const States &x_old, const Controls &u_old, double *new_cost,
                        double *dcost, double *expected, double *ratio,
                        ILQRSolverState<T, M, N> *ilqr_state) const;

    double RollOut(const State &x0, const Controls &u, ILQRSolverState<T, M, N> *ilqr_state) const;

    void BackwardProcess(ILQRSolverState<T, M, N> *ilqr_state) const;

    void IncreaseRho(double &drho, double &rho) const;

    void DecreaseRho(double &drho, double &rho) const;

    void UpdateDualsAndPenalties(ILQRSolverState<T, M, N> *ilqr_state) const;

//    void UpdatePenalties() const;

    double GetGradientNorm(const VecUs &l, const Controls &u) const;

    void InitializeILQRSolverState(ILQRSolverState<T, M, N> *ilqr_state) const;



    // The following methods compute derivatives
    void CalcKinematicsDerivatives(const States &x, const Controls &u,
                                   MatrixLXXs *f_x, MatrixLXUs *f_u) const;

    const std::vector<double> &step_sizes() const { return step_sizes_; }

private:
    std::unique_ptr<OCPInterface<T, M, N>> problem_;
    std::vector<double> step_sizes_;
    static const int kMaxIter_;
    static const double kTolFun_;
    static const double kTolGrad_;
    static const double kViolationTol_;
    static const double krhoFactor_;
    static const double krhoMax_;
    static const double krhoMin_;
    static const double krhoMinGrad_;
    static const double kRatioMin_;
    static const double kRationMax_;
    static const int kMaxRegCount_;
    static const std::array<double, 11> alpha_vec_;
    SolverStatus status_;
    bool use_timer_ = false;
};


#endif //CILQR_ILQR_SOLVER_H
