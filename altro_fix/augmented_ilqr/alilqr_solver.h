//
// Created by LTsommer on 2024/5/30.
//

#ifndef CILQR_ILQR_SOLVER_H
#define CILQR_ILQR_SOLVER_H

#include "ocp_problem.hpp"
#include "ilqr_process.hpp"
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

template<typename T, unsigned int M, unsigned int N>
class ALILQRSolver {
public:
    OCP_VARIABLES(T, M, N)
    // Throws std::invalid_argument unless every step size is finite and positive: the dynamics
    // (and their integrators) are only defined for such steps, and every solver path uses them.
    explicit ALILQRSolver(
            std::unique_ptr<OCPInterface<T, M, N>> &&ocp_interface,
            const vector<double> &steps);

    ALILQRSolver(const ALILQRSolver &) = delete;
    ALILQRSolver &operator=(const ALILQRSolver &) = delete;

    const OCPInterface<T, M, N> &Problem() const { return *problem_; }

    // Returned by value: returning a reference to a temporary was a dangling reference.
    static Controls GenerateNominalControlSeq(int steps) {
        return Controls(steps, 0.5 * Control::Ones());
    }

    [[maybe_unused]]
    void AddCostFuncs(const CostTermConfig &config) {
        problem_->SetConfig(config);
    }

    void Solve(const State &x0,
               const Controls &u0,
               States &x_res_seq,
               Controls &u_res_seq);

    void SetTimer(const bool use_timer) { use_timer_ = use_timer; }

    double GetMaxViolation() const;

    SolverStatus GetSolverStatus() { return status_; }

    void ShowSolverState() const;

    void SetUpdateConfig(const bool update) { update_config_ = update; }

    // Per-iteration console logs (off by default: console output is slow and would distort
    // timing). The end-of-solve summary is printed after the clock stops; turn it off with
    // SetPrintSummary(false).
    void SetVerbose(const bool verbose) { verbose_ = verbose; }

    void SetPrintSummary(const bool print) { print_summary_ = print; }

    bool Verbose() const { return verbose_; }

    int TotalIterations() const { return total_iters_; }

    double SolveTimeMs() const { return solve_time_ms_; }

private:
    double InitTraj(const State &x_0,
                    const Controls &u_0,
                    ILQRSolverState<T, M, N> *ilqr_state) const;

    // The following methods implement the iLQR algorithm
    void GenerateTrajectory(ILQRSolverState<T, M, N> *ilqr_state);

    void PrintSummary(double max_violation,
                      std::uint32_t total_time_microseconds,
                      std::uint32_t derivative_time_microseconds,
                      std::uint32_t backward_time_microseconds,
                      std::uint32_t forward_time_microseconds,
                      std::uint32_t update_config_time_microseconds) const;

    bool ForwardProcess(const States &x_old,
                        const Controls &u_old,
                        double *new_cost,
                        double *dcost,
                        double *expected,
                        double *ratio,
                        ILQRSolverState<T, M, N> *ilqr_state) const;

    double RollOut(const State &x0,
                   const Controls &u,
                   ILQRSolverState<T, M, N> *ilqr_state) const;

    // Cost (including AL terms) of a given trajectory, without changing it.
    double EvaluateTotalCost(const States &x, const Controls &u) const;

    // Returns false if Quu could not be made invertible even at the maximum regularization.
    bool BackwardProcess(ILQRSolverState<T, M, N> *ilqr_state) const;

    void IncreaseRho(double &drho, double &rho) const;

    void DecreaseRho(double &drho, double &rho) const;

    void UpdateDualsAndPenalties() const;

    void UpdateConvergenceState();

    bool AnyPenaltyAtMax() const;

    double GetGradientNorm(const VecUs &l, const Controls &u) const;

    void InitializeILQRSolverState(ILQRSolverState<T, M, N> *ilqr_state) const;

    // The following methods compute derivatives
    void CalcKinematicsDerivatives(const States &x,
                                   const Controls &u,
                                   MatrixLXXs *f_x,
                                   MatrixLXUs *f_u) const;

    const std::vector<double> &step_sizes() const { return step_sizes_; }

private:
    std::unique_ptr<OCPInterface<T, M, N>> problem_;
    std::vector<double> step_sizes_;
    // max inner iLQR iterations per AL outer iteration
    static const int kMaxIter_;
    // max AL outer iterations (dual / penalty updates)
    static const int kMaxOuterIter_;
    // inner: accepted cost decrease / |cost| below this
    static const double kTolFunRel_;
    // inner: predicted full-step decrease / |cost| below this
    static const double kTolExpRel_;
    // outer: violation must drop below this fraction of the last one
    static const double kStallRatio_;
    static const double kViolationTol_;
    static const double krhoFactor_;
    static const double krhoMax_;
    static const double krhoMin_;
    static const double krhoMinGrad_;
    static const double kRatioMin_;
    static const double kRationMax_;
    static const int kMaxRegCount_;
    static const std::array<double, 11> alpha_vec_;
    SolverStatus status_ = SolverStatus::OCPUnsolved;
    bool use_timer_ = false;
    bool update_config_ = false;
    bool verbose_ = false;
    bool print_summary_ = true;
    int total_iters_ = 0;
    double solve_time_ms_ = 0.0;
    double max_penalty_ = 0.0;
};

#endif //CILQR_ILQR_SOLVER_H
