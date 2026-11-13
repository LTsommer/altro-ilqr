//
// Created by LTsommer on 2024/5/30.
//

#ifndef CILQR_ILQR_PROCESS_H
#define CILQR_ILQR_PROCESS_H

#include "../ilqr_system_declaration.hpp"

enum class SolverStatus {
    OCPSolved,
    OCPUnsolved,
    MaxIterReached,
    RegularizationFailed,
    CostIncrease,
    MaxPenalty,
};

template<typename T, unsigned int M, unsigned int N>
struct ILQRSolverState {
    OCP_VARIABLES(T, M, N)
    /* cost(0) & cons(0)      cost(1) & cons(1)           ... cost(n-1) & cons(n-1) cost(n) & cons(n)
     *         ↓                      ↓                                 ↓                   ↓
     * x(0)->f(x(0), u(0)) => x(1)->f(x(1), u(1)) => x(2) ... x(n-1)->f(x(n-1), u(n-1)) => x(n)
     *                 ↑                      ↑                                    ↑
     *               u(0)                   u(1)          ...                   u(n-1)
     *  steps_num indicates that there are totally N control inputs
     *  and N+1 states are equivalent.
     *  Everything calculated which is related to u has only N dimensions
     * */
    explicit ILQRSolverState(int steps_num)
        : x_seq_(steps_num + 1, State::Zero()),
          u_seq_(steps_num, Control::Zero()),
          fx_(steps_num, MatrixLXX::Zero()),
          fu_(steps_num, MatrixLXU::Zero()),
          lx_(steps_num, VecX::Zero()),
          lu_(steps_num, VecU::Zero()),
          cx_(steps_num, VecX::Zero()),
          cu_(steps_num, VecU::Zero()),
          lxx_(steps_num, MatrixLXX::Zero()),
          lxu_(steps_num, MatrixLXU::Zero()),
          luu_(steps_num, MatrixLUU::Zero()),
          cxx_(steps_num, MatrixLXX::Zero()),
          cxu_(steps_num, MatrixLXU::Zero()),
          cuu_(steps_num, MatrixLUU::Zero()),
          Vx_(steps_num + 1, VecX::Zero()),
          Vxx_(steps_num + 1, MatrixLXX::Zero()),
          k_(steps_num, VecU::Zero()),
          K_(steps_num, MatrixLUX::Zero()) {}

    const States &x_seq() const { return x_seq_; }
    const Controls u_seq() const { return u_seq_; }

    State x0_;       // initial state
    States x_seq_;   // current working state sequence
    Controls u_seq_; // current working control sequence
    double cost_;    // current working cost value

    // M = dims(state), N = dims(control)
    MatrixLXXs fx_; // M * M * (T+1), jacobian of f w.r.t x
    MatrixLXUs fu_; // M * N * (T+1), jacobian of f w.r.t u
    VecXs lx_;      // M * (T+1), gradient of cost w.r.t x
    VecUs lu_;      // N * (T+1), gradient of cost w.r.t u
    VecXs cx_; // M * (T+1), augmented lag gradient of constraint w.r.t x : c_x * (lambda + penalty * c(x, u))
    VecUs cu_; // N * (T+1), augmented lag gradient of constraint w.r.t u : c_u * (lambda + penalty * c(x, u))
    MatrixLXXs lxx_; // M * M * (T+1), hessian of cost w.r.t x
    MatrixLXUs lxu_; // M * N * (T+1), hessian of cost w.r.t x, u
    MatrixLUUs luu_; // N * N * (T+1), hessian of cost w.r.t u
    // M * M * (T+1), augmented lag hessian of cost w.r.t x : c_x * penalty * c_x
    MatrixLXXs cxx_;
    // N * N * (T+1), augmented lag hessian of cost w.r.t x, u : c_x * penalty * c_u
    MatrixLXUs cxu_;
    // N * N * (T+1), augmented lag hessian of cost w.r.t u : c_u * penalty * c_u
    MatrixLUUs cuu_;

    Vector2d dV_;    // delta_V 2 * 1
    VecXs Vx_;       // M * (T+1)
    MatrixLXXs Vxx_; // M * M * (T+1)
    VecUs k_;        // N * T, open loop gain term
    MatrixLUXs K_;   // N * M * T, feedback gain term

    VecX Qx_;
    VecU Qu_;
    VecU k_i_;
    MatrixLXX Qxx_;
    MatrixLUX Qux_;
    MatrixLUU Quu_;
    MatrixLUX K_i_;
    MatrixLUU QuuF_;

    double rho_ = 1e-8;
    double drho_ = 1.0;

    double g_norm_ = 0.0;
    SolverStatus status_;
};

#endif //CILQR_ILQR_PROCESS_H
