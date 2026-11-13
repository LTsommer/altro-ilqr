//
// Created by LTsommer on 2024/6/6.
//

#ifndef CILQR_MODEL_H
#define CILQR_MODEL_H

#include "../ilqr_system_declaration.hpp"

template<typename T, unsigned int M, unsigned int N>
class ModelBase {
public:
    OCP_VARIABLES(T, M, N)
    ModelBase() = default;
    virtual ~ModelBase() = default;

    // Convert the externally supplied initial state to the model's representation.
    virtual State PrepareInitialState(const State &state) const { return state; }

    // Difference in the local coordinates used by the dynamics Jacobians and feedback gains.
    virtual State StateDifference(const State &state, const State &reference) const { return state - reference; }

    // Linearize a valid model step in the coordinates used by StateDifference.
    virtual inline A JacobianX(const State &state, const Control &u, const double step) const = 0;

    virtual inline B JacobianU(const State &state, const Control &u, const double step) const = 0;

    virtual inline State ForwardCalculation(const State &state, const Control &u, const double step) const = 0;

    virtual inline void SetTimer(const double timer) = 0;

    virtual inline void UpdateTimer(const double dt) = 0;

protected:
    double timer_;
};

#endif //CILQR_MODEL_H
