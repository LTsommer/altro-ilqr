//
// Created by LTsommer on 2024/6/4.
//

#ifndef CILQR_SMOOTH_KAPPA_MODEL_H
#define CILQR_SMOOTH_KAPPA_MODEL_H

#include "../ilqr_system_declaration.hpp"
#include "heading.h"
#include "model.h"
#include <limits>

class SmoothKappaModel final : public ModelBase<double, 5, 1> {
public:
    SmoothKappaModel() = default;
    virtual ~SmoothKappaModel() = default;

    State PrepareInitialState(const State &state) const override;

    State StateDifference(const State &state,
                          const State &reference) const override;

    // step here is ds
    /* x(k+1) = x(k) + cos(theta(k)) * ds
     * y(k+1) = y(k) + sin(theta(k)) * ds
     * theta(k+1) = theta(k) + kappa(k) * ds + 1/2 * dkappa(k) * ds^2 + 1/6 * u(k) * ds^3
     * kappa(k+1) = kappa(k) + dkappa(k) * ds + 1/2 * u(k) * ds^2
     * dkappa(k+1) = dkappa(k) + u(k) * ds
    */
    inline A JacobianX(const State &state,
                       const Control &u,
                       const double step) const override {
        A fx;
        fx.setIdentity();
        fx(kXPos, kTheta) = -trig_table::SinValue(state(kTheta)) * step;
        fx(kYPos, kTheta) = trig_table::CosValue(state(kTheta)) * step;
        fx(kTheta, kKappa) = step;
        fx(kTheta, kDkappa) = 0.5 * sqr<double>(step);
        fx(kKappa, kDkappa) = step;
        return fx;
    };

    inline B JacobianU(const State &state,
                       const Control &u,
                       const double step) const override {
        B fu;
        fu.setZero();
        fu(kTheta) = 1.0 / 6.0 * cube<double>(step);
        fu(kKappa) = 0.5 * sqr<double>(step);
        fu(kDkappa) = step;
        return fu;
    }

    inline State ForwardCalculation(const State &state,
                                    const Control &u,
                                    const double step) const override {
        const double heading_change =
                state(kKappa) * step +
                0.5 * state(kDkappa) * sqr<double>(step) +
                (1.0 / 6.0) * u(kDDkappa) * cube<double>(step);
        if (!vehicle_heading::IsSupportedAngle(heading_change)) {
            return State::Constant(std::numeric_limits<double>::quiet_NaN());
        }
        State new_state;
        new_state(kXPos) =
                state(kXPos) + trig_table::CosValue(state(kTheta)) * step;
        new_state(kYPos) =
                state(kYPos) + trig_table::SinValue(state(kTheta)) * step;
        new_state(kTheta) = NormalizeAngle(state(kTheta) + heading_change);
        new_state(kKappa) = state(kKappa) + state(kDkappa) * step +
                            0.5 * u(kDDkappa) * sqr<double>(step);
        new_state(kDkappa) = state(kDkappa) + u(kDDkappa) * step;
        return new_state;
    }

    virtual inline void SetTimer(const double timer) override { return; }

    virtual inline void UpdateTimer(const double dt) override { return; }
};

#endif //CILQR_SMOOTH_KAPPA_MODEL_H
