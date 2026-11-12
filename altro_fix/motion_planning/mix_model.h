//
// Created by LTsommer on 2024/6/12.
//

#ifndef CILQR_MIX_MODEL_H
#define CILQR_MIX_MODEL_H

#include "model.h"
#include "../ilqr_system_declaration.hpp"
#include "../common/math_utils.h"
#include "../common/calculus.h"
#include "spline.h"
#include <array>
#include <cmath>

// https://zhuanlan.zhihu.com/p/681311263

// state    [x, y, theta, kappa, v, a]
// control  [j, kappa_dot]

/* x(k+1) = x(k) + int cos(theta(t)) * v(t) dt   (10-point Gauss-Legendre)
 * y(k+1) = y(k) + int sin(theta(t)) * v(t) dt
 * theta(k+1) = theta(k) + kappa(k) * v(k) * dt + 1/2(kappa(k)*a(k) + v(k)*u2(k)) * dt^2
 *              + 1/3 (1/2 kappa(k)*u1(k) + a(k)*u2(k))dt^3 + 1/8 u1(k)*u2(k)*dt^4
 * kappa(k+1) = kappa(k) + u2(k) * dt
 * v(k+1) = v(k) + a(k) * dt + 1/2 u1(k)*dt^2
 * a(k+1) = a(k) + u1(k) * dt
 * */
enum MModelStateIndex : unsigned int {
    mXPos = 0,
    mYPos,
    mTheta,
    mKappa,
    mVel,
    mAcc,
};

enum MModelActionIndex : unsigned int {
    mJerk = 0,
    mDotKappa,
};

class MixModel final : public ModelBase<double, 6, 2> {
public:
    MixModel() = default;
    ~MixModel() = default;

    inline A JacobianX(const State &x, const Control &u, const double dt) const override {
        A fx;
        fx.setIdentity();
        // theta / v / a rows: exact derivatives of the closed-form updates.
        // Note: integer literals like 1/6 evaluate to 0 in C++, so use floating-point constants.
        fx(mTheta, mKappa) = x(mVel) * dt + 0.5 * x(mAcc) * sqr<double>(dt) + (1.0 / 6.0) * u(mJerk) * cube<double>(dt);
        fx(mTheta, mVel) = x(mKappa) * dt + 0.5 * u(mDotKappa) * sqr<double>(dt);
        fx(mTheta, mAcc) = 0.5 * x(mKappa) * sqr<double>(dt) + (1.0 / 3.0) * u(mDotKappa) * cube<double>(dt);
        fx(mVel, mAcc) = dt;

        // x / y rows: derivatives of the same Gauss-Legendre position increment used in
        // ForwardCalculation (they used to be the Euler approximation, which also missed the
        // dependence on kappa, a and u through theta_{k+1} and v_{k+1}).
        using P = PlanarPositionPartials;
        const P p = PositionPartials(x, u, dt);
        for (const int r : {mXPos, mYPos}) {
            const std::array<double, 4> &g = r == mXPos ? p.dx : p.dy;
            fx.row(r) = g[P::kTheta1] * fx.row(mTheta) + g[P::kVel1] * fx.row(mVel);
            fx(r, r) += 1.0;
            fx(r, mTheta) += g[P::kTheta0];
            fx(r, mVel) += g[P::kVel0];
        }
        return fx;
    }

    inline B JacobianU(const State &x, const Control &u, const double dt) const override {
        B fu;
        fu.setZero();
        fu(mTheta, mJerk) = ((1.0 / 6.0) * x(mKappa) + 0.125 * u(mDotKappa) * dt) * cube<double>(dt);
        // d(theta)/d(dkappa) = 1/2 v dt^2 + 1/3 a dt^3 + 1/8 j dt^4 (the a-term was missing)
        fu(mTheta, mDotKappa) =
                (0.5 * x(mVel) + (1.0 / 3.0) * x(mAcc) * dt + 0.125 * u(mJerk) * sqr<double>(dt)) * sqr<double>(dt);
        fu(mKappa, mDotKappa) = dt;
        fu(mVel, mJerk) = 0.5 * sqr<double>(dt);
        fu(mAcc, mJerk) = dt;

        // x / y depend on u only through theta_{k+1} and v_{k+1}
        using P = PlanarPositionPartials;
        const P p = PositionPartials(x, u, dt);
        fu.row(mXPos) = p.dx[P::kTheta1] * fu.row(mTheta) + p.dx[P::kVel1] * fu.row(mVel);
        fu.row(mYPos) = p.dy[P::kTheta1] * fu.row(mTheta) + p.dy[P::kVel1] * fu.row(mVel);
        return fu;
    }

    inline State ForwardCalculation(const State &x, const Control &u, const double dt) const override {
        State state;
        state(mTheta) = NextTheta(x, u, dt);
        state(mKappa) = x(mKappa) + u(mDotKappa) * dt;
        state(mVel) = NextVel(x, u, dt);
        state(mAcc) = x(mAcc) + u(mJerk) * dt;
        const Eigen::Vector2d displacement =
                IntegratePlanarPosition(x(mTheta), state(mTheta), x(mVel), state(mVel), dt);
        state(mXPos) = x(mXPos) + displacement.x();
        state(mYPos) = x(mYPos) + displacement.y();
        return state;
    }

    void set_traj_info(const vector<double> &t,
                       const vector<double> &s,
                       const vector<double> &v,
                       const vector<double> &y) {
        sy.set_points(s, y);
        ts.set_points(t, s);
        tv.set_points(t, v);
    }

    void SetTimer(const double timer) override { timer_ = timer; }

    void UpdateTimer(const double dt) override { timer_ += dt; }

private:
    // theta_{k+1} = theta + int_0^dt kappa(t) v(t) dt, with kappa linear and v quadratic in t
    static double NextTheta(const State &x, const Control &u, const double dt) {
        return x(mTheta) + x(mKappa) * x(mVel) * dt +
               0.5 * (x(mKappa) * x(mAcc) + x(mVel) * u(mDotKappa)) * sqr<double>(dt) +
               (1.0 / 3.0) * (0.5 * x(mKappa) * u(mJerk) + x(mAcc) * u(mDotKappa)) * cube<double>(dt) +
               0.125 * u(mDotKappa) * u(mJerk) * cube<double>(dt) * dt;
    }

    static double NextVel(const State &x, const Control &u, const double dt) {
        return x(mVel) + x(mAcc) * dt + 0.5 * u(mJerk) * sqr<double>(dt);
    }

    // Partials of the position displacement computed by ForwardCalculation.
    static PlanarPositionPartials PositionPartials(const State &x, const Control &u, const double dt) {
        return IntegratePlanarPositionPartials(x(mTheta), NextTheta(x, u, dt), x(mVel), NextVel(x, u, dt), dt);
    }

    spline sy;
    spline ts;
    spline tv;
};

#endif //CILQR_MIX_MODEL_H
