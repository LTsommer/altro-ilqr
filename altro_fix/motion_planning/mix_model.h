//
// Created by LTsommer on 2024/6/12.
//

#ifndef CILQR_MIX_MODEL_H
#define CILQR_MIX_MODEL_H

#include "model.h"
#include "../ilqr_system_declaration.hpp"
#include "../calculus.h"
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
enum MModelStateIndex : unsigned int { mXPos = 0, mYPos, mTheta, mKappa, mVel, mAcc };

enum MModelActionIndex : unsigned int { mJerk = 0, mDotKappa };

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
        const PositionPartials p = ComputePositionPartials(x, u, dt);
        for (const unsigned r : {mXPos, mYPos}) {
            const std::array<double, 4> &g = r == mXPos ? p.dx : p.dy;
            fx.row(r) = g[kTheta1] * fx.row(mTheta) + g[kVel1] * fx.row(mVel);
            fx(r, r) += 1.0;
            fx(r, mTheta) += g[kTheta0];
            fx(r, mVel) += g[kVel0];
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
        const PositionPartials p = ComputePositionPartials(x, u, dt);
        fu.row(mXPos) = p.dx[kTheta1] * fu.row(mTheta) + p.dx[kVel1] * fu.row(mVel);
        fu.row(mYPos) = p.dy[kTheta1] * fu.row(mTheta) + p.dy[kVel1] * fu.row(mVel);
        return fu;
    }

    inline State ForwardCalculation(const State &x, const Control &u, const double dt) const override {
        State state;
        state(mTheta) = NextTheta(x, u, dt);
        state(mKappa) = x(mKappa) + u(mDotKappa) * dt;
        state(mVel) = NextVel(x, u, dt);
        state(mAcc) = x(mAcc) + u(mJerk) * dt;
        double dx = GaussLegendreIntegrate<delta_x, double>(x(mTheta), state(mTheta), x(mVel), state(mVel), dt);
        double dy = GaussLegendreIntegrate<delta_y, double>(x(mTheta), state(mTheta), x(mVel), state(mVel), dt);
        state(mXPos) = x(mXPos) + dx;
        state(mYPos) = x(mYPos) + dy;
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

    // Partials of the position increments dx, dy (10-point Gauss-Legendre, theta and v linear
    // between the two ends, exactly as in GaussLegendreIntegrate) w.r.t. the end values.
    enum PartialIndex { kTheta0 = 0, kTheta1, kVel0, kVel1 };
    struct PositionPartials {
        std::array<double, 4> dx{};
        std::array<double, 4> dy{};
    };

    static PositionPartials ComputePositionPartials(const State &x, const Control &u, const double dt) {
        static constexpr std::array<double, 10> kWeights{0.0666713443086881,
                                                         0.1494513491505806,
                                                         0.2190863625159820,
                                                         0.2692667193099963,
                                                         0.2955242247147529,
                                                         0.2955242247147529,
                                                         0.2692667193099963,
                                                         0.2190863625159820,
                                                         0.1494513491505806,
                                                         0.0666713443086881};
        static constexpr std::array<double, 10> kNodes{-0.9739065285171717,
                                                       -0.8650633666889845,
                                                       -0.6794095682990244,
                                                       -0.4333953941292472,
                                                       -0.1488743389816312,
                                                       0.1488743389816312,
                                                       0.4333953941292472,
                                                       0.6794095682990244,
                                                       0.8650633666889845,
                                                       0.9739065285171717};
        const double th0 = x(mTheta), th1 = NextTheta(x, u, dt);
        const double v0 = x(mVel), v1 = NextVel(x, u, dt);
        PositionPartials p;
        for (size_t i = 0; i < kNodes.size(); ++i) {
            const double s = 0.5 * (kNodes[i] + 1.0); // position of the node in [0, 1]
            const double w = 0.5 * dt * kWeights[i];
            const double th = th0 + (th1 - th0) * s;
            const double v = v0 + (v1 - v0) * s;
            const double c = std::cos(th), sn = std::sin(th);
            // dx = sum w * cos(th) * v,  dy = sum w * sin(th) * v
            p.dx[kTheta0] += w * (-sn * v) * (1.0 - s);
            p.dx[kTheta1] += w * (-sn * v) * s;
            p.dx[kVel0] += w * c * (1.0 - s);
            p.dx[kVel1] += w * c * s;
            p.dy[kTheta0] += w * (c * v) * (1.0 - s);
            p.dy[kTheta1] += w * (c * v) * s;
            p.dy[kVel0] += w * sn * (1.0 - s);
            p.dy[kVel1] += w * sn * s;
        }
        return p;
    }

    spline sy;
    spline ts;
    spline tv;
};

#endif //CILQR_MIX_MODEL_H
