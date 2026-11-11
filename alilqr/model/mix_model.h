//
// Created by LTsommer on 2024/6/12.
//

#ifndef CILQR_MIX_MODEL_H
#define CILQR_MIX_MODEL_H

#include "model.h"
#include "../ilqr_system_declaration.hpp"
#include "../calculus.h"
#include "../frenet_coordinate/spline.h"

// https://zhuanlan.zhihu.com/p/681311263

// state    [x, y, theta, kappa, v, a]
// control  [j, kappa_dot]

/* x(k+1) = x(k) + cos(theta(k)) * v(k) * dt
 * y(k+1) = y(k) + sin(theta(k)) * v(k) * dt
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
    mAcc
};

enum MModelActionIndex : unsigned int {
    mJerk = 0,
    mDotKappa
};


class MixModel final : public Model<double, 6, 2> {
public:
    MixModel() = default;
    ~MixModel() = default;

    inline A JacobianX(const State &x,
                       const Control &u,
                       const double dt) const override {
        A fx;
        fx.setIdentity();
        fx(mXPos, mTheta) = -sin(x(mTheta)) * x(mVel) * dt;
        fx(mXPos, mVel) = cos(x(mTheta)) * dt;
        fx(mYPos, mTheta) = cos(x(mTheta)) * x(mVel) * dt;
        fx(mYPos, mVel) = sin(x(mTheta)) * dt;
        fx(mTheta, mKappa) = x(mVel) * dt + 0.5 * x(mAcc) * sqr<double>(dt) + 1/6 * u(mJerk) * cube<double>(dt);
        fx(mTheta, mVel) = x(mKappa) * dt + 0.5 * u(mDotKappa) * sqr<double>(dt);
        fx(mTheta, mAcc) = 0.5 * x(mKappa) * sqr<double>(dt) + 1/3 * u(mDotKappa) * cube<double>(dt);
        fx(mVel, mAcc) = dt;

        return fx;
    }

    inline B JacobianU(const State &x,
                       const Control &u,
                       const double dt) const override{
        B fu;
        fu.setZero();
        fu(mTheta, mJerk) = (1/6 * x(mKappa) + 1/8 * u(mDotKappa) * dt) * cube<double>(dt);
        fu(mTheta, mDotKappa) = (0.5 * x(mVel) + 1/8 * u(mJerk) * sqr<double>(dt)) * sqr<double>(dt);
        fu(mKappa, mDotKappa) = dt;
        fu(mVel, mJerk) = 0.5 * sqr<double>(dt);
        fu(mAcc, mJerk) = dt;
        return fu;
    }

    inline State ForwardCalculation(const State &x,
                                    const Control &u,
                                    const double dt) const override {
        State state;
//        double dx = GaussLegendreIntegrate<delta_x, double>(sy, ts, tv, timer_, dt);
//        double dy = GaussLegendreIntegrate<delta_y, double>(sy, ts, tv, timer_, dt);
//        std::cout << "dx = " << dx << " dy = " << dy << "\n";
//        state(mXPos) = x(mXPos) + dx;
//        state(mYPos) = x(mYPos) + dy;
        state(mXPos) = x(mXPos) + cos(state(mTheta)) * state(mVel) * dt;
        state(mYPos) = x(mYPos) + sin(state(mTheta)) * state(mVel) * dt;
        state(mTheta) = x(mTheta) + x(mKappa) * x(mVel) * dt +
                1/2 * (x(mKappa) * x(mAcc) + x(mVel) * u(mDotKappa)) * sqr<double>(dt) +
                1/3 * (1/2 * x(mKappa) * u(mJerk) + x(mAcc) * u(mDotKappa)) * cube<double>(dt) +
                1/8 * u(mDotKappa) * u(mJerk) * cube<double>(dt) * dt;
        state(mKappa) = x(mKappa) + u(mDotKappa) * dt;
        state(mVel) = x(mVel) + x(mAcc) * dt + 1/2 * u(mJerk) * sqr<double>(dt);
        state(mAcc) = x(mAcc) + u(mJerk) * dt;

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

    void SetTimer(const double timer) override {
        timer_ = timer;
    }

    void UpdateTimer(const double dt) override {
        timer_ += dt;
    }

private:
    spline sy;
    spline ts;
    spline tv;
};

#endif //CILQR_MIX_MODEL_H
