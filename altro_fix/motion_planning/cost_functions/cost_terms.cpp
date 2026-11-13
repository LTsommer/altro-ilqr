//
// Created by LTsommer on 2024/6/28.
//

#include "cost_terms.hpp"

CentripetalAccCostFunc::CentripetalAccCostFunc(const CostTermConfig &config)
    : w_lat_acc_(config.weights.at(WeightIndex::WeightLatAcc)) {
    SetHorizon(config.steps.size());
}

bool CentripetalAccCostFunc::Evaluate(const int step, const State &x, const Control &u, double &val) const {
    if (step > horizon) {
        return false;
    }

    val = 0.5 * sqr(LateralAcc(x)) * w_lat_acc_;
    return true;
}

bool CentripetalAccCostFunc::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step > horizon) {
        return false;
    }

    lx(mVel) = doppel(cube(x(mVel)) * sqr(x(mKappa))) * w_lat_acc_;
    lx(mKappa) = sqr(sqr(x(mVel))) * x(mKappa) * w_lat_acc_;
    return true;
}

bool CentripetalAccCostFunc::Hessian(const int step,
                                     const State &x,
                                     const Control &u,
                                     Eigen::Ref<MatrixLXX> lxx,
                                     Eigen::Ref<MatrixLUU> luu,
                                     Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step > horizon) {
        return false;
    }

    lxx(mVel, mVel) = doppel(3 * sqr(x(mVel)) * sqr(x(mKappa))) * w_lat_acc_;
    lxx(mVel, mKappa) = doppel(2 * cube(x(mVel)) * x(mKappa)) * w_lat_acc_;
    lxx(mKappa, mKappa) = sqr(x(mVel)) * sqr(x(mVel)) * w_lat_acc_;
    lxx(mKappa, mVel) = lxx(mVel, mKappa); // was 3 * v^3 * kappa * w; must be symmetric (4 * v^3 * kappa * w)
    return true;
}

CentripetalJerkCostFunc::CentripetalJerkCostFunc(const CostTermConfig &config)
    : w_lat_jerk_(config.weights.at(WeightIndex::WeightLatJerk)) {
    SetHorizon(config.steps.size());
}

bool CentripetalJerkCostFunc::Evaluate(const int step, const State &x, const Control &u, double &val) const {
    if (step >= horizon) {
        val = 0.0;
        return false;
    }

    const double coe_term = LateralJerk(x, u); // lateral jerk at (x, u)
    val = 0.5 * sqr(coe_term) * w_lat_jerk_;
    return true;
}

bool CentripetalJerkCostFunc::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step >= horizon) {
        return false;
    }
    const double coe_term = LateralJerk(x, u); // lateral jerk at (x, u)
    lx(mVel) = coe_term * doppel(x(mAcc) * x(mKappa) + x(mVel) * u(mDotKappa)) * w_lat_jerk_;
    lx(mKappa) = coe_term * doppel(x(mVel) * x(mAcc)) * w_lat_jerk_;
    lx(mAcc) = coe_term * doppel(x(mVel) * x(mKappa)) * w_lat_jerk_;
    lu(mDotKappa) = coe_term * sqr(x(mVel)) * w_lat_jerk_;
    return true;
}

bool CentripetalJerkCostFunc::Hessian(const int step,
                                      const State &x,
                                      const Control &u,
                                      Eigen::Ref<MatrixLXX> lxx,
                                      Eigen::Ref<MatrixLUU> luu,
                                      Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step >= horizon) {
        return false;
    }
    const double coe_term = LateralJerk(x, u); // lateral jerk at (x, u)
    /* df / dv = w * (2 * vel * acc * kappa + vel ^ 2 * kappa_dot) * (2 * acc * kappa + 2 * vel * kappa_dot)
     * df / dk = w * (2 * vel * acc * kappa + vel ^ 2 * kappa_dot) * (2 * vel * acc)
     * df / da = w * (2 * vel * acc * kappa + vel ^ 2 * kappa_dot) * (2 * vel * kappa)
     * df / ddk = w * (2 * vel * acc * kappa + vel ^ 2 * kappa_dot) * vel^2
     * */
    double dv_term = doppel(x(mAcc) * x(mKappa) + x(mVel) * u(mDotKappa));
    double dk_term = doppel(x(mVel) * x(mAcc));
    double da_term = doppel(x(mVel) * x(mKappa));
    // d(df/dv) / dv, da, dk
    lxx(mVel, mVel) = (sqr(dv_term) + coe_term * doppel(u(mDotKappa))) * w_lat_jerk_;
    lxx(mVel, mAcc) = (doppel(x(mVel) * x(mKappa)) * dv_term + coe_term * doppel(x(mKappa))) * w_lat_jerk_;
    lxx(mVel, mKappa) = (doppel(x(mVel) * x(mAcc)) * dv_term + coe_term * doppel(x(mAcc))) * w_lat_jerk_;
    // d(df/dk) / dv, da, dk
    lxx(mKappa, mVel) =
            (doppel(x(mAcc) * x(mKappa) + x(mVel) * u(mDotKappa)) * dk_term + coe_term * doppel(x(mAcc))) * w_lat_jerk_;
    lxx(mKappa, mAcc) = (doppel(x(mVel) * x(mKappa)) * dk_term + coe_term * doppel(x(mVel))) * w_lat_jerk_;
    lxx(mKappa, mKappa) = doppel(x(mVel) * x(mAcc)) * dk_term * w_lat_jerk_;
    // d(df/da) / dv, da, dk
    lxx(mAcc, mVel) = (doppel(x(mAcc) * x(mKappa) + x(mVel) * u(mDotKappa)) * da_term + coe_term * doppel(x(mKappa))) *
                      w_lat_jerk_;
    lxx(mAcc, mAcc) = doppel(x(mVel) * x(mKappa)) * da_term * w_lat_jerk_;
    lxx(mAcc, mKappa) = (doppel(x(mVel) * x(mAcc)) * da_term + coe_term * doppel(x(mVel))) * w_lat_jerk_;
    // df(df/dv) / dj ddk
    lxu(mVel, mDotKappa) = (sqr(x(mVel)) * dv_term + coe_term * doppel(x(mVel))) * w_lat_jerk_;
    lxu(mKappa, mDotKappa) = sqr(x(mVel)) * dk_term * w_lat_jerk_;
    lxu(mAcc, mDotKappa) = sqr(x(mVel)) * da_term * w_lat_jerk_;

    // df(df/ddk) / dj ddk
    luu(mDotKappa, mDotKappa) = sqr(x(mVel)) * sqr(x(mVel)) * w_lat_jerk_;
    return true;
}

DotCurveCostFunc::DotCurveCostFunc(const CostTermConfig &config)
    : w_dot_jerk_(config.weights.at(WeightIndex::WeightDotKappa)) {
    SetHorizon(config.steps.size());
}

bool DotCurveCostFunc::Evaluate(const int step, const State &x, const Control &u, double &val) const {
    if (step >= horizon) {
        val = 0.0;
        return false;
    }

    val = 0.5 * sqr(u(mDotKappa)) * w_dot_jerk_;
    return true;
}

bool DotCurveCostFunc::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step >= horizon) {
        return false;
    }

    lu(mDotKappa) = w_dot_jerk_ * u(mDotKappa);
    return true;
}

bool DotCurveCostFunc::Hessian(const int step,
                               const State &x,
                               const Control &u,
                               Eigen::Ref<MatrixLXX> lxx,
                               Eigen::Ref<MatrixLUU> luu,
                               Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step >= horizon) {
        return false;
    }

    luu(mDotKappa, mDotKappa) = w_dot_jerk_;
    return true;
}

LinearJerkCostFunc::LinearJerkCostFunc(const CostTermConfig &config)
    : w_jerk_(config.weights.at(WeightIndex::WeightJerk)) {
    SetHorizon(config.steps.size());
}

bool LinearJerkCostFunc::Evaluate(const int step, const State &x, const Control &u, double &val) const {
    if (step >= horizon) {
        val = 0.0;
        return false;
    }

    val = 0.5 * sqr(u(mJerk)) * w_jerk_;
    return true;
}

bool LinearJerkCostFunc::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step >= horizon) {
        return false;
    }

    lu(mJerk) = w_jerk_ * u(mJerk);
    return true;
}

bool LinearJerkCostFunc::Hessian(const int step,
                                 const State &x,
                                 const Control &u,
                                 Eigen::Ref<MatrixLXX> lxx,
                                 Eigen::Ref<MatrixLUU> luu,
                                 Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step >= horizon) {
        return false;
    }

    luu(mJerk, mJerk) = w_jerk_;
    return true;
}

TargetSpeedCostFunc::TargetSpeedCostFunc(const CostTermConfig &config)
    : w_vel_diff_(config.weights.at(WeightVelDiff)), velocity_(config.velocity) {
    SetHorizon(config.steps.size());
}

bool TargetSpeedCostFunc::Evaluate(const int step, const State &x, const Control &u, double &val) const {
    if (step > horizon) {
        val = 0.0;
        return false;
    }

    val = 0.5 * w_vel_diff_ * sqr<double>(x(mVel) - velocity_[step]);
    return true;
}

bool TargetSpeedCostFunc::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step > horizon) {
        return false;
    }

    lx(mVel) = w_vel_diff_ * (x(mVel) - velocity_[step]);
    return true;
}

bool TargetSpeedCostFunc::Hessian(const int step,
                                  const State &x,
                                  const Control &u,
                                  Eigen::Ref<MatrixLXX> lxx,
                                  Eigen::Ref<MatrixLUU> luu,
                                  Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step > horizon) { // same range as Evaluate / Gradient (was >=, dropping the last step)
        return false;
    }

    lxx(mVel, mVel) = w_vel_diff_;
    return true;
}
