//
// Created by LTsommer on 2024/6/28.
//

#include "constraints.hpp"

AccLimitsConstraint::AccLimitsConstraint(const CostTermConfig &config, const bool is_lb) {
    horizon_ = config.steps.size();
    if (is_lb) {
        a_ = -1;
        b_ = config.acc_min;
    } else {
        a_ = 1;
        b_ = -config.acc_max;
    }
}

bool AccLimitsConstraint::Evaluate(const int step, const State &x, const Control &u, double &cost_val) const {
    if (step > horizon_) {
        return false;
    }
    cost_val = a_ * x(mAcc) + b_;
    return true;
}

bool AccLimitsConstraint::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step > horizon_) {
        return false;
    }
    lx(mAcc) = a_;
    return true;
}

bool AccLimitsConstraint::Hessian(const int step,
                                  const State &x,
                                  const Control &u,
                                  Eigen::Ref<MatrixLXX> lxx,
                                  Eigen::Ref<MatrixLUU> luu,
                                  Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step > horizon_) {
        return false;
    }
    return true;
}

JerkLimitsConstraint::JerkLimitsConstraint(const CostTermConfig &config, const bool is_lb) {
    horizon_ = config.steps.size();
    if (is_lb) {
        a_ = -1;
        b_ = config.jerk_min;
    } else {
        a_ = 1;
        b_ = -config.jerk_max;
    }
}

bool JerkLimitsConstraint::Evaluate(const int step, const State &x, const Control &u, double &cost_val) const {
    if (step >= horizon_) {
        return false;
    }
    cost_val = a_ * u(mJerk) + b_;
    return true;
}

bool JerkLimitsConstraint::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step >= horizon_) {
        return false;
    }
    lu(mJerk) = a_;
    return true;
}

bool JerkLimitsConstraint::Hessian(const int step,
                                   const State &x,
                                   const Control &u,
                                   Eigen::Ref<MatrixLXX> lxx,
                                   Eigen::Ref<MatrixLUU> luu,
                                   Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step >= horizon_) {
        return false;
    }
    return true;
}

VelLimitsConstraint::VelLimitsConstraint(const CostTermConfig &config, const bool is_lb) {
    horizon_ = config.steps.size();
    if (is_lb) {
        a_ = -1;
        b_ = config.vel_min;
    } else {
        a_ = 1;
        b_ = -config.vel_max;
    }
}

bool VelLimitsConstraint::Evaluate(const int step, const State &x, const Control &u, double &cost_val) const {
    if (step > horizon_) {
        return false;
    }
    cost_val = a_ * x(mVel) + b_;
    return true;
}

bool VelLimitsConstraint::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step > horizon_) {
        return false;
    }
    lx(mVel) = a_;
    return true;
}

bool VelLimitsConstraint::Hessian(const int step,
                                  const State &x,
                                  const Control &u,
                                  Eigen::Ref<MatrixLXX> lxx,
                                  Eigen::Ref<MatrixLUU> luu,
                                  Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step > horizon_) {
        return false;
    }
    return true;
}

CurveLimitsConstraint::CurveLimitsConstraint(const CostTermConfig &config, const bool is_lb) {
    horizon_ = config.steps.size();
    if (is_lb) {
        a_ = -1;
        b_ = config.kappa_min;
    } else {
        a_ = 1;
        b_ = -config.kappa_max;
    }
}

bool CurveLimitsConstraint::Evaluate(const int step, const State &x, const Control &u, double &cost_val) const {
    if (step > horizon_) {
        return false;
    }
    cost_val = a_ * x(mKappa) + b_;
    return true;
}

bool CurveLimitsConstraint::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step > horizon_) {
        return false;
    }
    lx(mKappa) = a_;
    return true;
}

bool CurveLimitsConstraint::Hessian(const int step,
                                    const State &x,
                                    const Control &u,
                                    Eigen::Ref<MatrixLXX> lxx,
                                    Eigen::Ref<MatrixLUU> luu,
                                    Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step > horizon_) {
        return false;
    }
    return true;
}

CentripetalAccLimitsConstraint::CentripetalAccLimitsConstraint(const CostTermConfig &config, const bool is_lb) {
    horizon_ = config.steps.size();
    if (is_lb) {
        a_ = -1;
        b_ = config.lat_acc_min;
    } else {
        a_ = 1;
        b_ = -config.lat_acc_max;
    }
}

bool CentripetalAccLimitsConstraint::Evaluate(const int step,
                                              const State &x,
                                              const Control &u,
                                              double &cost_val) const {
    if (step > horizon_) {
        return false;
    }
    cost_val = a_ * LateralAcc(x) + b_;
    return true;
}

bool CentripetalAccLimitsConstraint::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step > horizon_) {
        return false;
    }
    lx(mVel) = 2 * a_ * x(mVel) * x(mKappa);
    lx(mKappa) = a_ * sqr(x(mVel));
    return true;
}

bool CentripetalAccLimitsConstraint::Hessian(const int step,
                                             const State &x,
                                             const Control &u,
                                             Eigen::Ref<MatrixLXX> lxx,
                                             Eigen::Ref<MatrixLUU> luu,
                                             Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step > horizon_) {
        return false;
    }

    lxx(mVel, mVel) = doppel(a_ * x(mKappa));
    lxx(mVel, mKappa) = doppel(a_ * x(mVel));
    lxx(mKappa, mVel) = doppel(a_ * x(mVel));
    return true;
}

CentripetalJerkLimitsConstraint::CentripetalJerkLimitsConstraint(const CostTermConfig &config, const bool is_lb) {
    horizon_ = config.steps.size();
    if (is_lb) {
        a_ = -1;
        b_ = config.lat_jerk_min;
    } else {
        a_ = 1;
        b_ = -config.lat_jerk_max;
    }
}

bool CentripetalJerkLimitsConstraint::Evaluate(const int step,
                                               const State &x,
                                               const Control &u,
                                               double &cost_val) const {
    if (step >= horizon_) {
        cost_val = 0.0;
        return false;
    }

    cost_val = a_ * LateralJerk(x, u) + b_;
    return true;
}

bool CentripetalJerkLimitsConstraint::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step >= horizon_) {
        return false;
    }

    lx(mVel) = a_ * doppel((x(mAcc) * x(mKappa) + x(mVel) * u(mDotKappa)));
    lx(mKappa) = a_ * doppel(x(mVel) * x(mAcc));
    lx(mAcc) = a_ * doppel(x(mVel) * x(mKappa));
    lu(mDotKappa) = a_ * sqr(x(mVel));
    return true;
}

bool CentripetalJerkLimitsConstraint::Hessian(const int step,
                                              const State &x,
                                              const Control &u,
                                              Eigen::Ref<MatrixLXX> lxx,
                                              Eigen::Ref<MatrixLUU> luu,
                                              Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step >= horizon_) {
        return false;
    }
    lxx(mVel, mVel) = a_ * doppel(u(mDotKappa));
    lxx(mVel, mAcc) = a_ * doppel(x(mKappa));
    lxx(mVel, mKappa) = a_ * doppel(x(mAcc));
    lxu(mVel, mDotKappa) = a_ * doppel(x(mVel));

    lxx(mKappa, mVel) = a_ * doppel(x(mAcc));
    lxx(mKappa, mAcc) = a_ * doppel(x(mVel));

    lxx(mAcc, mVel) = a_ * doppel(x(mKappa));
    lxx(mAcc, mKappa) = a_ * doppel(x(mVel));

    return true;
}

void HeadingDiffLimitsConstraint::UpdateConfig(const CostTermConfig &config, const bool is_lb) {
    a_ = is_lb ? -1 : 1;
    offset_ = is_lb ? config.yaw_diff_min : -config.yaw_diff_max;
    yaw_ = config.yaw;
}

HeadingDiffLimitsConstraint::HeadingDiffLimitsConstraint(const CostTermConfig &config, const bool is_lb) {
    horizon_ = config.steps.size();
    UpdateConfig(config, is_lb);
}

bool HeadingDiffLimitsConstraint::Evaluate(const int step, const State &x, const Control &u, double &cost_val) const {
    if (step > horizon_) {
        cost_val = 0.0;
        return false;
    }
    cost_val = a_ * AngleDifference(x(mTheta), yaw_[step]) + offset_;
    return true;
}

bool HeadingDiffLimitsConstraint::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step > horizon_) {
        return false;
    }
    lx(mTheta) = a_;
    return true;
}

bool HeadingDiffLimitsConstraint::Hessian(const int step,
                                          const State &x,
                                          const Control &u,
                                          Eigen::Ref<MatrixLXX> lxx,
                                          Eigen::Ref<MatrixLUU> luu,
                                          Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step > horizon_) {
        return false;
    }
    return true;
}

AccConstraint::AccConstraint(const CostTermConfig &config)
    : w_clb_(config.weights.at(WeightClb)),
      w_cub_(config.weights.at(WeightCub)),
      acc_limit_constraint_lb_(config, true),
      acc_limit_constraint_ub_(config, false) {
    horizon_ = config.steps.size();
}

bool AccConstraint::RawConstraints(const int step,
                                   const State &x,
                                   const Control &u,
                                   RawConstraintRows<double, 6, 2> &out) const {
    AppendRawConstraint<double, 6, 2>(acc_limit_constraint_lb_, step, x, u, out);
    AppendRawConstraint<double, 6, 2>(acc_limit_constraint_ub_, step, x, u, out);
    return true;
}

bool AccConstraint::Evaluate(const int step, const State &x, const Control &u, double &cost_val) const {
    if (step > horizon_) {
        return false;
    }

    double g_lb{0.0};
    double g_ub{0.0};
    acc_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
    acc_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

    double f_lb = one_side_cost_.Evaluate(g_lb);
    double f_ub = one_side_cost_.Evaluate(g_ub);

    cost_val = w_clb_ * f_lb + w_cub_ * f_ub;
    return true;
}

bool AccConstraint::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step > horizon_) {
        return false;
    }

    double g_lb{0.0};
    double g_ub{0.0};
    acc_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
    acc_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

    VecX grad_gx_lb, grad_gx_ub;
    VecU grad_gu_lb, grad_gu_ub;
    acc_limit_constraint_lb_.Gradient(step, x, u, grad_gx_lb, grad_gu_lb);
    acc_limit_constraint_ub_.Gradient(step, x, u, grad_gx_ub, grad_gu_ub);

    double grad_f_lb{0.0};
    double grad_f_ub{0.0};
    one_side_cost_.Gradient(g_lb, grad_f_lb);
    one_side_cost_.Gradient(g_ub, grad_f_ub);

    lx = w_clb_ * grad_f_lb * grad_gx_lb + w_cub_ * grad_f_ub * grad_gx_ub;
    lu = w_clb_ * grad_f_lb * grad_gu_lb + w_cub_ * grad_f_ub * grad_gu_ub;
    return true;
}

bool AccConstraint::Hessian(const int step,
                            const State &x,
                            const Control &u,
                            Eigen::Ref<MatrixLXX> lxx,
                            Eigen::Ref<MatrixLUU> luu,
                            Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step > horizon_) {
        return false;
    }

    double g_lb{0.0};
    double g_ub{0.0};
    acc_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
    acc_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

    VecX grad_gx_lb, grad_gx_ub;
    VecU grad_gu_lb, grad_gu_ub;
    acc_limit_constraint_lb_.Gradient(step, x, u, grad_gx_lb, grad_gu_lb);
    acc_limit_constraint_ub_.Gradient(step, x, u, grad_gx_ub, grad_gu_ub);

    double grad_f_lb{0.0};
    double grad_f_ub{0.0};
    one_side_cost_.Gradient(g_lb, grad_f_lb);
    one_side_cost_.Gradient(g_ub, grad_f_ub);

    MatrixLXX hess_gxx_lb, hess_gxx_ub;
    MatrixLUU hess_guu_lb, hess_guu_ub;
    MatrixLXU hess_gxu_lb, hess_gxu_ub;
    acc_limit_constraint_lb_.Hessian(step, x, u, hess_gxx_lb, hess_guu_lb, hess_gxu_lb);
    acc_limit_constraint_ub_.Hessian(step, x, u, hess_gxx_ub, hess_guu_ub, hess_gxu_ub);

    double hess_f_lb{0.0};
    double hess_f_ub{0.0};
    one_side_cost_.Hessian(g_lb, hess_f_lb);
    one_side_cost_.Hessian(g_ub, hess_f_ub);

    lxx = w_clb_ * (hess_f_lb * grad_gx_lb * grad_gx_lb.transpose() + grad_f_lb * hess_gxx_lb) +
          w_cub_ * (hess_f_ub * grad_gx_ub * grad_gx_ub.transpose() + grad_f_ub * hess_gxx_ub);
    luu = w_clb_ * (hess_f_lb * grad_gu_lb * grad_gu_lb.transpose() + grad_f_lb * hess_guu_lb) +
          w_cub_ * (hess_f_ub * grad_gu_ub * grad_gu_ub.transpose() + grad_f_ub * hess_guu_ub);
    lxu = w_clb_ * (hess_f_lb * grad_gx_lb * grad_gu_lb.transpose() + grad_f_lb * hess_gxu_lb) +
          w_cub_ * (hess_f_ub * grad_gx_ub * grad_gu_ub.transpose() + grad_f_ub * hess_gxu_ub);

    return true;
}

VelConstraint::VelConstraint(const CostTermConfig &config)
    : w_clb_(config.weights.at(WeightClb)),
      w_cub_(config.weights.at(WeightCub)),
      vel_limit_constraint_lb_(config, true),
      vel_limit_constraint_ub_(config, false) {
    horizon_ = config.steps.size();
}

bool VelConstraint::RawConstraints(const int step,
                                   const State &x,
                                   const Control &u,
                                   RawConstraintRows<double, 6, 2> &out) const {
    AppendRawConstraint<double, 6, 2>(vel_limit_constraint_lb_, step, x, u, out);
    AppendRawConstraint<double, 6, 2>(vel_limit_constraint_ub_, step, x, u, out);
    return true;
}

bool VelConstraint::Evaluate(const int step, const State &x, const Control &u, double &cost_val) const {
    if (step > horizon_) {
        return false;
    }

    double g_lb{0.0};
    double g_ub{0.0};
    vel_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
    vel_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

    double f_lb = one_side_cost_.Evaluate(g_lb);
    double f_ub = one_side_cost_.Evaluate(g_ub);

    cost_val = w_clb_ * f_lb + w_cub_ * f_ub;
    return true;
}

bool VelConstraint::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step > horizon_) {
        return false;
    }

    double g_lb{0.0};
    double g_ub{0.0};
    vel_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
    vel_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

    VecX grad_gx_lb, grad_gx_ub;
    VecU grad_gu_lb, grad_gu_ub;
    vel_limit_constraint_lb_.Gradient(step, x, u, grad_gx_lb, grad_gu_lb);
    vel_limit_constraint_ub_.Gradient(step, x, u, grad_gx_ub, grad_gu_ub);

    double grad_f_lb{0.0};
    double grad_f_ub{0.0};
    one_side_cost_.Gradient(g_lb, grad_f_lb);
    one_side_cost_.Gradient(g_ub, grad_f_ub);

    lx = w_clb_ * grad_f_lb * grad_gx_lb + w_cub_ * grad_f_ub * grad_gx_ub;
    lu = w_clb_ * grad_f_lb * grad_gu_lb + w_cub_ * grad_f_ub * grad_gu_ub;
    return true;
}

bool VelConstraint::Hessian(const int step,
                            const State &x,
                            const Control &u,
                            Eigen::Ref<MatrixLXX> lxx,
                            Eigen::Ref<MatrixLUU> luu,
                            Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step > horizon_) {
        return false;
    }

    double g_lb{0.0};
    double g_ub{0.0};
    vel_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
    vel_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

    VecX grad_gx_lb, grad_gx_ub;
    VecU grad_gu_lb, grad_gu_ub;
    vel_limit_constraint_lb_.Gradient(step, x, u, grad_gx_lb, grad_gu_lb);
    vel_limit_constraint_ub_.Gradient(step, x, u, grad_gx_ub, grad_gu_ub);

    double grad_f_lb{0.0};
    double grad_f_ub{0.0};
    one_side_cost_.Gradient(g_lb, grad_f_lb);
    one_side_cost_.Gradient(g_ub, grad_f_ub);

    MatrixLXX hess_gxx_lb, hess_gxx_ub;
    MatrixLUU hess_guu_lb, hess_guu_ub;
    MatrixLXU hess_gxu_lb, hess_gxu_ub;
    vel_limit_constraint_lb_.Hessian(step, x, u, hess_gxx_lb, hess_guu_lb, hess_gxu_lb);
    vel_limit_constraint_ub_.Hessian(step, x, u, hess_gxx_ub, hess_guu_ub, hess_gxu_ub);

    double hess_f_lb{0.0};
    double hess_f_ub{0.0};
    one_side_cost_.Hessian(g_lb, hess_f_lb);
    one_side_cost_.Hessian(g_ub, hess_f_ub);

    lxx = w_clb_ * (hess_f_lb * grad_gx_lb * grad_gx_lb.transpose() + grad_f_lb * hess_gxx_lb) +
          w_cub_ * (hess_f_ub * grad_gx_ub * grad_gx_ub.transpose() + grad_f_ub * hess_gxx_ub);
    luu = w_clb_ * (hess_f_lb * grad_gu_lb * grad_gu_lb.transpose() + grad_f_lb * hess_guu_lb) +
          w_cub_ * (hess_f_ub * grad_gu_ub * grad_gu_ub.transpose() + grad_f_ub * hess_guu_ub);
    lxu = w_clb_ * (hess_f_lb * grad_gx_lb * grad_gu_lb.transpose() + grad_f_lb * hess_gxu_lb) +
          w_cub_ * (hess_f_ub * grad_gx_ub * grad_gu_ub.transpose() + grad_f_ub * hess_gxu_ub);

    return true;
}

JerkConstraint::JerkConstraint(const CostTermConfig &config)
    : w_clb_(config.weights.at(WeightClb)),
      w_cub_(config.weights.at(WeightCub)),
      jerk_limit_constraint_lb_(config, true),
      jerk_limit_constraint_ub_(config, false) {
    horizon_ = config.steps.size();
}

bool JerkConstraint::RawConstraints(const int step,
                                    const State &x,
                                    const Control &u,
                                    RawConstraintRows<double, 6, 2> &out) const {
    AppendRawConstraint<double, 6, 2>(jerk_limit_constraint_lb_, step, x, u, out);
    AppendRawConstraint<double, 6, 2>(jerk_limit_constraint_ub_, step, x, u, out);
    return true;
}

bool JerkConstraint::Evaluate(const int step, const State &x, const Control &u, double &cost_val) const {
    if (step >= horizon_) {
        return false;
    }

    double g_lb{0.0};
    double g_ub{0.0};
    jerk_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
    jerk_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

    double f_lb = one_side_cost_.Evaluate(g_lb);
    double f_ub = one_side_cost_.Evaluate(g_ub);

    cost_val = w_clb_ * f_lb + w_cub_ * f_ub;
    return true;
}

bool JerkConstraint::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step >= horizon_) {
        return false;
    }

    double g_lb{0.0};
    double g_ub{0.0};
    jerk_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
    jerk_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

    VecX grad_gx_lb, grad_gx_ub;
    VecU grad_gu_lb, grad_gu_ub;
    jerk_limit_constraint_lb_.Gradient(step, x, u, grad_gx_lb, grad_gu_lb);
    jerk_limit_constraint_ub_.Gradient(step, x, u, grad_gx_ub, grad_gu_ub);

    double grad_f_lb{0.0};
    double grad_f_ub{0.0};
    one_side_cost_.Gradient(g_lb, grad_f_lb);
    one_side_cost_.Gradient(g_ub, grad_f_ub);

    lx = w_clb_ * grad_f_lb * grad_gx_lb + w_cub_ * grad_f_ub * grad_gx_ub;
    lu = w_clb_ * grad_f_lb * grad_gu_lb + w_cub_ * grad_f_ub * grad_gu_ub;
    return true;
}

bool JerkConstraint::Hessian(const int step,
                             const State &x,
                             const Control &u,
                             Eigen::Ref<MatrixLXX> lxx,
                             Eigen::Ref<MatrixLUU> luu,
                             Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step >= horizon_) {
        return false;
    }

    double g_lb{0.0};
    double g_ub{0.0};
    jerk_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
    jerk_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

    VecX grad_gx_lb, grad_gx_ub;
    VecU grad_gu_lb, grad_gu_ub;
    jerk_limit_constraint_lb_.Gradient(step, x, u, grad_gx_lb, grad_gu_lb);
    jerk_limit_constraint_ub_.Gradient(step, x, u, grad_gx_ub, grad_gu_ub);

    double grad_f_lb{0.0};
    double grad_f_ub{0.0};
    one_side_cost_.Gradient(g_lb, grad_f_lb);
    one_side_cost_.Gradient(g_ub, grad_f_ub);

    MatrixLXX hess_gxx_lb, hess_gxx_ub;
    MatrixLUU hess_guu_lb, hess_guu_ub;
    MatrixLXU hess_gxu_lb, hess_gxu_ub;
    jerk_limit_constraint_lb_.Hessian(step, x, u, hess_gxx_lb, hess_guu_lb, hess_gxu_lb);
    jerk_limit_constraint_ub_.Hessian(step, x, u, hess_gxx_ub, hess_guu_ub, hess_gxu_ub);

    double hess_f_lb{0.0};
    double hess_f_ub{0.0};
    one_side_cost_.Hessian(g_lb, hess_f_lb);
    one_side_cost_.Hessian(g_ub, hess_f_ub);

    lxx = w_clb_ * (hess_f_lb * grad_gx_lb * grad_gx_lb.transpose() + grad_f_lb * hess_gxx_lb) +
          w_cub_ * (hess_f_ub * grad_gx_ub * grad_gx_ub.transpose() + grad_f_ub * hess_gxx_ub);
    luu = w_clb_ * (hess_f_lb * grad_gu_lb * grad_gu_lb.transpose() + grad_f_lb * hess_guu_lb) +
          w_cub_ * (hess_f_ub * grad_gu_ub * grad_gu_ub.transpose() + grad_f_ub * hess_guu_ub);
    lxu = w_clb_ * (hess_f_lb * grad_gx_lb * grad_gu_lb.transpose() + grad_f_lb * hess_gxu_lb) +
          w_cub_ * (hess_f_ub * grad_gx_ub * grad_gu_ub.transpose() + grad_f_ub * hess_gxu_ub);

    return true;
}

CurveConstraint::CurveConstraint(const CostTermConfig &config)
    : w_clb_(config.weights.at(WeightClb)),
      w_cub_(config.weights.at(WeightCub)),
      kappa_limit_constraint_lb_(config, true),
      kappa_limit_constraint_ub_(config, false) {
    horizon_ = config.steps.size();
}

bool CurveConstraint::RawConstraints(const int step,
                                     const State &x,
                                     const Control &u,
                                     RawConstraintRows<double, 6, 2> &out) const {
    AppendRawConstraint<double, 6, 2>(kappa_limit_constraint_lb_, step, x, u, out);
    AppendRawConstraint<double, 6, 2>(kappa_limit_constraint_ub_, step, x, u, out);
    return true;
}

bool CurveConstraint::Evaluate(const int step, const State &x, const Control &u, double &cost_val) const {
    if (step > horizon_) {
        cost_val = 0.0;
        return false;
    }

    double g_lb{0.0};
    double g_ub{0.0};
    kappa_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
    kappa_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

    double f_lb = one_side_cost_.Evaluate(g_lb);
    double f_ub = one_side_cost_.Evaluate(g_ub);

    cost_val = w_clb_ * f_lb + w_cub_ * f_ub;
    return true;
}

bool CurveConstraint::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step > horizon_) {
        return false;
    }

    double g_lb{0.0};
    double g_ub{0.0};
    kappa_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
    kappa_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

    VecX grad_gx_lb, grad_gx_ub;
    VecU grad_gu_lb, grad_gu_ub;
    kappa_limit_constraint_lb_.Gradient(step, x, u, grad_gx_lb, grad_gu_lb);
    kappa_limit_constraint_ub_.Gradient(step, x, u, grad_gx_ub, grad_gu_ub);

    double grad_f_lb{0.0};
    double grad_f_ub{0.0};
    one_side_cost_.Gradient(g_lb, grad_f_lb);
    one_side_cost_.Gradient(g_ub, grad_f_ub);

    lx = w_clb_ * grad_f_lb * grad_gx_lb + w_cub_ * grad_f_ub * grad_gx_ub;
    lu = w_clb_ * grad_f_lb * grad_gu_lb + w_cub_ * grad_f_ub * grad_gu_ub;
    return true;
}

bool CurveConstraint::Hessian(const int step,
                              const State &x,
                              const Control &u,
                              Eigen::Ref<MatrixLXX> lxx,
                              Eigen::Ref<MatrixLUU> luu,
                              Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step > horizon_) {
        return false;
    }

    double g_lb{0.0};
    double g_ub{0.0};
    kappa_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
    kappa_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

    VecX grad_gx_lb, grad_gx_ub;
    VecU grad_gu_lb, grad_gu_ub;
    kappa_limit_constraint_lb_.Gradient(step, x, u, grad_gx_lb, grad_gu_lb);
    kappa_limit_constraint_ub_.Gradient(step, x, u, grad_gx_ub, grad_gu_ub);

    double grad_f_lb{0.0};
    double grad_f_ub{0.0};
    one_side_cost_.Gradient(g_lb, grad_f_lb);
    one_side_cost_.Gradient(g_ub, grad_f_ub);

    MatrixLXX hess_gxx_lb, hess_gxx_ub;
    MatrixLUU hess_guu_lb, hess_guu_ub;
    MatrixLXU hess_gxu_lb, hess_gxu_ub;
    kappa_limit_constraint_lb_.Hessian(step, x, u, hess_gxx_lb, hess_guu_lb, hess_gxu_lb);
    kappa_limit_constraint_ub_.Hessian(step, x, u, hess_gxx_ub, hess_guu_ub, hess_gxu_ub);

    double hess_f_lb{0.0};
    double hess_f_ub{0.0};
    one_side_cost_.Hessian(g_lb, hess_f_lb);
    one_side_cost_.Hessian(g_ub, hess_f_ub);

    lxx = w_clb_ * (hess_f_lb * grad_gx_lb * grad_gx_lb.transpose() + grad_f_lb * hess_gxx_lb) +
          w_cub_ * (hess_f_ub * grad_gx_ub * grad_gx_ub.transpose() + grad_f_ub * hess_gxx_ub);
    luu = w_clb_ * (hess_f_lb * grad_gu_lb * grad_gu_lb.transpose() + grad_f_lb * hess_guu_lb) +
          w_cub_ * (hess_f_ub * grad_gu_ub * grad_gu_ub.transpose() + grad_f_ub * hess_guu_ub);
    lxu = w_clb_ * (hess_f_lb * grad_gx_lb * grad_gu_lb.transpose() + grad_f_lb * hess_gxu_lb) +
          w_cub_ * (hess_f_ub * grad_gx_ub * grad_gu_ub.transpose() + grad_f_ub * hess_gxu_ub);

    return true;
}

CentripetalAccConstraint::CentripetalAccConstraint(const CostTermConfig &config)
    : w_clb_(config.weights.at(WeightClb)),
      w_cub_(config.weights.at(WeightCub)),
      centripetal_acc_limit_constraint_lb_(config, true),
      centripetal_acc_limit_constraint_ub_(config, false) {
    horizon_ = config.steps.size();
}

bool CentripetalAccConstraint::RawConstraints(const int step,
                                              const State &x,
                                              const Control &u,
                                              RawConstraintRows<double, 6, 2> &out) const {
    AppendRawConstraint<double, 6, 2>(centripetal_acc_limit_constraint_lb_, step, x, u, out);
    AppendRawConstraint<double, 6, 2>(centripetal_acc_limit_constraint_ub_, step, x, u, out);
    return true;
}

bool CentripetalAccConstraint::Evaluate(const int step, const State &x, const Control &u, double &cost_val) const {
    if (step > horizon_) {
        cost_val = 0.0;
        return false;
    }

    double g_lb{0.0};
    double g_ub{0.0};
    centripetal_acc_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
    centripetal_acc_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

    double f_lb = one_side_cost_.Evaluate(g_lb);
    double f_ub = one_side_cost_.Evaluate(g_ub);

    cost_val = w_clb_ * f_lb + w_cub_ * f_ub;
    return true;
}

bool CentripetalAccConstraint::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step > horizon_) {
        return false;
    }

    double g_lb{0.0};
    double g_ub{0.0};
    centripetal_acc_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
    centripetal_acc_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

    VecX grad_gx_lb, grad_gx_ub;
    VecU grad_gu_lb, grad_gu_ub;
    centripetal_acc_limit_constraint_lb_.Gradient(step, x, u, grad_gx_lb, grad_gu_lb);
    centripetal_acc_limit_constraint_ub_.Gradient(step, x, u, grad_gx_ub, grad_gu_ub);

    double grad_f_lb{0.0};
    double grad_f_ub{0.0};
    one_side_cost_.Gradient(g_lb, grad_f_lb);
    one_side_cost_.Gradient(g_ub, grad_f_ub);

    lx = w_clb_ * grad_f_lb * grad_gx_lb + w_cub_ * grad_f_ub * grad_gx_ub;
    lu = w_clb_ * grad_f_lb * grad_gu_lb + w_cub_ * grad_f_ub * grad_gu_ub;
    return true;
}

bool CentripetalAccConstraint::Hessian(const int step,
                                       const State &x,
                                       const Control &u,
                                       Eigen::Ref<MatrixLXX> lxx,
                                       Eigen::Ref<MatrixLUU> luu,
                                       Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step > horizon_) {
        return false;
    }

    double g_lb{0.0};
    double g_ub{0.0};
    centripetal_acc_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
    centripetal_acc_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

    VecX grad_gx_lb, grad_gx_ub;
    VecU grad_gu_lb, grad_gu_ub;
    centripetal_acc_limit_constraint_lb_.Gradient(step, x, u, grad_gx_lb, grad_gu_lb);
    centripetal_acc_limit_constraint_ub_.Gradient(step, x, u, grad_gx_ub, grad_gu_ub);

    double grad_f_lb{0.0};
    double grad_f_ub{0.0};
    one_side_cost_.Gradient(g_lb, grad_f_lb);
    one_side_cost_.Gradient(g_ub, grad_f_ub);

    MatrixLXX hess_gxx_lb, hess_gxx_ub;
    MatrixLUU hess_guu_lb, hess_guu_ub;
    MatrixLXU hess_gxu_lb, hess_gxu_ub;
    centripetal_acc_limit_constraint_lb_.Hessian(step, x, u, hess_gxx_lb, hess_guu_lb, hess_gxu_lb);
    centripetal_acc_limit_constraint_ub_.Hessian(step, x, u, hess_gxx_ub, hess_guu_ub, hess_gxu_ub);

    double hess_f_lb{0.0};
    double hess_f_ub{0.0};
    one_side_cost_.Hessian(g_lb, hess_f_lb);
    one_side_cost_.Hessian(g_ub, hess_f_ub);

    lxx = w_clb_ * (hess_f_lb * grad_gx_lb * grad_gx_lb.transpose() + grad_f_lb * hess_gxx_lb) +
          w_cub_ * (hess_f_ub * grad_gx_ub * grad_gx_ub.transpose() + grad_f_ub * hess_gxx_ub);
    luu = w_clb_ * (hess_f_lb * grad_gu_lb * grad_gu_lb.transpose() + grad_f_lb * hess_guu_lb) +
          w_cub_ * (hess_f_ub * grad_gu_ub * grad_gu_ub.transpose() + grad_f_ub * hess_guu_ub);
    lxu = w_clb_ * (hess_f_lb * grad_gx_lb * grad_gu_lb.transpose() + grad_f_lb * hess_gxu_lb) +
          w_cub_ * (hess_f_ub * grad_gx_ub * grad_gu_ub.transpose() + grad_f_ub * hess_gxu_ub);
    return true;
}

CentripetalJerkConstraint::CentripetalJerkConstraint(const CostTermConfig &config)
    : w_clb_(config.weights.at(WeightClb)),
      w_cub_(config.weights.at(WeightCub)),
      centripetal_jerk_limit_constraint_lb_(config, true),
      centripetal_jerk_limit_constraint_ub_(config, false) {
    horizon_ = config.steps.size();
}

bool CentripetalJerkConstraint::RawConstraints(const int step,
                                               const State &x,
                                               const Control &u,
                                               RawConstraintRows<double, 6, 2> &out) const {
    AppendRawConstraint<double, 6, 2>(centripetal_jerk_limit_constraint_lb_, step, x, u, out);
    AppendRawConstraint<double, 6, 2>(centripetal_jerk_limit_constraint_ub_, step, x, u, out);
    return true;
}

bool CentripetalJerkConstraint::Evaluate(const int step, const State &x, const Control &u, double &cost_val) const {
    if (step >= horizon_) {
        cost_val = 0.0;
        return false;
    }

    double g_lb{0.0};
    double g_ub{0.0};
    centripetal_jerk_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
    centripetal_jerk_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

    double f_lb = one_side_cost_.Evaluate(g_lb);
    double f_ub = one_side_cost_.Evaluate(g_ub);

    cost_val = w_clb_ * f_lb + w_cub_ * f_ub;
    return true;
}

bool CentripetalJerkConstraint::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step >= horizon_) {
        return false;
    }

    double g_lb{0.0};
    double g_ub{0.0};
    centripetal_jerk_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
    centripetal_jerk_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

    VecX grad_gx_lb, grad_gx_ub;
    VecU grad_gu_lb, grad_gu_ub;
    centripetal_jerk_limit_constraint_lb_.Gradient(step, x, u, grad_gx_lb, grad_gu_lb);
    centripetal_jerk_limit_constraint_ub_.Gradient(step, x, u, grad_gx_ub, grad_gu_ub);

    double grad_f_lb{0.0};
    double grad_f_ub{0.0};
    one_side_cost_.Gradient(g_lb, grad_f_lb);
    one_side_cost_.Gradient(g_ub, grad_f_ub);

    lx = w_clb_ * grad_f_lb * grad_gx_lb + w_cub_ * grad_f_ub * grad_gx_ub;
    lu = w_clb_ * grad_f_lb * grad_gu_lb + w_cub_ * grad_f_ub * grad_gu_ub;
    return true;
}

bool CentripetalJerkConstraint::Hessian(const int step,
                                        const State &x,
                                        const Control &u,
                                        Eigen::Ref<MatrixLXX> lxx,
                                        Eigen::Ref<MatrixLUU> luu,
                                        Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step >= horizon_) {
        return false;
    }

    double g_lb{0.0};
    double g_ub{0.0};
    centripetal_jerk_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
    centripetal_jerk_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

    VecX grad_gx_lb, grad_gx_ub;
    VecU grad_gu_lb, grad_gu_ub;
    centripetal_jerk_limit_constraint_lb_.Gradient(step, x, u, grad_gx_lb, grad_gu_lb);
    centripetal_jerk_limit_constraint_ub_.Gradient(step, x, u, grad_gx_ub, grad_gu_ub);

    double grad_f_lb{0.0};
    double grad_f_ub{0.0};
    one_side_cost_.Gradient(g_lb, grad_f_lb);
    one_side_cost_.Gradient(g_ub, grad_f_ub);

    MatrixLXX hess_gxx_lb, hess_gxx_ub;
    MatrixLUU hess_guu_lb, hess_guu_ub;
    MatrixLXU hess_gxu_lb, hess_gxu_ub;
    centripetal_jerk_limit_constraint_lb_.Hessian(step, x, u, hess_gxx_lb, hess_guu_lb, hess_gxu_lb);
    centripetal_jerk_limit_constraint_ub_.Hessian(step, x, u, hess_gxx_ub, hess_guu_ub, hess_gxu_ub);

    double hess_f_lb{0.0};
    double hess_f_ub{0.0};
    one_side_cost_.Hessian(g_lb, hess_f_lb);
    one_side_cost_.Hessian(g_ub, hess_f_ub);

    lxx = w_clb_ * (hess_f_lb * grad_gx_lb * grad_gx_lb.transpose() + grad_f_lb * hess_gxx_lb) +
          w_cub_ * (hess_f_ub * grad_gx_ub * grad_gx_ub.transpose() + grad_f_ub * hess_gxx_ub);
    luu = w_clb_ * (hess_f_lb * grad_gu_lb * grad_gu_lb.transpose() + grad_f_lb * hess_guu_lb) +
          w_cub_ * (hess_f_ub * grad_gu_ub * grad_gu_ub.transpose() + grad_f_ub * hess_guu_ub);
    lxu = w_clb_ * (hess_f_lb * grad_gx_lb * grad_gu_lb.transpose() + grad_f_lb * hess_gxu_lb) +
          w_cub_ * (hess_f_ub * grad_gx_ub * grad_gu_ub.transpose() + grad_f_ub * hess_gxu_ub);
    return true;
}

HeadingDiffConstraint::HeadingDiffConstraint(const CostTermConfig &config)
    : w_clb_(config.weights.at(WeightClb)),
      w_cub_(config.weights.at(WeightCub)),
      heading_diff_limit_constraint_lb_(config, true),
      heading_diff_limit_constraint_ub_(config, false) {
    horizon_ = config.steps.size();
}

void HeadingDiffConstraint::UpdateConfig(const CostTermConfig &config) {
    heading_diff_limit_constraint_lb_.UpdateConfig(config, true);
    heading_diff_limit_constraint_ub_.UpdateConfig(config, false);
}

bool HeadingDiffConstraint::RawConstraints(const int step,
                                           const State &x,
                                           const Control &u,
                                           RawConstraintRows<double, 6, 2> &out) const {
    AppendRawConstraint<double, 6, 2>(heading_diff_limit_constraint_lb_, step, x, u, out);
    AppendRawConstraint<double, 6, 2>(heading_diff_limit_constraint_ub_, step, x, u, out);
    return true;
}

bool HeadingDiffConstraint::Evaluate(const int step, const State &x, const Control &u, double &cost_val) const {
    if (step > horizon_) {
        cost_val = 0.0;
        return false;
    }

    double g_lb{0.0};
    double g_ub{0.0};
    heading_diff_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
    heading_diff_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

    double f_lb{0.0};
    double f_ub{0.0};
    f_lb = one_side_cost_.Evaluate(g_lb);
    f_ub = one_side_cost_.Evaluate(g_ub);

    cost_val = w_clb_ * f_lb + w_cub_ * f_ub;
    return true;
}

bool HeadingDiffConstraint::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step > horizon_) {
        return false;
    }

    double g_lb{0.0};
    double g_ub{0.0};
    heading_diff_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
    heading_diff_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

    VecX grad_gx_lb, grad_gx_ub;
    VecU grad_gu_lb, grad_gu_ub;
    heading_diff_limit_constraint_lb_.Gradient(step, x, u, grad_gx_lb, grad_gu_lb);
    heading_diff_limit_constraint_ub_.Gradient(step, x, u, grad_gx_ub, grad_gu_ub);

    double grad_f_lb{0.0};
    double grad_f_ub{0.0};
    one_side_cost_.Gradient(g_lb, grad_f_lb);
    one_side_cost_.Gradient(g_ub, grad_f_ub);

    lx = w_clb_ * grad_f_lb * grad_gx_lb + w_cub_ * grad_f_ub * grad_gx_ub;
    lu = w_clb_ * grad_f_lb * grad_gu_lb + w_cub_ * grad_f_ub * grad_gu_ub;
    return true;
}

bool HeadingDiffConstraint::Hessian(const int step,
                                    const State &x,
                                    const Control &u,
                                    Eigen::Ref<MatrixLXX> lxx,
                                    Eigen::Ref<MatrixLUU> luu,
                                    Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step > horizon_) {
        return false;
    }

    double g_lb{0.0};
    double g_ub{0.0};
    heading_diff_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
    heading_diff_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

    VecX grad_gx_lb, grad_gx_ub;
    VecU grad_gu_lb, grad_gu_ub;
    heading_diff_limit_constraint_lb_.Gradient(step, x, u, grad_gx_lb, grad_gu_lb);
    heading_diff_limit_constraint_ub_.Gradient(step, x, u, grad_gx_ub, grad_gu_ub);

    double grad_f_lb{0.0};
    double grad_f_ub{0.0};
    one_side_cost_.Gradient(g_lb, grad_f_lb);
    one_side_cost_.Gradient(g_ub, grad_f_ub);

    MatrixLXX hess_gxx_lb, hess_gxx_ub;
    MatrixLUU hess_guu_lb, hess_guu_ub;
    MatrixLXU hess_gxu_lb, hess_gxu_ub;
    heading_diff_limit_constraint_lb_.Hessian(step, x, u, hess_gxx_lb, hess_guu_lb, hess_gxu_lb);
    heading_diff_limit_constraint_ub_.Hessian(step, x, u, hess_gxx_ub, hess_guu_ub, hess_gxu_ub);

    double hess_f_lb{0.0};
    double hess_f_ub{0.0};
    one_side_cost_.Hessian(g_lb, hess_f_lb);
    one_side_cost_.Hessian(g_ub, hess_f_ub);

    lxx = w_clb_ * (hess_f_lb * grad_gx_lb * grad_gx_lb.transpose() + grad_f_lb * hess_gxx_lb) +
          w_cub_ * (hess_f_ub * grad_gx_ub * grad_gx_ub.transpose() + grad_f_ub * hess_gxx_ub);
    luu = w_clb_ * (hess_f_lb * grad_gu_lb * grad_gu_lb.transpose() + grad_f_lb * hess_guu_lb) +
          w_cub_ * (hess_f_ub * grad_gu_ub * grad_gu_ub.transpose() + grad_f_ub * hess_guu_ub);
    lxu = w_clb_ * (hess_f_lb * grad_gx_lb * grad_gu_lb.transpose() + grad_f_lb * hess_gxu_lb) +
          w_cub_ * (hess_f_ub * grad_gx_ub * grad_gu_ub.transpose() + grad_f_ub * hess_gxu_ub);

    return true;
}

SafeDistanceToBoundaryConstraint::SafeDistanceToBoundaryConstraint(const CostTermConfig &config)
    : w_plb_(config.weights.at(WeightPlb)),
      w_prb_(config.weights.at(WeightPrb)),
      steps_num_(config.p_bound.size()),
      line_segment_constraint_lb_(config, true),
      line_segment_constraint_rb_(config, false) {
    horizon_ = config.steps.size();
}

void SafeDistanceToBoundaryConstraint::UpdateConfig(const CostTermConfig &config) {
    line_segment_constraint_lb_.UpdateConfig(config, true);
    line_segment_constraint_rb_.UpdateConfig(config, false);
}

bool SafeDistanceToBoundaryConstraint::RawConstraints(const int step,
                                                      const State &x,
                                                      const Control &u,
                                                      RawConstraintRows<double, 6, 2> &out) const {
    AppendRawConstraint<double, 6, 2>(line_segment_constraint_lb_, step, x, u, out);
    AppendRawConstraint<double, 6, 2>(line_segment_constraint_rb_, step, x, u, out);
    return true;
}

bool SafeDistanceToBoundaryConstraint::Evaluate(const int step, const State &x, const Control &u, double &val) const {
    if (step > horizon_) {
        val = 0.0;
        return false;
    }

    // using one side quadratic cost f(g(x))
    double g_lb{0.0};
    double g_rb{0.0};
    line_segment_constraint_lb_.Evaluate(step, x, u, g_lb);
    line_segment_constraint_rb_.Evaluate(step, x, u, g_rb);

    double f_lb = one_side_cost_.Evaluate(g_lb);
    double f_rb = one_side_cost_.Evaluate(g_rb);

    val = w_plb_ * f_lb + w_prb_ * f_rb;
    return true;
}

bool SafeDistanceToBoundaryConstraint::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step > steps_num_) {
        return false;
    }

    double g_lb{0.0};
    double g_rb{0.0};
    line_segment_constraint_lb_.Evaluate(step, x, u, g_lb);
    line_segment_constraint_rb_.Evaluate(step, x, u, g_rb);

    VecX grad_gx_lb, grad_gx_rb;
    VecU grad_gu_lb, grad_gu_rb;
    line_segment_constraint_lb_.Gradient(step, x, u, grad_gx_lb, grad_gu_lb);
    line_segment_constraint_rb_.Gradient(step, x, u, grad_gx_rb, grad_gu_rb);

    double grad_f_lb{0.0};
    double grad_f_rb{0.0};
    one_side_cost_.Gradient(g_lb, grad_f_lb);
    one_side_cost_.Gradient(g_rb, grad_f_rb);

    lx = w_plb_ * grad_f_lb * grad_gx_lb + w_prb_ * grad_f_rb * grad_gx_rb;
    lu = w_plb_ * grad_f_lb * grad_gu_lb + w_prb_ * grad_f_rb * grad_gu_rb;
    return true;
}

bool SafeDistanceToBoundaryConstraint::Hessian(const int step,
                                               const State &x,
                                               const Control &u,
                                               Eigen::Ref<MatrixLXX> lxx,
                                               Eigen::Ref<MatrixLUU> luu,
                                               Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step > steps_num_) {
        return false;
    }

    double g_lb{0.0};
    double g_rb{0.0};
    line_segment_constraint_lb_.Evaluate(step, x, u, g_lb);
    line_segment_constraint_rb_.Evaluate(step, x, u, g_rb);

    VecX grad_gx_lb, grad_gx_rb;
    VecU grad_gu_lb, grad_gu_rb;
    line_segment_constraint_lb_.Gradient(step, x, u, grad_gx_lb, grad_gu_lb);
    line_segment_constraint_rb_.Gradient(step, x, u, grad_gx_rb, grad_gu_rb);

    double grad_f_lb{0.0};
    double grad_f_rb{0.0};
    one_side_cost_.Gradient(g_lb, grad_f_lb);
    one_side_cost_.Gradient(g_rb, grad_f_rb);

    MatrixLXX hess_gxx_lb, hess_gxx_rb;
    MatrixLUU hess_guu_lb, hess_guu_rb;
    MatrixLXU hess_gxu_lb, hess_gxu_rb;
    line_segment_constraint_lb_.Hessian(step, x, u, hess_gxx_lb, hess_guu_lb, hess_gxu_lb);
    line_segment_constraint_rb_.Hessian(step, x, u, hess_gxx_rb, hess_guu_rb, hess_gxu_rb);

    double hess_f_lb{0.0};
    double hess_f_rb{0.0};
    one_side_cost_.Hessian(g_lb, hess_f_lb);
    one_side_cost_.Hessian(g_rb, hess_f_rb);

    lxx = w_plb_ * (hess_f_lb * grad_gx_lb * grad_gx_lb.transpose() + grad_f_lb * hess_gxx_lb) +
          w_prb_ * (hess_f_rb * grad_gx_rb * grad_gx_rb.transpose() + grad_f_rb * hess_gxx_rb);
    luu = w_plb_ * (hess_f_lb * grad_gu_lb * grad_gu_lb.transpose() + grad_f_lb * hess_guu_lb) +
          w_prb_ * (hess_f_rb * grad_gu_rb * grad_gu_rb.transpose() + grad_f_rb * hess_guu_rb);
    lxu = w_plb_ * (hess_f_lb * grad_gx_lb * grad_gu_lb.transpose() + grad_f_lb * hess_gxu_lb) +
          w_prb_ * (hess_f_rb * grad_gx_rb * grad_gu_rb.transpose() + grad_f_rb * hess_gxu_rb);
    return true;
}

SpeedLimitBySConstraint::SpeedLimitBySConstraint(const CostTermConfig &config,
                                                 const FrenetCoordinateSystem *fcs,
                                                 vector<double> s_pts,
                                                 vector<double> v_lim_pts)
    : fcs_(fcs), s_pts_(std::move(s_pts)), v_lim_pts_(std::move(v_lim_pts)) {
    horizon_ = config.steps.size();
    last_s_.assign(horizon_ + 1, 0.0);
    has_s_.assign(horizon_ + 1, false);
    last_p_.assign(horizon_ + 1, Vector2d::Zero());
    last_proj_.assign(horizon_ + 1, Projection{});
}

bool SpeedLimitBySConstraint::RawConstraints(const int step,
                                             const State &x,
                                             const Control &u,
                                             RawConstraintRows<double, 6, 2> &out) const {
    AppendRawConstraint<double, 6, 2>(*this, step, x, u, out);
    return true;
}

bool SpeedLimitBySConstraint::Evaluate(const int step, const State &x, const Control &u, double &val) const {
    val = 0.0;
    Projection p;
    if (step > horizon_ or s_pts_.empty() or !Project(step, x, p)) {
        return false;
    }
    double slope = 0.0;
    val = x(mVel) - InterpolatePiecewiseLinear(s_pts_, v_lim_pts_, p.s, &slope);
    return true;
}

bool SpeedLimitBySConstraint::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    Projection p;
    if (step > horizon_ or s_pts_.empty() or !Project(step, x, p)) {
        return false;
    }
    double slope = 0.0;
    (void)InterpolatePiecewiseLinear(s_pts_, v_lim_pts_, p.s, &slope);
    lx(mXPos) = -slope * p.ds_dx;
    lx(mYPos) = -slope * p.ds_dy;
    lx(mVel) = 1.0;
    return true;
}

bool SpeedLimitBySConstraint::Hessian(const int step,
                                      const State &x,
                                      const Control &u,
                                      Eigen::Ref<MatrixLXX> lxx,
                                      Eigen::Ref<MatrixLUU> luu,
                                      Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    return step <= horizon_;
}

bool SpeedLimitBySConstraint::Project(const int step, const State &x, Projection &p) const {
    const double px = x(mXPos), py = x(mYPos);
    // Evaluate and Gradient are usually called at the same position one after the other.
    if (has_s_[step] and px == last_p_[step].x() and py == last_p_[step].y()) {
        p = last_proj_[step];
        return true;
    }
    double s = 0.0;
    bool ok = false;
    if (has_s_[step]) {
        s = last_s_[step];
        ok = true;
        for (int k = 0; k < kNewtonSteps and ok; ++k) {
            const Point2D r = fcs_->GetRefCurvePoint(s);
            const Point2D tangent = fcs_->GetRefCurveUnitTangent(s);
            const double dx = px - r.x, dy = py - r.y;
            const double l = -tangent.y * dx + tangent.x * dy;
            const double denom = std::max(0.1, 1.0 - fcs_->GetRefCurveCurvature(s) * l);
            s += (tangent.x * dx + tangent.y * dy) / denom;
            ok = std::fabs(s - last_s_[step]) < kMaxLocalShift and s >= 0.0 and s <= fcs_->GetSlength();
        }
    }
    if (!ok) {
        Vec2d sl;
        if (fcs_->CartCoord2FrenetCoord(Vec2d(px, py), sl) != TRANSFORM_SUCCESS) {
            return false;
        }
        s = sl.x();
    }
    last_s_[step] = s;
    has_s_[step] = true;
    const Point2D r = fcs_->GetRefCurvePoint(s);
    const Point2D tangent = fcs_->GetRefCurveUnitTangent(s);
    const double l = -tangent.y * (px - r.x) + tangent.x * (py - r.y);
    const double scale = 1.0 / std::max(0.1, 1.0 - fcs_->GetRefCurveCurvature(s) * l);
    p.s = s;
    p.ds_dx = scale * tangent.x;
    p.ds_dy = scale * tangent.y;
    last_p_[step] = Vector2d(px, py);
    last_proj_[step] = p;
    return true;
}
