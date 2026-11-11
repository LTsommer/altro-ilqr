//
// Created by LTsommer on 2024/6/28.
//

#ifndef ALILQR_CONSTRAINTS_HPP
#define ALILQR_CONSTRAINTS_HPP

#include "../../ilqr_system_declaration.hpp"
#include "../../cost_function/cost_function.hpp"
#include "../../constraint_definition/constraint_formulation.h"
#include "../../constraint_definition/constraint.hpp"
#include "../../basic_constraint/basic_constraint.hpp"
#include "../mix_model.h"
#include "../../frenet_coordinate/frenet_coordinate_system.h"

/* linear acceleration limitation
 * acc_min < a < acc_max
  g(u) = a_ * u + b_[i] > 0
  represents u_min < u < u_max
  for u_min < u, g(u) = -u + u_min < 0
      u < u_max, g(u) = u - u_max < 0
 */
class AccLimitsConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    AccLimitsConstraint(const CostTermConfig &config, const bool is_lb) {
        horizon_ = config.steps.size();
        if (is_lb) {
            a_ = -1;
            b_ = config.acc_min;
        } else {
            a_ = 1;
            b_ = -config.acc_max;
        }
    }
    ~AccLimitsConstraint() = default;

    std::string GetName() const override { return std::string("AccLimitsConstraint"); }

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(AccLimitsConstraint)); }

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override {
        if (step > horizon_) {
            return false;
        }
        cost_val = a_ * x(mAcc) + b_;
        return true;
    }

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
        lx.setZero();
        lu.setZero();
        if (step > horizon_) {
            return false;
        }
        lx(mAcc) = a_;
        return true;
    }

    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override {
        lxx.setZero();
        luu.setZero();
        lxu.setZero();
        if (step > horizon_) {
            return false;
        }
        return true;
    };

private:
    int a_;
    double b_;
};

/* linear jerk limitation
 * acc_min < a < acc_max
  g(u) = a_ * u + b_[i] > 0
  represents u_min < u < u_max
  for u_min < u, g(u) = -u + u_min < 0
      u < u_max, g(u) = u - u_max < 0
 */
class JerkLimitsConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    JerkLimitsConstraint(const CostTermConfig &config, const bool is_lb) {
        horizon_ = config.steps.size();
        if (is_lb) {
            a_ = -1;
            b_ = config.jerk_min;
        } else {
            a_ = 1;
            b_ = -config.jerk_max;
        }
    }
    ~JerkLimitsConstraint() = default;

    std::string GetName() const override { return std::string("JerkLimitsConstraint"); }

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(JerkLimitsConstraint)); }

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override {
        if (step >= horizon_) {
            return false;
        }
        cost_val = a_ * u(mJerk) + b_;
        return true;
    }

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
        lx.setZero();
        lu.setZero();
        if (step >= horizon_) {
            return false;
        }
        lu(mJerk) = a_;
        return true;
    }

    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override {
        lxx.setZero();
        luu.setZero();
        lxu.setZero();
        if (step >= horizon_) {
            return false;
        }
        return true;
    };

private:
    int a_;
    double b_;
};

/* linear velocity limitation
 * vel_min < vel < acc_max
  g(u) = a_ * u + b_[i] > 0
  represents u_min < u < u_max
  for u_min < u, g(u) = -u + u_min < 0
      u < u_max, g(u) = u - u_max < 0
 */
class VelLimitsConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    VelLimitsConstraint(const CostTermConfig &config, const bool is_lb) {
        horizon_ = config.steps.size();
        if (is_lb) {
            a_ = -1;
            b_ = config.vel_min;
        } else {
            a_ = 1;
            b_ = -config.vel_max;
        }
    }
    ~VelLimitsConstraint() = default;

    std::string GetName() const override { return std::string("VelLimitsConstraint"); }

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(VelLimitsConstraint)); }

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override {
        if (step > horizon_) {
            return false;
        }
        cost_val = a_ * x(mVel) + b_;
        return true;
    }

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
        lx.setZero();
        lu.setZero();
        if (step > horizon_) {
            return false;
        }
        lx(mVel) = a_;
        return true;
    }

    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override {
        lxx.setZero();
        luu.setZero();
        lxu.setZero();
        if (step > horizon_) {
            return false;
        }
        return true;
    };

private:
    int a_;
    double b_;
};

/*
  g(u) = a_ * u + b_[i] > 0
  represents u_min < u < u_max
  for u_min < u, g(u) = -u + u_min < 0
      u < u_max, g(u) = u - u_max < 0
 */
class CurveLimitsConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    CurveLimitsConstraint(const CostTermConfig &config, const bool is_lb) {
        horizon_ = config.steps.size();
        if (is_lb) {
            a_ = -1;
            b_ = config.kappa_min;
        } else {
            a_ = 1;
            b_ = -config.kappa_max;
        }
    }
    ~CurveLimitsConstraint() = default;

    std::string GetName() const override { return std::string("CurveLimitsConstraint"); }

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(CurveLimitsConstraint)); }

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override {
        if (step > horizon_) {
            return false;
        }
        cost_val = a_ * x(mKappa) + b_;
        return true;
    }

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
        lx.setZero();
        lu.setZero();
        if (step > horizon_) {
            return false;
        }
        lx(mKappa) = a_;
        return true;
    }

    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override {
        lxx.setZero();
        luu.setZero();
        lxu.setZero();
        if (step > horizon_) {
            return false;
        }
        return true;
    }

private:
    int a_;
    double b_;
};

/*
  u = v^2 * kappa
  g(u) = a_ * u + b_[i] > 0
  represents u_min < u < u_max
  for u_min < u, g(u) = -u + u_min < 0
      u < u_max, g(u) = u - u_max < 0
 */
class CentripetalAccLimitsConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    CentripetalAccLimitsConstraint(const CostTermConfig &config, const bool is_lb) {
        horizon_ = config.steps.size();
        if (is_lb) {
            a_ = -1;
            b_ = config.lat_acc_min;
        } else {
            a_ = 1;
            b_ = -config.lat_acc_max;
        }
    }

    ~CentripetalAccLimitsConstraint() = default;

    std::string GetName() const override { return std::string("CentripetalAccLimitsConstraint"); }

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(CentripetalAccLimitsConstraint)); }

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override {
        if (step > horizon_) {
            return false;
        }
        cost_val = a_ * (sqr(x(mVel)) * x(mKappa)) + b_;
        return true;
    }

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
        lx.setZero();
        lu.setZero();
        if (step > horizon_) {
            return false;
        }
        lx(mVel) = 2 * a_ * x(mVel) * x(mKappa);
        lx(mKappa) = a_ * sqr(x(mVel));
        return true;
    }

    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override {
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

private:
    int a_;
    double b_;
};

/*
 *u = 2 * v * a * kappa + v^2 * dot_kappa
  g(u) = a_ * u + b_[i] > 0
  represents u_min < u < u_max
  for u_min < u, g(u) = -u + u_min < 0
      u < u_max, g(u) = u - u_max < 0
 */
class CentripetalJerkLimitsConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    CentripetalJerkLimitsConstraint(const CostTermConfig &config, const bool is_lb) {
        horizon_ = config.steps.size();
        if (is_lb) {
            a_ = -1;
            b_ = config.lat_jerk_min;
        } else {
            a_ = 1;
            b_ = -config.lat_jerk_max;
        }
    }

    ~CentripetalJerkLimitsConstraint() = default;

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(CentripetalJerkLimitsConstraint)); }

    std::string GetName() const override { return std::string("CentripetalJerkLimitsConstraint"); }

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override {
        if (step >= horizon_) {
            cost_val = 0.0;
            return false;
        }

        cost_val = a_ * (2 * x(mVel) * x(mAcc) * x(mKappa) + sqr(x(mVel)) * u(mDotKappa)) + b_;
        return true;
    }

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
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

    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override {
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

private:
    int a_;
    double b_;
};

/*
 *u = u - u_ref
  g(u) = a_ * u + b_[i] > 0
  represents u_min < u < u_max
  for u_min < u, g(u) = -u + u_min < 0
      u < u_max, g(u) = u - u_max < 0
 */
class HeadingDiffLimitsConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    HeadingDiffLimitsConstraint(const CostTermConfig &config, const bool is_lb) {
        horizon_ = config.steps.size();
        int sign = is_lb ? 1 : -1;
        const vector<double> &ref_yaw = config.yaw;
        int steps = config.yaw.size();
        a_ = is_lb ? -1 : 1;
        b_.resize(steps);
        for (int i = 0; i < steps; ++i) {
            double diff_yaw = is_lb ? config.yaw_diff_min : -config.yaw_diff_max;
            b_[i] = sign * ref_yaw[i] + diff_yaw;
        }
    }

    void UpdateConfig(const CostTermConfig &config, const bool is_lb) {
        int sign = is_lb ? 1 : -1;
        const vector<double> &ref_yaw = config.yaw;
        int steps = config.yaw.size();
        a_ = is_lb ? -1 : 1;
        b_.resize(steps);
        for (int i = 0; i < steps; ++i) {
            double diff_yaw = is_lb ? config.yaw_diff_min : -config.yaw_diff_max;
            b_[i] = sign * ref_yaw[i] + diff_yaw;
        }
    }

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(HeadingDiffLimitsConstraint)); }

    std::string GetName() const override { return std::string("HeadingDiffLimitsConstraint"); }

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override {
        if (step > horizon_) {
            cost_val = 0.0;
            return false;
        }
        cost_val = a_ * x(mTheta) + b_[step];
        return true;
    }

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
        lx.setZero();
        lu.setZero();
        if (step > horizon_) {
            return false;
        }
        lx(mTheta) = a_;
        return true;
    }

    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override {
        lxx.setZero();
        luu.setZero();
        lxu.setZero();
        if (step > horizon_) return false;
        return true;
    }

private:
    int a_;
    vector<double> b_;
};

/*----------- Constraint Penalty Functions Definitions-----------*/

class AccConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    AccConstraint(const CostTermConfig &config)
        : w_clb_(config.weights.at(WeightClb)),
          w_cub_(config.weights.at(WeightCub)),
          acc_limit_constraint_lb_(config, true),
          acc_limit_constraint_ub_(config, false) {
        horizon_ = config.steps.size();
    }

    ~AccConstraint() = default;

    std::string GetName() const override { return std::string("AccConstraint"); }

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(AccConstraint)); }

    // Raw sub-constraints g <= 0 behind the hinge, used by ALTRO polishing.
    bool RawConstraints(const int step,
                        const State &x,
                        const Control &u,
                        RawConstraintRows<double, 6, 2> &out) const override {
        AppendRawConstraint<double, 6, 2>(acc_limit_constraint_lb_, step, x, u, out);
        AppendRawConstraint<double, 6, 2>(acc_limit_constraint_ub_, step, x, u, out);
        return true;
    }

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override {
        if (step > horizon_) return false;

        double g_lb{0.0};
        double g_ub{0.0};
        acc_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
        acc_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

        double f_lb = one_side_cost_.Evaluate(g_lb);
        double f_ub = one_side_cost_.Evaluate(g_ub);

        cost_val = w_clb_ * f_lb + w_cub_ * f_ub;
        return true;
    };

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
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

    /* f(g)'' = f''(g)g'g' + f'(g)g''
     * */
    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override {
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
    };

private:
    const double w_clb_; // control lower bound weight
    const double w_cub_; // control upper bound weight
    const OneSideCost one_side_cost_;
    const AccLimitsConstraint acc_limit_constraint_lb_;
    const AccLimitsConstraint acc_limit_constraint_ub_;
};

class VelConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    VelConstraint(const CostTermConfig &config)
        : w_clb_(config.weights.at(WeightClb)),
          w_cub_(config.weights.at(WeightCub)),
          vel_limit_constraint_lb_(config, true),
          vel_limit_constraint_ub_(config, false) {
        horizon_ = config.steps.size();
    }

    ~VelConstraint() = default;

    std::string GetName() const override { return std::string("VelConstraint"); }

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(VelConstraint)); }

    // Raw sub-constraints g <= 0 behind the hinge, used by ALTRO polishing.
    bool RawConstraints(const int step,
                        const State &x,
                        const Control &u,
                        RawConstraintRows<double, 6, 2> &out) const override {
        AppendRawConstraint<double, 6, 2>(vel_limit_constraint_lb_, step, x, u, out);
        AppendRawConstraint<double, 6, 2>(vel_limit_constraint_ub_, step, x, u, out);
        return true;
    }

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override {
        if (step > horizon_) return false;

        double g_lb{0.0};
        double g_ub{0.0};
        vel_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
        vel_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

        double f_lb = one_side_cost_.Evaluate(g_lb);
        double f_ub = one_side_cost_.Evaluate(g_ub);

        cost_val = w_clb_ * f_lb + w_cub_ * f_ub;
        return true;
    };

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
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

    /* f(g)'' = f''(g)g'g' + f'(g)g''
     * */
    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override {
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
    };

private:
    const double w_clb_; // control lower bound weight
    const double w_cub_; // control upper bound weight
    const OneSideCost one_side_cost_;
    const VelLimitsConstraint vel_limit_constraint_lb_;
    const VelLimitsConstraint vel_limit_constraint_ub_;
};

class JerkConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    JerkConstraint(const CostTermConfig &config)
        : w_clb_(config.weights.at(WeightClb)),
          w_cub_(config.weights.at(WeightCub)),
          jerk_limit_constraint_lb_(config, true),
          jerk_limit_constraint_ub_(config, false) {
        horizon_ = config.steps.size();
    }

    ~JerkConstraint() = default;

    std::string GetName() const override { return std::string("JerkConstraint"); }

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(JerkConstraint)); }

    // Raw sub-constraints g <= 0 behind the hinge, used by ALTRO polishing.
    bool RawConstraints(const int step,
                        const State &x,
                        const Control &u,
                        RawConstraintRows<double, 6, 2> &out) const override {
        AppendRawConstraint<double, 6, 2>(jerk_limit_constraint_lb_, step, x, u, out);
        AppendRawConstraint<double, 6, 2>(jerk_limit_constraint_ub_, step, x, u, out);
        return true;
    }

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override {
        if (step >= horizon_) return false;

        double g_lb{0.0};
        double g_ub{0.0};
        jerk_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
        jerk_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

        double f_lb = one_side_cost_.Evaluate(g_lb);
        double f_ub = one_side_cost_.Evaluate(g_ub);

        cost_val = w_clb_ * f_lb + w_cub_ * f_ub;
        return true;
    };

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
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

    /* f(g)'' = f''(g)g'g' + f'(g)g''
     * */
    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override {
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
    };

private:
    const double w_clb_; // control lower bound weight
    const double w_cub_; // control upper bound weight
    const OneSideCost one_side_cost_;
    const JerkLimitsConstraint jerk_limit_constraint_lb_;
    const JerkLimitsConstraint jerk_limit_constraint_ub_;
};

/*
  J_ulimit = w_clb * f(g_clb(u)) + w_cub * f(g_cub(u))
  where f() is the penalty function, currently using one-sided quadratic
  function, and g_clb(u), g_cub(u) represents the control limit constraints
 */
class CurveConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    CurveConstraint(const CostTermConfig &config)
        : w_clb_(config.weights.at(WeightClb)),
          w_cub_(config.weights.at(WeightCub)),
          kappa_limit_constraint_lb_(config, true),
          kappa_limit_constraint_ub_(config, false) {
        horizon_ = config.steps.size();
    }

    ~CurveConstraint() = default;

    std::string GetName() const override { return std::string("CurveConstraint"); }

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(CurveLimitsConstraint)); }

    // Raw sub-constraints g <= 0 behind the hinge, used by ALTRO polishing.
    bool RawConstraints(const int step,
                        const State &x,
                        const Control &u,
                        RawConstraintRows<double, 6, 2> &out) const override {
        AppendRawConstraint<double, 6, 2>(kappa_limit_constraint_lb_, step, x, u, out);
        AppendRawConstraint<double, 6, 2>(kappa_limit_constraint_ub_, step, x, u, out);
        return true;
    }

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override {
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
    };

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
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
    };

    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override {
        lxx.setZero();
        luu.setZero();
        lxu.setZero();
        if (step > horizon_) return false;

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
    };

private:
    const double w_clb_; // control lower bound weight
    const double w_cub_; // control upper bound weight
    const OneSideCost one_side_cost_;
    const CurveLimitsConstraint kappa_limit_constraint_lb_;
    const CurveLimitsConstraint kappa_limit_constraint_ub_;
};

class CentripetalAccConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    CentripetalAccConstraint(const CostTermConfig &config)
        : w_clb_(config.weights.at(WeightClb)),
          w_cub_(config.weights.at(WeightCub)),
          centripetal_acc_limit_constraint_lb_(config, true),
          centripetal_acc_limit_constraint_ub_(config, false) {
        horizon_ = config.steps.size();
    }

    ~CentripetalAccConstraint() = default;

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(CentripetalAccConstraint)); }

    std::string GetName() const override { return std::string("CentripetalAccConstraint"); }

    // Raw sub-constraints g <= 0 behind the hinge, used by ALTRO polishing.
    bool RawConstraints(const int step,
                        const State &x,
                        const Control &u,
                        RawConstraintRows<double, 6, 2> &out) const override {
        AppendRawConstraint<double, 6, 2>(centripetal_acc_limit_constraint_lb_, step, x, u, out);
        AppendRawConstraint<double, 6, 2>(centripetal_acc_limit_constraint_ub_, step, x, u, out);
        return true;
    }

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override {
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
    };

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
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
    };

    /* f(g)'' = f''(g)g'g' + f'(g)g''
     * */
    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override {
        lxx.setZero();
        luu.setZero();
        lxu.setZero();
        if (step > horizon_) return false;

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
    };

private:
    const double w_clb_; // control lower bound weight
    const double w_cub_; // control upper bound weight
    const OneSideCost one_side_cost_;
    const CentripetalAccLimitsConstraint centripetal_acc_limit_constraint_lb_;
    const CentripetalAccLimitsConstraint centripetal_acc_limit_constraint_ub_;
};

class CentripetalJerkConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    CentripetalJerkConstraint(const CostTermConfig &config)
        : w_clb_(config.weights.at(WeightClb)),
          w_cub_(config.weights.at(WeightCub)),
          centripetal_jerk_limit_constraint_lb_(config, true),
          centripetal_jerk_limit_constraint_ub_(config, false) {
        horizon_ = config.steps.size();
    }

    ~CentripetalJerkConstraint() = default;

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(CentripetalJerkConstraint)); }

    std::string GetName() const override { return std::string("CentripetalJerkConstraint"); }

    // Raw sub-constraints g <= 0 behind the hinge, used by ALTRO polishing.
    bool RawConstraints(const int step,
                        const State &x,
                        const Control &u,
                        RawConstraintRows<double, 6, 2> &out) const override {
        AppendRawConstraint<double, 6, 2>(centripetal_jerk_limit_constraint_lb_, step, x, u, out);
        AppendRawConstraint<double, 6, 2>(centripetal_jerk_limit_constraint_ub_, step, x, u, out);
        return true;
    }

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override {
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
    };

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
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
    };

    /* f(g)'' = f''(g)g'g' + f'(g)g''
     * */
    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override {
        lxx.setZero();
        luu.setZero();
        lxu.setZero();
        if (step >= horizon_) return false;

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
    };

private:
    const double w_clb_; // control lower bound weight
    const double w_cub_; // control upper bound weight
    const OneSideCost one_side_cost_;
    const CentripetalJerkLimitsConstraint centripetal_jerk_limit_constraint_lb_;
    const CentripetalJerkLimitsConstraint centripetal_jerk_limit_constraint_ub_;
};

class HeadingDiffConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    HeadingDiffConstraint(const CostTermConfig &config)
        : w_clb_(config.weights.at(WeightClb)),
          w_cub_(config.weights.at(WeightCub)),
          heading_diff_limit_constraint_lb_(config, true),
          heading_diff_limit_constraint_ub_(config, false) {
        horizon_ = config.steps.size();
    }

    void UpdateConfig(const CostTermConfig &config) override {
        heading_diff_limit_constraint_lb_.UpdateConfig(config, true);
        heading_diff_limit_constraint_ub_.UpdateConfig(config, false);
    }

    ~HeadingDiffConstraint() = default;

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(HeadingDiffConstraint)); }

    std::string GetName() const override { return std::string("HeadingDiffConstraint"); }

    // Raw sub-constraints g <= 0 behind the hinge, used by ALTRO polishing.
    bool RawConstraints(const int step,
                        const State &x,
                        const Control &u,
                        RawConstraintRows<double, 6, 2> &out) const override {
        AppendRawConstraint<double, 6, 2>(heading_diff_limit_constraint_lb_, step, x, u, out);
        AppendRawConstraint<double, 6, 2>(heading_diff_limit_constraint_ub_, step, x, u, out);
        return true;
    }

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override {
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

    // (f(g))' = f'g'
    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
        lx.setZero();
        lu.setZero();
        if (step > horizon_) return false;

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

    // (f(g))'' = (f'(g)g')' = f''g'g' + f'g''
    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override {
        lxx.setZero();
        luu.setZero();
        lxu.setZero();
        if (step > horizon_) return false;

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

private:
    const double w_clb_; // control lower bound weight
    const double w_cub_; // control upper bound weight
    const OneSideCost one_side_cost_;
    HeadingDiffLimitsConstraint heading_diff_limit_constraint_lb_;
    HeadingDiffLimitsConstraint heading_diff_limit_constraint_ub_;
};

// TODO change to HuberLoss
class SafeDistanceToBoundaryConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    SafeDistanceToBoundaryConstraint(const CostTermConfig &config)
        : w_plb_(config.weights.at(WeightPlb)),
          w_prb_(config.weights.at(WeightPrb)),
          steps_num_(config.p_bound.size()),
          line_segment_constraint_lb_(config, true),
          line_segment_constraint_rb_(config, false) {
        horizon_ = config.steps.size();
    }

    ~SafeDistanceToBoundaryConstraint() = default;

    void UpdateConfig(const CostTermConfig &config) override {
        line_segment_constraint_lb_.UpdateConfig(config, true);
        line_segment_constraint_rb_.UpdateConfig(config, false);
    }

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(SafeDistanceToBoundaryConstraint)); }

    virtual std::string GetName() const override { return std::string("SafeDistanceToBoundaryConstraint"); }

    // Raw sub-constraints g <= 0 behind the hinge, used by ALTRO polishing.
    bool RawConstraints(const int step,
                        const State &x,
                        const Control &u,
                        RawConstraintRows<double, 6, 2> &out) const override {
        AppendRawConstraint<double, 6, 2>(line_segment_constraint_lb_, step, x, u, out);
        AppendRawConstraint<double, 6, 2>(line_segment_constraint_rb_, step, x, u, out);
        return true;
    }

    bool Evaluate(const int step, const State &x, const Control &u, double &val) const override {
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
    };

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
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
    };

    // f(g)'' = f'' * g' * g' + f' * g''
    virtual bool Hessian(const int step,
                         const State &x,
                         const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const override {
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
    };

private:
    const double w_plb_;       // path left bound weight
    const double w_prb_;       // path right bound weight
    const uint32_t steps_num_; // total number of steps
    const OneSideCost one_side_cost_;
    LateralOffsetConstraint<double, 6, 2> line_segment_constraint_lb_;
    LateralOffsetConstraint<double, 6, 2> line_segment_constraint_rb_;
};

/*
  Position-dependent speed limit, e.g. lower speed in narrow passages:
      g(x) = v - v_lim(s(x)) <= 0
  s(x) is the projection of the position onto the reference line and v_lim(s) is linear
  between the given points (constant outside). The limit follows where the vehicle actually is,
  not the time step, so it needs no re-anchoring by the solver.

  The global projection onto the reference line (search for the nearest point) is expensive and
  the solver evaluates every step many times with positions that barely move, so each step keeps
  its last s and refines it with a few local Newton steps on the reference curve r(s):
      s <- s + tangent(s) . (p - r(s)) / (1 - kappa(s) * l)
  This gives the exact projection; the global search is only used for the first evaluation or
  when the local iteration moves too far.
 */
class SpeedLimitBySConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    static constexpr int kNewtonSteps = 2;
    static constexpr double kMaxLocalShift = 2.0; // m of s; beyond this, project globally

    SpeedLimitBySConstraint(const CostTermConfig &config,
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

    std::string GetName() const override { return std::string("SpeedLimitBySConstraint"); }

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(SpeedLimitBySConstraint)); }

    bool RawConstraints(const int step,
                        const State &x,
                        const Control &u,
                        RawConstraintRows<double, 6, 2> &out) const override {
        AppendRawConstraint<double, 6, 2>(*this, step, x, u, out);
        return true;
    }

    bool Evaluate(const int step, const State &x, const Control &u, double &val) const override {
        val = 0.0;
        Projection p;
        if (step > horizon_ or s_pts_.empty() or !Project(step, x, p)) return false;
        double slope = 0.0;
        val = x(mVel) - Limit(p.s, slope);
        return true;
    }

    // dg/dv = 1; dg/d(x, y) = -dv_lim/ds * ds/d(x, y), ds/d(x, y) = tangent / (1 - kappa * l).
    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
        lx.setZero();
        lu.setZero();
        Projection p;
        if (step > horizon_ or s_pts_.empty() or !Project(step, x, p)) return false;
        double slope = 0.0;
        (void)Limit(p.s, slope);
        lx(mXPos) = -slope * p.ds_dx;
        lx(mYPos) = -slope * p.ds_dy;
        lx(mVel) = 1.0;
        return true;
    }

    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override {
        lxx.setZero();
        luu.setZero();
        lxu.setZero();
        return step <= horizon_;
    }

private:
    struct Projection {
        double s = 0.0;
        double ds_dx = 0.0;
        double ds_dy = 0.0;
    };

    // Exact projection of the position onto the reference line, warm-started from the
    // previous s of this step. Returns false if it cannot be projected at all.
    bool Project(const int step, const State &x, Projection &p) const {
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
            if (fcs_->CartCoord2FrenetCoord(Vec2d(px, py), sl) != TRANSFORM_SUCCESS) return false;
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

    // v_lim(s) and its slope dv_lim/ds.
    double Limit(const double s, double &slope) const {
        slope = 0.0;
        if (s <= s_pts_.front()) return v_lim_pts_.front();
        if (s >= s_pts_.back()) return v_lim_pts_.back();
        const size_t i = std::upper_bound(s_pts_.begin(), s_pts_.end(), s) - s_pts_.begin();
        const double ds = s_pts_[i] - s_pts_[i - 1];
        if (ds <= 1e-9) return std::min(v_lim_pts_[i - 1], v_lim_pts_[i]);
        slope = (v_lim_pts_[i] - v_lim_pts_[i - 1]) / ds;
        return v_lim_pts_[i - 1] + slope * (s - s_pts_[i - 1]);
    }

    const FrenetCoordinateSystem *fcs_;
    vector<double> s_pts_;
    vector<double> v_lim_pts_;
    mutable std::vector<double> last_s_; // warm start of the projection, per step
    mutable std::vector<bool> has_s_;
    mutable std::vector<Vector2d, Eigen::aligned_allocator<Vector2d>> last_p_; // position of last_proj_
    mutable std::vector<Projection> last_proj_;
};

#endif //ALILQR_CONSTRAINTS_HPP
