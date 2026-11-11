//
// Created by LTsommer on 2024/6/28.
//

#ifndef ALILQR_COST_TERMS_HPP
#define ALILQR_COST_TERMS_HPP

#include "../../ilqr_system_declaration.hpp"
#include "../../cost_function/cost_function.hpp"
#include "../cost_function/cost_term_config.hpp"
#include "../mix_model.h"

/*
 * state     [x, y, theta, kappa, vel, acc]
 * control   [jerk, kappa_dot]
 * state_space function f(x, u) is implemented in mix_model.h
 * calculus is implemented in calculus.h by using a Gauss-Legendre Integration.
 * */

/* final state cost function is working when step = horizon
 * otherwise cost function is working as the running cost term
 * */

/* centripetal acceleration term = 1/2 * w * (vel^2 * kappa)^2
 * */
class CentripetalAccCostFunc final : public CostFunc<double, 6, 2> {
public:
    CentripetalAccCostFunc(const CostTermConfig &config) : w_lat_acc_(config.weights.at(WeightIndex::WeightLatAcc)) {
        SetHorizon(config.steps.size());
    }

    ~CentripetalAccCostFunc() = default;

    virtual std::type_index GetTypeIndex() const override { return std::type_index(typeid(CentripetalAccCostFunc)); }

    virtual std::string GetName() const override { return std::string("CentripetalAccCostFunc"); }

    virtual bool Evaluate(const int step, const State &x, const Control &u, double &val) const override {
        if (step > horizon) return false;

        val = 0.5 * sqr(sqr(x(mVel)) * x(mKappa)) * w_lat_acc_;
        return true;
    };

    /* df / dv: w * kappa^2 * 2 * vel^3
     * df / dk: w * vel^4 * kappa
     * */
    virtual bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
        lx.setZero();
        lu.setZero();
        if (step > horizon) return false;

        lx(mVel) = doppel(cube(x(mVel)) * sqr(x(mKappa))) * w_lat_acc_;
        lx(mKappa) = sqr(sqr(x(mVel))) * x(mKappa) * w_lat_acc_;
        return true;
    };

    /* df / dv: w * 2 * kappa^2 * vel^3
     * df / dk: w * vel^4 * kappa
     * */
    virtual bool Hessian(const int step,
                         const State &x,
                         const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const override {
        lxx.setZero();
        luu.setZero();
        lxu.setZero();
        if (step > horizon) return false;

        lxx(mVel, mVel) = doppel(3 * sqr(x(mVel)) * sqr(x(mKappa))) * w_lat_acc_;
        lxx(mVel, mKappa) = doppel(2 * cube(x(mVel)) * x(mKappa)) * w_lat_acc_;
        lxx(mKappa, mKappa) = sqr(x(mVel)) * sqr(x(mVel)) * w_lat_acc_;
        lxx(mKappa, mVel) = lxx(mVel, mKappa); // was 3 * v^3 * kappa * w; must be symmetric (4 * v^3 * kappa * w)
        return true;
    };

private:
    double w_lat_acc_;
};

/* centripetal jerk term = 1/2 * w * (2 * vel * acc * kappa + vel ^ 2 * kappa_dot)^2
 * */
class CentripetalJerkCostFunc final : public CostFunc<double, 6, 2> {
public:
    CentripetalJerkCostFunc(const CostTermConfig &config) : w_lat_jerk_(config.weights.at(WeightIndex::WeightLatJerk)) {
        SetHorizon(config.steps.size());
    }

    ~CentripetalJerkCostFunc() = default;

    virtual std::type_index GetTypeIndex() const override { return std::type_index(typeid(CentripetalJerkCostFunc)); }

    virtual std::string GetName() const override { return std::string("CentripetalJerkCostFunc"); }

    // Lateral jerk 2 * v * a * kappa + v^2 * kappa_dot, computed from the (x, u) being evaluated.
    // (It used to be cached by SetCoeTerm(), which was never called, so this cost was always 0.)
    static double CoeTerm(const State &x, const Control &u) {
        return doppel(x(mVel) * x(mAcc) * x(mKappa)) + sqr(x(mVel)) * u(mDotKappa);
    }

    //    centripetal jerk term = 1/2 * w * (2 * vel * acc * kappa + vel ^ 2 * kappa_dot)^2
    virtual bool Evaluate(const int step, const State &x, const Control &u, double &val) const override {
        if (step >= horizon) {
            val = 0.0;
            return false;
        }

        const double coe_term = CoeTerm(x, u);
        val = 0.5 * sqr(coe_term) * w_lat_jerk_;
        return true;
    };

    /* df / dv = w * (2 * vel * acc * kappa + vel ^ 2 * kappa_dot) * (2 * acc * kappa + 2 * vel * kappa_dot)
     * df / dk = w * (2 * vel * acc * kappa + vel ^ 2 * kappa_dot) * (2 * vel * acc)
     * df / da = w * (2 * vel * acc * kappa + vel ^ 2 * kappa_dot) * (2 * vel * kappa)
     * df / ddk = w * (2 * vel * acc * kappa + vel ^ 2 * kappa_dot) * vel^2
     * */
    virtual bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
        lx.setZero();
        lu.setZero();
        if (step >= horizon) return false;
        const double coe_term = CoeTerm(x, u);
        lx(mVel) = coe_term * doppel(x(mAcc) * x(mKappa) + x(mVel) * u(mDotKappa)) * w_lat_jerk_;
        lx(mKappa) = coe_term * doppel(x(mVel) * x(mAcc)) * w_lat_jerk_;
        lx(mAcc) = coe_term * doppel(x(mVel) * x(mKappa)) * w_lat_jerk_;
        lu(mDotKappa) = coe_term * sqr(x(mVel)) * w_lat_jerk_;
        return true;
    };

    virtual bool Hessian(const int step,
                         const State &x,
                         const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const override {
        lxx.setZero();
        luu.setZero();
        lxu.setZero();
        if (step >= horizon) return false;
        const double coe_term = CoeTerm(x, u);
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
                (doppel(x(mAcc) * x(mKappa) + x(mVel) * u(mDotKappa)) * dk_term + coe_term * doppel(x(mAcc))) *
                w_lat_jerk_;
        lxx(mKappa, mAcc) = (doppel(x(mVel) * x(mKappa)) * dk_term + coe_term * doppel(x(mVel))) * w_lat_jerk_;
        lxx(mKappa, mKappa) = doppel(x(mVel) * x(mAcc)) * dk_term * w_lat_jerk_;
        // d(df/da) / dv, da, dk
        lxx(mAcc, mVel) =
                (doppel(x(mAcc) * x(mKappa) + x(mVel) * u(mDotKappa)) * da_term + coe_term * doppel(x(mKappa))) *
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
    };

private:
    double w_lat_jerk_;
};

/* dot_kappa cost term = 1/2 * w * dot_kappa^2
 * */
class DotCurveCostFunc final : public CostFunc<double, 6, 2> {
public:
    DotCurveCostFunc(const CostTermConfig &config) : w_dot_jerk_(config.weights.at(WeightIndex::WeightDotKappa)) {
        SetHorizon(config.steps.size());
    }

    ~DotCurveCostFunc() = default;

    virtual std::type_index GetTypeIndex() const override { return std::type_index(typeid(DotCurveCostFunc)); }

    virtual std::string GetName() const override { return std::string("DotCurveCostFunc"); }

    virtual bool Evaluate(const int step, const State &x, const Control &u, double &val) const override {
        if (step >= horizon) {
            val = 0.0;
            return false;
        }

        val = 0.5 * sqr(u(mDotKappa)) * w_dot_jerk_;
        return true;
    };

    virtual bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
        lx.setZero();
        lu.setZero();
        if (step >= horizon) return false;

        lu(mDotKappa) = w_dot_jerk_ * u(mDotKappa);
        return true;
    };

    virtual bool Hessian(const int step,
                         const State &x,
                         const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const override {
        lxx.setZero();
        luu.setZero();
        lxu.setZero();
        if (step >= horizon) return false;

        luu(mDotKappa, mDotKappa) = w_dot_jerk_;
        return true;
    };

private:
    double w_dot_jerk_;
};

/* linear jerk cost term = 1/2 * w * jerk^2
 * */
class LinearJerkCostFunc final : public CostFunc<double, 6, 2> {
public:
    LinearJerkCostFunc(const CostTermConfig &config) : w_jerk_(config.weights.at(WeightIndex::WeightJerk)) {
        SetHorizon(config.steps.size());
    }

    ~LinearJerkCostFunc() = default;

    virtual std::type_index GetTypeIndex() const override { return std::type_index(typeid(LinearJerkCostFunc)); }

    virtual std::string GetName() const override { return std::string("LinearJerkCostFunc"); }

    virtual bool Evaluate(const int step, const State &x, const Control &u, double &val) const override {
        if (step >= horizon) {
            val = 0.0;
            return false;
        }

        val = 0.5 * sqr(u(mJerk)) * w_jerk_;
        return true;
    };

    virtual bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
        lx.setZero();
        lu.setZero();
        if (step >= horizon) return false;

        lu(mJerk) = w_jerk_ * u(mJerk);
        return true;
    };

    virtual bool Hessian(const int step,
                         const State &x,
                         const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const override {
        lxx.setZero();
        luu.setZero();
        lxu.setZero();
        if (step >= horizon) return false;

        luu(mJerk, mJerk) = w_jerk_;
        return true;
    };

private:
    double w_jerk_;
};

// TODO change to HuberLoss
class TargetSpeedCostFunc final : public CostFunc<double, 6, 2> {
public:
    TargetSpeedCostFunc(const CostTermConfig &config)
        : w_vel_diff_(config.weights.at(WeightVelDiff)), velocity_(config.velocity) {
        SetHorizon(config.steps.size());
    };

    ~TargetSpeedCostFunc() = default;

    virtual std::type_index GetTypeIndex() const override { return std::type_index(typeid(TargetSpeedCostFunc)); }

    void UpdateConfig(const CostTermConfig &config) override { velocity_ = config.velocity; }

    std::string GetName() const override { return std::string("TargetSpeedCostFunc"); }

    virtual bool Evaluate(const int step, const State &x, const Control &u, double &val) const override {
        if (step > horizon) {
            val = 0.0;
            return false;
        }

        val = 0.5 * w_vel_diff_ * sqr<double>(x(mVel) - velocity_[step]);
        return true;
    };

    virtual bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
        lx.setZero();
        lu.setZero();
        if (step > horizon) return false;

        lx(mVel) = w_vel_diff_ * (x(mVel) - velocity_[step]);
        return true;
    };

    virtual bool Hessian(const int step,
                         const State &x,
                         const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const override {
        lxx.setZero();
        luu.setZero();
        lxu.setZero();
        if (step > horizon) // same range as Evaluate / Gradient (was >=, dropping the last step)
            return false;

        lxx(mVel, mVel) = w_vel_diff_;
        return true;
    };

private:
    double w_vel_diff_;
    vector<double> velocity_;
};

#endif //ALILQR_COST_TERMS_HPP
