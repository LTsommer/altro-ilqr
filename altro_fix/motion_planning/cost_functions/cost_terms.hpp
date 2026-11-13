//
// Created by LTsommer on 2024/6/28.
//

#ifndef ALILQR_COST_TERMS_HPP
#define ALILQR_COST_TERMS_HPP

#include "../../ilqr_system_declaration.hpp"
#include "../../common/math_utils.h"
#include "../../cost_function/cost_function.hpp"
#include "../cost_function/cost_term_config.hpp"
#include "../mix_model.h"
#include "../lateral_kinematics.h"

/*
 * state     [x, y, theta, kappa, vel, acc]
 * control   [jerk, kappa_dot]
 * state_space function f(x, u) is implemented in mix_model.h
 * calculus is implemented in common/calculus.h by using a Gauss-Legendre Integration.
 * */

/* final state cost function is working when step = horizon
 * otherwise cost function is working as the running cost term
 * */

/* centripetal acceleration term = 1/2 * w * (vel^2 * kappa)^2
 * */
class CentripetalAccCostFunc final : public CostFunc<double, 6, 2> {
public:
    CentripetalAccCostFunc(const CostTermConfig &config);

    ~CentripetalAccCostFunc() = default;

    virtual std::type_index GetTypeIndex() const override { return std::type_index(typeid(CentripetalAccCostFunc)); }

    virtual std::string GetName() const override { return std::string("CentripetalAccCostFunc"); }

    virtual bool Evaluate(const int step, const State &x, const Control &u, double &val) const override;

    /* df / dv: w * kappa^2 * 2 * vel^3
     * df / dk: w * vel^4 * kappa
     * */
    virtual bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    /* df / dv: w * 2 * kappa^2 * vel^3
     * df / dk: w * vel^4 * kappa
     * */
    virtual bool Hessian(const int step,
                         const State &x,
                         const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const override;

private:
    double w_lat_acc_;
};

/* centripetal jerk term = 1/2 * w * (2 * vel * acc * kappa + vel ^ 2 * kappa_dot)^2
 * */
class CentripetalJerkCostFunc final : public CostFunc<double, 6, 2> {
public:
    CentripetalJerkCostFunc(const CostTermConfig &config);

    ~CentripetalJerkCostFunc() = default;

    virtual std::type_index GetTypeIndex() const override { return std::type_index(typeid(CentripetalJerkCostFunc)); }

    virtual std::string GetName() const override { return std::string("CentripetalJerkCostFunc"); }

    //    centripetal jerk term = 1/2 * w * (2 * vel * acc * kappa + vel ^ 2 * kappa_dot)^2
    virtual bool Evaluate(const int step, const State &x, const Control &u, double &val) const override;

    /* df / dv = w * (2 * vel * acc * kappa + vel ^ 2 * kappa_dot) * (2 * acc * kappa + 2 * vel * kappa_dot)
     * df / dk = w * (2 * vel * acc * kappa + vel ^ 2 * kappa_dot) * (2 * vel * acc)
     * df / da = w * (2 * vel * acc * kappa + vel ^ 2 * kappa_dot) * (2 * vel * kappa)
     * df / ddk = w * (2 * vel * acc * kappa + vel ^ 2 * kappa_dot) * vel^2
     * */
    virtual bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    virtual bool Hessian(const int step,
                         const State &x,
                         const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const override;

private:
    double w_lat_jerk_;
};

/* dot_kappa cost term = 1/2 * w * dot_kappa^2
 * */
class DotCurveCostFunc final : public CostFunc<double, 6, 2> {
public:
    DotCurveCostFunc(const CostTermConfig &config);

    ~DotCurveCostFunc() = default;

    virtual std::type_index GetTypeIndex() const override { return std::type_index(typeid(DotCurveCostFunc)); }

    virtual std::string GetName() const override { return std::string("DotCurveCostFunc"); }

    virtual bool Evaluate(const int step, const State &x, const Control &u, double &val) const override;

    virtual bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    virtual bool Hessian(const int step,
                         const State &x,
                         const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const override;

private:
    double w_dot_jerk_;
};

/* linear jerk cost term = 1/2 * w * jerk^2
 * */
class LinearJerkCostFunc final : public CostFunc<double, 6, 2> {
public:
    LinearJerkCostFunc(const CostTermConfig &config);

    ~LinearJerkCostFunc() = default;

    virtual std::type_index GetTypeIndex() const override { return std::type_index(typeid(LinearJerkCostFunc)); }

    virtual std::string GetName() const override { return std::string("LinearJerkCostFunc"); }

    virtual bool Evaluate(const int step, const State &x, const Control &u, double &val) const override;

    virtual bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    virtual bool Hessian(const int step,
                         const State &x,
                         const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const override;

private:
    double w_jerk_;
};

// TODO change to HuberLoss
class TargetSpeedCostFunc final : public CostFunc<double, 6, 2> {
public:
    TargetSpeedCostFunc(const CostTermConfig &config);
    ;

    ~TargetSpeedCostFunc() = default;

    virtual std::type_index GetTypeIndex() const override { return std::type_index(typeid(TargetSpeedCostFunc)); }

    void UpdateConfig(const CostTermConfig &config) override { velocity_ = config.velocity; }

    std::string GetName() const override { return std::string("TargetSpeedCostFunc"); }

    virtual bool Evaluate(const int step, const State &x, const Control &u, double &val) const override;

    virtual bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    virtual bool Hessian(const int step,
                         const State &x,
                         const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const override;

private:
    double w_vel_diff_;
    vector<double> velocity_;
};

#endif //ALILQR_COST_TERMS_HPP
