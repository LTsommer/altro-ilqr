//
// Created by LTsommer on 2024/5/30.
//

#ifndef CILQR_COST_TERMS_H
#define CILQR_COST_TERMS_H


#include "cost_function.hpp"
#include "../frenet_coordinate/vec2d.h"
#include "../frenet_coordinate/frenet_coordinate_system.h"
#include "cost_term_config.hpp"
#include "../constrain_definition/constraint_formulation.h"
#include <memory>

/*
  J_acc = 1/2 * w_acc * acc^2
  J_acc = 1/2 * w_acc * u.T * [1 0;0 0] * u
 */
template<typename T, unsigned M, unsigned int N>
class AccCostFunc final : public CostFunc<T, M, N> {
public:
    OCP_VARIABLES(T, M, N)
    AccCostFunc(const CostTermConfig &config):
    w_acc_(config.weights.at(WeightIndex::WeightAcc))
    { CostFunc<T, M, N>::set_horizon(config.steps.size());}

    ~AccCostFunc() = default;

    virtual std::type_index get_type_index() override {
        return std::type_index(typeid(AccCostFunc));
    }

    virtual bool Evaluate(const int step,
                          const State &state,
                          const Control &ctrl,
                          double &val) const override;

    virtual bool Gradient(const int step, const State &x, const Control &u,
                          Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    virtual bool Hessian(const int step, const State &x, const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const override;
private:
    double w_acc_;
};

/*
  J = 1/2 * w_acc * yr^2
  J = 1/2 * w_acc * u.T * [0 0;0 1] * u
 */
template<typename T, unsigned M, unsigned int N>
class YawRateCostFunc final : public CostFunc<T, M, N>{
public:
    OCP_VARIABLES(T, M, N)
    YawRateCostFunc(const CostTermConfig &config):
    w_yaw_rate_(config.weights.at(WeightIndex::WeightYawRate))
    { CostFunc<T, M, N>::set_horizon(config.steps.size());};

    ~YawRateCostFunc() = default;

    virtual std::type_index get_type_index() override {
        return std::type_index(typeid(YawRateCostFunc));
    }

    virtual bool Evaluate(const int step,
                          const State &state,
                          const Control &ctrl,
                          double &val) const override;

    virtual bool Gradient(const int step, const State &x, const Control &u,
                          Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    virtual bool Hessian(const int step, const State &x, const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const override;
private:
    double w_yaw_rate_;
};

/*
  J = 1/2 * w_ref_offset * l^2
 */
template<typename T, unsigned M, unsigned int N>
class OffsetCostFunc final : public CostFunc<T, M, N> {
public:
    OCP_VARIABLES(T, M, N);
    OffsetCostFunc(const CostTermConfig &config):
    w_offset_(config.weights.at(WeightIndex::WeightRefOffset)),
    p_ref_(config.p_ref)
    { CostFunc<T, M, N>::set_horizon(config.steps.size());};

    ~OffsetCostFunc() = default;

    virtual std::type_index get_type_index() override {
        return std::type_index(typeid(OffsetCostFunc));
    }

    virtual bool Evaluate(const int step,
                          const State &x,
                          const Control &u,
                          double &val) const override;

    virtual bool Gradient(const int step, const State &x, const Control &u,
                          Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    virtual bool Hessian(const int step, const State &x, const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const override;
private:
    double w_offset_;
    const std::vector<Vector2d> p_ref_;
};

/*
  J = 1/2 * w_vel_diff * (v - v_ref)^2
 */
template<typename T, unsigned M, unsigned int N>
class VelDiffCostFunc final : public CostFunc<T, M, N> {
public:
    OCP_VARIABLES(T, M, N);
    VelDiffCostFunc(const CostTermConfig &config):
    w_vel_diff_(config.weights.at(WeightVelDiff)),
    velocity_(config.velocity)
    { CostFunc<T, M, N>::set_horizon(config.steps.size());};

    ~VelDiffCostFunc() = default;

    virtual std::type_index get_type_index() override {
        return std::type_index(typeid(VelDiffCostFunc));
    }

    virtual bool Evaluate(const int step,
                          const State &state,
                          const Control &ctrl,
                          double &val) const override;

    virtual bool Gradient(const int step, const State &x, const Control &u,
                          Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    virtual bool Hessian(const int step, const State &x, const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const override;

private:
    double w_vel_diff_;
    vector<double> velocity_;
};

/* J = 1 / 2 * w_k * kappa ^ 2 + 1 / 2 * w_dk * dkappa^ 2 + 1/ 2 w_u ddkappa ^ 2
 * */
template<typename T, unsigned M, unsigned int N>
class CurvatureCostFunc final : public CostFunc<T, M, N> {
public:
    OCP_VARIABLES(T, M, N);
    CurvatureCostFunc(const CostTermConfig &config):
    w_dkappa_(config.weights.at(WeightKappa)),
    w_kappa_(config.weights.at(WeightKappa)),
    w_u_(config.weights.at(WeightU))
    { CostFunc<T, M, N>::set_horizon(config.steps.size());};

    ~CurvatureCostFunc() = default;

    virtual std::type_index get_type_index() override {
        return std::type_index(typeid(CurvatureCostFunc));
    }

    virtual bool Evaluate(const int step,
                          const State &state,
                          const Control &ctrl,
                          double &val) const override;

    virtual bool Gradient(const int step, const State &x, const Control &u,
                          Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    virtual bool Hessian(const int step, const State &x, const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const override;



private:
    const double w_kappa_;  // curvature weight
    const double w_dkappa_; // curvature change rate weight
    const double w_u_;      // control input weight
};


/*
  J_comfort = 1/2 * w_acc * (V^2 * kappa)^2
            + 1/2 * w_jerk * (3 * V * A * kappa + V^3 * dkappa)^2
  where V is the velocity, A is the acceleration, kappa is the curvature, and
  dkappa is the curvature's rate of change
 */
template<typename T, unsigned M, unsigned int N>
class LateralComfortCostFunc final : public CostFunc<T, M, N> {
public:
    OCP_VARIABLES(T, M, N);
    LateralComfortCostFunc(const CostTermConfig &config)
            : w_acc_(config.weights.at(WeightAcc)),
              w_jerk_(config.weights.at(WeightJerk)),
              velocity_(config.velocity),
              acceleration_(config.acceleration) {
        cross_coeff_.resize(velocity_.size());
        for (size_t i = 0; i < velocity_.size(); i++) {
            cross_coeff_[i] = 3 * w_jerk_ * pow(velocity_[i], 4) * acceleration_[i];
        }
        CostFunc<T, M, N>::set_horizon(config.steps.size());
    }

    ~LateralComfortCostFunc() = default;

    virtual std::type_index get_type_index() override {
        return std::type_index(typeid(LateralComfortCostFunc));
    }

    virtual bool Evaluate(const int step,
                          const State &state,
                          const Control &ctrl,
                          double &val) const override;

    virtual bool Gradient(const int step, const State &x, const Control &u,
                          Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    virtual bool Hessian(const int step, const State &x, const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const override;

private:
    const double w_acc_;               // acceleration weight
    const double w_jerk_;              // jerk weight
    std::vector<double> velocity_;     // longitudinal speed
    std::vector<double> acceleration_; // longitudinal acceleration
    std::vector<double>
            cross_coeff_; // cross_coeff_ = 3 * w_jerk_ * V^4 * \dot(V);
};

/*--------------------calculate the cost of the final state--------------------*/
/*
   J_N = 1/2 * x_N.T * Q * x_N
 */
//template<typename T, unsigned M, unsigned int N>
//class FinalAccCostFunc final : public FinalStateCostFunc<T, M, N> {
//public:
//    OCP_VARIABLES(T, M, N);
//    FinalAccCostFunc(const CostTermConfig &config)
//    :acc_(config.acceleration.back()),
//    w_acc_(config.weights.at(WeightAcc))
//    {};
//
//    ~FinalAccCostFunc() = default;
//
//    virtual std::type_index get_type_index() override{
//        return std::type_index(typeid(FinalAccCostFunc));
//    }
//
//    virtual double value(const State &state) const override;
//
//    virtual void gradient(const State &state, VecX &g) const override;
//
//    virtual void hessian(const State &state, MatrixLXX &h) const override;
//
//private:
//    double acc_;
//    double w_acc_;
//};

//template<typename T, unsigned M, unsigned int N>
//class FinalYawRateCostFunc final : public FinalStateCostFunc<T, M, N> {
//public:
//    OCP_VARIABLES(T, M, N);
//    FinalYawRateCostFunc(const CostTermConfig &config)
//    :yaw_rate_(config.final_yaw_rate),
//    w_yaw_rate_(config.weights.at(WeightYawRate))
//    {};
//
//    ~FinalYawRateCostFunc() = default;
//
//    virtual std::type_index get_type_index() override{
//        return std::type_index(typeid(FinalYawRateCostFunc));
//    }
//
//    virtual double value(const State &state) const override;
//
//    virtual void gradient(const State &state, VecX &g) const override;
//
//    virtual void hessian(const State &state, MatrixLXX &h) const override;
//
//private:
//    double yaw_rate_;
//    double w_yaw_rate_;
//};

//template<typename T, unsigned M, unsigned int N>
//class FinalOffsetCostFunc final : public FinalStateCostFunc<T, M, N> {
//public:
//    OCP_VARIABLES(T, M, N);
//    FinalOffsetCostFunc(const CostTermConfig &config):
//    w_offset_(config.weights.at(WeightRefOffset)),
//    p_ref_(config.p_ref.back())
//    {};
//
//    ~FinalOffsetCostFunc() = default;
//
//    virtual std::type_index get_type_index() override{
//        return std::type_index(typeid(FinalOffsetCostFunc));
//    }
//
//    virtual double value(const State &state) const override;
//
//    virtual void gradient(const State &state, VecX &g) const override;
//
//    virtual void hessian(const State &state, MatrixLXX &h) const override;
//
//private:
//    double w_offset_;
//    Vector2d p_ref_;
//};

//template<typename T, unsigned M, unsigned int N>
//class FinalVelDiffCostFunc final : public FinalStateCostFunc<T, M, N> {
//public:
//    OCP_VARIABLES(T, M, N);
//    FinalVelDiffCostFunc(const CostTermConfig &config):
//    w_vel_diff_(config.weights.at(WeightVelDiff)),
//    vel(config.velocity.back())
//    {};
//
//    ~FinalVelDiffCostFunc() = default;
//
//    virtual std::type_index get_type_index() override{
//        return std::type_index(typeid(FinalVelDiffCostFunc));
//    }
//
//    virtual double value(const State &state) const override;
//
//    virtual void gradient(const State &state, VecX &g) const override;
//
//    virtual void hessian(const State &state, MatrixLXX &h) const override;
//
//private:
//    double w_vel_diff_;
//    double vel;
//};

/*
  J_comfort = 1/2 * w_acc * (V^2 * kappa)^2
            + 1/2 * w_jerk * (3 * V * A * kappa + V^3 * dkappa)^2
  where V is the velocity, A is the acceleration, kappa is the curvature, and
  dkappa is the curvature's rate of change
 */
//template<typename T, unsigned M, unsigned int N>
//class LateralComfortFinalCostFunc final : public FinalStateCostFunc<T, M, N> {
//public:
//    OCP_VARIABLES(T, M, N);
//    LateralComfortFinalCostFunc(const CostTermConfig &config)
//            : w_acc_(config.weights.at(WeightAcc)),
//              w_jerk_(config.weights.at(WeightJerk)),
//              velocity_(config.velocity.back()),
//              acceleration_(config.acceleration.back()) {
//        cross_coeff_ = 3 * w_jerk_ * pow(velocity_, 4) * acceleration_;
//    }
//
//    ~LateralComfortFinalCostFunc() = default;
//
//    std::type_index get_type_index() override {
//        return std::type_index(typeid(LateralComfortFinalCostFunc));
//    }
//
//    double value(const State &x) const override;
//
//    void gradient(const State &x, VecX &res) const override;
//
//    void hessian(const State &x, MatrixLXX &res) const override;
//
//private:
//    const double w_acc_;  // acceleration weight
//    const double w_jerk_; // jerk weight
//    double velocity_;     // longitudinal speed
//    double acceleration_; // longitudinal acceleration
//    double cross_coeff_;  // cross_coeff_ = 3 * w_jerk_ * V^4 * \dot(V);
//};

/*
  J_shape = 1/2 * w_kappa * kappa^2 + 1/2 * w_dkappa * dkappa^2
  where kappa is the curvature, dkappa is the curvature's rate of change
 */
//template<typename T, unsigned M, unsigned int N>
//class CurvatureFinalCostFunc final : public FinalStateCostFunc<T, M, N> {
//public:
//    OCP_VARIABLES(T, M, N);
//    CurvatureFinalCostFunc(const CostTermConfig &config)
//            : w_kappa_(config.weights.at(WeightKappa)),
//              w_dkappa_(config.weights.at(WeightDkappa))
//              {}
//
//    ~CurvatureFinalCostFunc() = default;
//
//
//    std::type_index get_type_index() override {
//        return std::type_index(typeid(CurvatureFinalCostFunc));
//    }
//
//    double value(const State &x) const override;
//
//    void gradient(const State &x, VecX &res) const override;
//
//    void hessian(const State &x, MatrixLXX &res) const override;
//
//private:
//    const double w_kappa_;  // curvature weight
//    const double w_dkappa_; // curvature change rate weight
//};



#endif //CILQR_COST_TERMS_H
