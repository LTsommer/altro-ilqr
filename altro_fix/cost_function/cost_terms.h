//
// Created by LTsommer on 2024/5/30.
//

#ifndef CILQR_COST_TERMS_H
#define CILQR_COST_TERMS_H

#include "cost_function.hpp"
#include "../frenet_coordinate/vec2d.h"
#include "../frenet_coordinate/frenet_coordinate_system.h"
#include "cost_term_config.hpp"
#include "../constraint_definition/constraint_formulation.h"
#include <memory>
#include <stdexcept>

/*
  J_acc = 1/2 * w_acc * acc^2
  J_acc = 1/2 * w_acc * u.T * [1 0;0 0] * u
 */
template<typename T, unsigned M, unsigned int N>
class AccCostFunc final : public CostFunc<T, M, N> {
public:
    OCP_VARIABLES(T, M, N)
    AccCostFunc(const CostTermConfig &config) : w_acc_(config.weights.at(WeightIndex::WeightAcc)) {
        CostFunc<T, M, N>::SetHorizon(config.steps.size());
    }

    ~AccCostFunc() = default;

    virtual std::type_index GetTypeIndex() const override { return std::type_index(typeid(AccCostFunc)); }

    virtual std::string GetName() const override { return std::string("AccCostFunc"); }

    virtual bool Evaluate(const int step, const State &state, const Control &ctrl, double &val) const override;

    virtual bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    virtual bool Hessian(const int step,
                         const State &x,
                         const Control &u,
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
class YawRateCostFunc final : public CostFunc<T, M, N> {
public:
    OCP_VARIABLES(T, M, N)
    YawRateCostFunc(const CostTermConfig &config) : w_yaw_rate_(config.weights.at(WeightIndex::WeightYawRate)) {
        CostFunc<T, M, N>::SetHorizon(config.steps.size());
    };

    ~YawRateCostFunc() = default;

    virtual std::type_index GetTypeIndex() const override { return std::type_index(typeid(YawRateCostFunc)); }

    virtual std::string GetName() const override { return std::string("YawRatwCostFunc"); }

    virtual bool Evaluate(const int step, const State &state, const Control &ctrl, double &val) const override;

    virtual bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    virtual bool Hessian(const int step,
                         const State &x,
                         const Control &u,
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
    OffsetCostFunc(const CostTermConfig &config)
        : w_offset_(config.weights.at(WeightIndex::WeightRefOffset)), p_ref_(config.p_ref) {
        CostFunc<T, M, N>::SetHorizon(config.steps.size());
    };

    ~OffsetCostFunc() = default;

    void UpdateConfig(const CostTermConfig &config) override { p_ref_ = config.p_ref; }

    virtual std::type_index GetTypeIndex() const override { return std::type_index(typeid(OffsetCostFunc)); }

    virtual std::string GetName() const override { return std::string("OffsetCostFunc"); }

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
    double w_offset_;
    std::vector<Vector2d> p_ref_;
};

/*
  J = 1/2 * w_vel_diff * (v - v_ref)^2
 */
template<typename T, unsigned M, unsigned int N>
class VelDiffCostFunc final : public CostFunc<T, M, N> {
public:
    OCP_VARIABLES(T, M, N);
    VelDiffCostFunc(const CostTermConfig &config)
        : w_vel_diff_(config.weights.at(WeightVelDiff)), velocity_(config.velocity) {
        CostFunc<T, M, N>::SetHorizon(config.steps.size());
    };

    ~VelDiffCostFunc() = default;

    virtual std::type_index GetTypeIndex() const override { return std::type_index(typeid(VelDiffCostFunc)); }

    virtual std::string GetName() const override { return std::string("VelDiffCostFunc"); }

    virtual bool Evaluate(const int step, const State &state, const Control &ctrl, double &val) const override;

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

/* J = 1 / 2 * w_k * kappa ^ 2 + 1 / 2 * w_dk * dkappa^ 2 + 1/ 2 w_u ddkappa ^ 2
 * */
template<typename T, unsigned M, unsigned int N>
class CurvatureCostFunc final : public CostFunc<T, M, N> {
public:
    OCP_VARIABLES(T, M, N);
    CurvatureCostFunc(const CostTermConfig &config)
        : w_dkappa_(config.weights.at(WeightDkappa)),
          w_kappa_(config.weights.at(WeightKappa)),
          w_u_(config.weights.at(WeightU)) {
        CostFunc<T, M, N>::SetHorizon(config.steps.size());
    };

    ~CurvatureCostFunc() = default;

    virtual std::type_index GetTypeIndex() const override { return std::type_index(typeid(CurvatureCostFunc)); }

    virtual std::string GetName() const override { return std::string("CurvatureCostFunc"); }

    virtual bool Evaluate(const int step, const State &state, const Control &ctrl, double &val) const override;

    virtual bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    virtual bool Hessian(const int step,
                         const State &x,
                         const Control &u,
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
            + 1/2 * w_jerk * (2 * V * A * kappa + V^3 * dkappa)^2
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
        // Both references are read at every node, the terminal one included.
        if (acceleration_.size() < velocity_.size()) {
            throw std::invalid_argument("LateralComfortCostFunc: config.acceleration needs one entry per node");
        }
        cross_coeff_.resize(velocity_.size());
        for (int i = 0; i < static_cast<int>(velocity_.size()); i++) {
            cross_coeff_[i] = 2 * w_jerk_ * pow(velocity_[i], 4) * acceleration_[i];
        }
        CostFunc<T, M, N>::SetHorizon(config.steps.size());
    }

    ~LateralComfortCostFunc() = default;

    virtual std::string GetName() const override { return std::string("LateralComfortCostFunc"); }

    virtual std::type_index GetTypeIndex() const override { return std::type_index(typeid(LateralComfortCostFunc)); }

    virtual bool Evaluate(const int step, const State &state, const Control &ctrl, double &val) const override;

    virtual bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    virtual bool Hessian(const int step,
                         const State &x,
                         const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const override;

private:
    const double w_acc_;               // acceleration weight
    const double w_jerk_;              // jerk weight
    std::vector<double> velocity_;     // longitudinal speed
    std::vector<double> acceleration_; // longitudinal acceleration
    std::vector<double> cross_coeff_;  // cross_coeff_ = 3 * w_jerk_ * V^4 * \dot(V);
};

#endif //CILQR_COST_TERMS_H
