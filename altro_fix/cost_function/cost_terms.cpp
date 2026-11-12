//
// Created by LTsommer on 2024/5/30.
//

#include "cost_terms.h"
#include "../common/math_utils.h"
template<typename T, unsigned M, unsigned int N>
bool AccCostFunc<T, M, N>::Evaluate(const int step, const State &state, const Control &ctrl, double &val) const {
    if (step > CostFunc<T, M, N>::horizon) {
        return false;
    }
    val = 0.5 * w_acc_ * sqr<double>(ctrl[UIndex::Acc]);
    return true;
}

template<typename T, unsigned M, unsigned int N>
bool AccCostFunc<T, M, N>::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step > CostFunc<T, M, N>::horizon) {
        return false;
    }

    lu(UIndex::Acc) = w_acc_ * u(UIndex::Acc);
    return true;
}

template<typename T, unsigned M, unsigned int N>
bool AccCostFunc<T, M, N>::Hessian(const int step,
                                   const State &x,
                                   const Control &u,
                                   Eigen::Ref<MatrixLXX> lxx,
                                   Eigen::Ref<MatrixLUU> luu,
                                   Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step > CostFunc<T, M, N>::horizon) {
        return false;
    }

    luu(UIndex::Acc, UIndex::Acc) = w_acc_;
    return true;
}

template class AccCostFunc<double, 5, 1>;
template class AccCostFunc<double, 4, 2>;

template<typename T, unsigned M, unsigned int N>
bool YawRateCostFunc<T, M, N>::Evaluate(const int step, const State &state, const Control &ctrl, double &val) const {
    if (step > CostFunc<T, M, N>::horizon) {
        return false;
    }

    val = 0.5 * w_yaw_rate_ * sqr<double>(ctrl[UIndex::W]);
    return true;
}

template<typename T, unsigned M, unsigned int N>
bool YawRateCostFunc<T, M, N>::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step > CostFunc<T, M, N>::horizon) {
        return false;
    }
    return true;
}

template<typename T, unsigned M, unsigned int N>
bool YawRateCostFunc<T, M, N>::Hessian(const int step,
                                       const State &x,
                                       const Control &u,
                                       Eigen::Ref<MatrixLXX> lxx,
                                       Eigen::Ref<MatrixLUU> luu,
                                       Eigen::Ref<MatrixLXU> lxu) const {
    luu.setZero();
    lxu.setZero();
    lxx.setZero();
    if (step > CostFunc<T, M, N>::horizon) {
        return false;
    }
    luu(UIndex::W, UIndex::W) = w_yaw_rate_;
    return true;
}

template class YawRateCostFunc<double, 5, 1>;
template class YawRateCostFunc<double, 4, 2>;

template<typename T, unsigned M, unsigned int N>
bool OffsetCostFunc<T, M, N>::Evaluate(const int step, const State &state, const Control &ctrl, double &val) const {
    if (step > CostFunc<T, M, N>::horizon) {
        return false;
    }

    val = 0.5 * w_offset_ * (sqr<double>(state(0) - p_ref_[step](0)) + sqr<double>(state(1) - p_ref_[step](1)));
    return true;
}

template<typename T, unsigned M, unsigned int N>
bool OffsetCostFunc<T, M, N>::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step > CostFunc<T, M, N>::horizon) {
        return false;
    }

    lx(0) = w_offset_ * (x(0) - p_ref_[step](0));
    lx(1) = w_offset_ * (x(1) - p_ref_[step](1));
    return true;
}

template<typename T, unsigned M, unsigned int N>
bool OffsetCostFunc<T, M, N>::Hessian(const int step,
                                      const State &x,
                                      const Control &u,
                                      Eigen::Ref<MatrixLXX> lxx,
                                      Eigen::Ref<MatrixLUU> luu,
                                      Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    lxu.setZero();
    luu.setZero();
    if (step > CostFunc<T, M, N>::horizon) {
        return false;
    }

    lxx(0, 0) = w_offset_;
    lxx(1, 1) = w_offset_;
    return true;
}

template class OffsetCostFunc<double, 5, 1>;
template class OffsetCostFunc<double, 4, 2>;
template class OffsetCostFunc<double, 6, 2>;

template<typename T, unsigned M, unsigned int N>
bool VelDiffCostFunc<T, M, N>::Evaluate(const int step, const State &state, const Control &ctrl, double &val) const {
    if (step > CostFunc<T, M, N>::horizon) {
        return false;
    }

    val = 0.5 * w_vel_diff_ * sqr<double>(state(V) - velocity_[step]);
    return true;
}

template<typename T, unsigned M, unsigned int N>
bool VelDiffCostFunc<T, M, N>::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lu.setZero();
    lx.setZero();
    if (step > CostFunc<T, M, N>::horizon) {
        return false;
    }

    lx(V) = w_vel_diff_ * (x(V) - velocity_[step]);
    return true;
}

template<typename T, unsigned M, unsigned int N>
bool VelDiffCostFunc<T, M, N>::Hessian(const int step,
                                       const State &x,
                                       const Control &u,
                                       Eigen::Ref<MatrixLXX> lxx,
                                       Eigen::Ref<MatrixLUU> luu,
                                       Eigen::Ref<MatrixLXU> lxu) const {
    luu.setZero();
    lxu.setZero();
    lxx.setZero();
    if (step > CostFunc<T, M, N>::horizon) {
        return false;
    }

    lxx(V, V) = w_vel_diff_;
    return true;
}

template class VelDiffCostFunc<double, 5, 1>;
template class VelDiffCostFunc<double, 4, 2>;

template<typename T, unsigned M, unsigned int N>
bool CurvatureCostFunc<T, M, N>::Evaluate(const int step, const State &state, const Control &ctrl, double &val) const {
    if (step > CostFunc<T, M, N>::horizon) {
        return false;
    }

    if (step == CostFunc<T, M, N>::horizon) {
        val = 0.5 * (w_kappa_ * sqr(state(kKappa)) + w_dkappa_ * sqr(state(kDkappa)));
        return true;
    }
    val = 0.5 * (w_kappa_ * sqr(state(kKappa)) + w_dkappa_ * sqr(state(kDkappa)) + w_u_ * sqr(ctrl(kDDkappa)));
    return true;
}

template<typename T, unsigned M, unsigned int N>
bool CurvatureCostFunc<T, M, N>::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step > CostFunc<T, M, N>::horizon) {
        return false;
    }

    if (step == CostFunc<T, M, N>::horizon) {
        lx(kKappa) = w_kappa_ * x(kKappa);
        lx(kDkappa) = w_dkappa_ * x(kDkappa);
        return true;
    }

    lu(kDDkappa) = w_u_ * u(kDDkappa);
    lx(kKappa) = w_kappa_ * x(kKappa);
    lx(kDkappa) = w_dkappa_ * x(kDkappa);
    return true;
}

template<typename T, unsigned M, unsigned int N>
bool CurvatureCostFunc<T, M, N>::Hessian(const int step,
                                         const State &x,
                                         const Control &u,
                                         Eigen::Ref<MatrixLXX> lxx,
                                         Eigen::Ref<MatrixLUU> luu,
                                         Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step > CostFunc<T, M, N>::horizon) {
        return false;
    }

    if (step == CostFunc<T, M, N>::horizon) {
        lxx(kKappa, kKappa) = w_kappa_;
        lxx(kDkappa, kDkappa) = w_dkappa_;
        return true;
    }

    lxx(kKappa, kKappa) = w_kappa_;
    lxx(kDkappa, kDkappa) = w_dkappa_;
    luu(kDDkappa, kDDkappa) = w_u_;
    return true;
}

template class CurvatureCostFunc<double, 5, 1>;
template class CurvatureCostFunc<double, 4, 2>;

template<typename T, unsigned M, unsigned int N>
bool LateralComfortCostFunc<T, M, N>::Evaluate(const int step,
                                               const State &state,
                                               const Control &ctrl,
                                               double &val) const {
    if (step > CostFunc<T, M, N>::horizon) {
        return false;
    }

    val = 0.5 * w_acc_ * sqr(sqr(velocity_[step]) * state(kKappa)) +
          0.5 * w_jerk_ *
                  // lateral jerk = d(v^2 kappa)/dt = 2 v a kappa + v^3 dkappa/ds (the factor was 3, which
                  // also disagreed with the cross term used in Gradient / Hessian)
                  sqr(2 * velocity_[step] * acceleration_[step] * state(kKappa) +
                      cube(velocity_[step]) * state(kDkappa));
    return true;
}

template<typename T, unsigned M, unsigned int N>
bool LateralComfortCostFunc<T, M, N>::Gradient(
        const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step > CostFunc<T, M, N>::horizon) {
        return false;
    }

    lx(kKappa) = (w_acc_ * pow(velocity_[step], 4) + 4 * w_jerk_ * sqr(velocity_[step]) * sqr(acceleration_[step])) *
                         x(kKappa) +
                 cross_coeff_[step] * x(kDkappa);
    lx(kDkappa) = cross_coeff_[step] * x(kKappa) + w_jerk_ * pow(velocity_[step], 6) * x(kDkappa);
    return true;
}

template<typename T, unsigned M, unsigned int N>
bool LateralComfortCostFunc<T, M, N>::Hessian(const int step,
                                              const State &x,
                                              const Control &u,
                                              Eigen::Ref<MatrixLXX> lxx,
                                              Eigen::Ref<MatrixLUU> luu,
                                              Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step > CostFunc<T, M, N>::horizon) {
        return false;
    }

    lxx(kKappa, kKappa) =
            w_acc_ * pow(velocity_[step], 4) + 4 * w_jerk_ * sqr(velocity_[step]) * sqr(acceleration_[step]);
    lxx(kKappa, kDkappa) = cross_coeff_[step];
    lxx(kDkappa, kKappa) = cross_coeff_[step];
    lxx(kDkappa, kDkappa) = w_jerk_ * pow(velocity_[step], 6);
    return true;
}

template class LateralComfortCostFunc<double, 5, 1>;
template class LateralComfortCostFunc<double, 4, 2>;

//template class CurvatureFinalCostFunc<double, 4, 2>;
