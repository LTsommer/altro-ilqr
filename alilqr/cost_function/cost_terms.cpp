//
// Created by LTsommer on 2024/5/30.
//

#include "cost_terms.h"
template<typename T, unsigned M, unsigned int N>
bool AccCostFunc<T, M, N>::Evaluate(const int step,
                                    const State &state,
                                    const Control &ctrl,
                                    double &val) const {
    if (step > CostFunc<T, M, N>::horizon)
        return false;
    val = 0.5 * w_acc_ * sqr<double>(ctrl[UIndex::Acc]);
    return true;
}

template<typename T, unsigned M, unsigned int N>
bool AccCostFunc<T, M, N>::Gradient(const int step, const State &x, const Control &u,
                                    Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step > CostFunc<T, M, N>::horizon)
        return false;

    lu(UIndex::Acc) = w_acc_ * u(UIndex::Acc);
    return true;
}

template<typename T, unsigned M, unsigned int N>
bool AccCostFunc<T, M, N>::Hessian(const int step, const State &x, const Control &u,
                                   Eigen::Ref<MatrixLXX> lxx,
                                   Eigen::Ref<MatrixLUU> luu,
                                   Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step > CostFunc<T, M, N>::horizon)
        return false;

    luu(UIndex::Acc, UIndex::Acc) = w_acc_;
    return true;
}

template class AccCostFunc<double, 5, 1>;
template class AccCostFunc<double, 4, 2>;

template<typename T, unsigned M, unsigned int N>
bool YawRateCostFunc<T, M, N>::Evaluate(const int step,
                                        const State &state,
                                        const Control &ctrl,
                                        double &val) const {
    if (step > CostFunc<T, M, N>::horizon)
        return false;

    val = 0.5 * w_yaw_rate_ * sqr<double>(ctrl[UIndex::W]);
    return true;
}

template<typename T, unsigned M, unsigned int N>
bool YawRateCostFunc<T, M, N>::Gradient(const int step, const State &x, const Control &u,
                                        Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step > CostFunc<T, M, N>::horizon)
        return false;
    return true;
}

template<typename T, unsigned M, unsigned int N>
bool YawRateCostFunc<T, M, N>::Hessian(const int step, const State &x, const Control &u,
                                       Eigen::Ref<MatrixLXX> lxx,
                                       Eigen::Ref<MatrixLUU> luu,
                                       Eigen::Ref<MatrixLXU> lxu) const {
    luu.setZero();
    lxu.setZero();
    lxx.setZero();
    if (step > CostFunc<T, M, N>::horizon)
        return false;
    luu(UIndex::W, UIndex::W) = w_yaw_rate_;
    return true;
}

template class YawRateCostFunc<double, 5, 1>;
template class YawRateCostFunc<double, 4, 2>;

template<typename T, unsigned M, unsigned int N>
bool OffsetCostFunc<T, M, N>::Evaluate(const int step, const State &state, const Control &ctrl, double &val) const {
    if (step > CostFunc<T, M, N>::horizon)
        return false;

    val = 0.5 * w_offset_ * (sqr<double>(state(X) - p_ref_[step](X)) +
                             sqr<double>(state(Y) - p_ref_[step](Y)));
//    std::cout << "step: " << step << "\n"
//              << "x: " << state(X) << ", " << state(Y) << "\n"
//              << "ref: " << p_ref_[step](X) << ", " << p_ref_[step](Y) << "\n";
    return true;
}

template<typename T, unsigned M, unsigned int N>
bool OffsetCostFunc<T, M, N>::Gradient(const int step, const State &x, const Control &u,
                                       Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step > CostFunc<T, M, N>::horizon)
        return false;

    lx(X) = w_offset_ * (x(X) - p_ref_[step](X));
    lx(Y) = w_offset_ * (x(Y) - p_ref_[step](Y));
    return true;
}

template<typename T, unsigned M, unsigned int N>
bool OffsetCostFunc<T, M, N>::Hessian(const int step, const State &x, const Control &u,
                                      Eigen::Ref<MatrixLXX> lxx,
                                      Eigen::Ref<MatrixLUU> luu,
                                      Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    lxu.setZero();
    luu.setZero();
    if (step > CostFunc<T, M, N>::horizon)
        return false;

    lxx(X, X) = w_offset_;
    lxx(Y, Y) = w_offset_;
    return true;
}

template class OffsetCostFunc<double, 5, 1>;
template class OffsetCostFunc<double, 4, 2>;

template<typename T, unsigned M, unsigned int N>
bool VelDiffCostFunc<T, M, N>::Evaluate(const int step, const State &state, const Control &ctrl, double &val) const {
    if (step > CostFunc<T, M, N>::horizon)
        return false;

    val = 0.5 * w_vel_diff_ * sqr<double>(state(V) - velocity_[step]);
    return true;
}

template<typename T, unsigned M, unsigned int N>
bool VelDiffCostFunc<T, M, N>::Gradient(const int step, const State &x, const Control &u,
                                        Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lu.setZero();
    lx.setZero();
    if (step > CostFunc<T, M, N>::horizon)
        return false;

    lx(V) = w_vel_diff_ * (x(V) - velocity_[step]);
    return true;
}

template<typename T, unsigned M, unsigned int N>
bool VelDiffCostFunc<T, M, N>::Hessian(const int step, const State &x, const Control &u,
                                       Eigen::Ref<MatrixLXX> lxx,
                                       Eigen::Ref<MatrixLUU> luu,
                                       Eigen::Ref<MatrixLXU> lxu) const {
    luu.setZero();
    lxu.setZero();
    lxx.setZero();
    if (step > CostFunc<T, M, N>::horizon)
        return false;

    lxx(V, V) = w_vel_diff_;
    return true;
}

template class VelDiffCostFunc<double, 5, 1>;
template class VelDiffCostFunc<double, 4, 2>;

template<typename T, unsigned M, unsigned int N>
bool CurvatureCostFunc<T, M, N>::Evaluate(const int step, const State &state, const Control &ctrl, double &val) const {
    if (step > CostFunc<T, M, N>::horizon)
        return false;

    val = 0.5 * (w_kappa_ * sqr(state(kKappa)) + w_dkappa_ * sqr(state(kDkappa)) +
                      w_u_ * sqr(ctrl(kDDkappa)));
    return true;
}

template<typename T, unsigned M, unsigned int N>
bool CurvatureCostFunc<T, M, N>::Gradient(const int step, const State &x, const Control &u,
                                          Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step > CostFunc<T, M, N>::horizon)
        return false;

    lu(kDDkappa) = w_u_ * u(kDDkappa);
    lx(kKappa) = w_kappa_ * x(kKappa);
    lx(kDkappa) = w_dkappa_ * x(kDkappa);
    return true;
}

template<typename T, unsigned M, unsigned int N>
bool CurvatureCostFunc<T, M, N>::Hessian(const int step, const State &x, const Control &u,
                                         Eigen::Ref<MatrixLXX> lxx,
                                         Eigen::Ref<MatrixLUU> luu,
                                         Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step > CostFunc<T, M, N>::horizon)
        return false;

    lxx(kKappa, kKappa) = w_kappa_;
    lxx(kDkappa, kDkappa) = w_dkappa_;
    luu(kDDkappa, kDDkappa) = w_u_;
    return true;
}

template class CurvatureCostFunc<double, 5, 1>;
template class CurvatureCostFunc<double, 4, 2>;

template<typename T, unsigned M, unsigned int N>
bool LateralComfortCostFunc<T, M, N>::Evaluate(const int step, const State &state, const Control &ctrl, double &val) const {
    if (step > CostFunc<T, M, N>::horizon)
        return false;

    val = 0.5 * w_acc_ * sqr(sqr(velocity_[step]) * state(kKappa)) +
          0.5 * w_jerk_ *
          sqr(3 * velocity_[step] * acceleration_[step] * state(kKappa) +
              cube(velocity_[step]) * state(kDkappa));
    return true;
}

template<typename T, unsigned M, unsigned int N>
bool LateralComfortCostFunc<T, M, N>::Gradient(const int step, const State &x, const Control &u,
                                               Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    if (step > CostFunc<T, M, N>::horizon)
        return false;

    lx(kKappa) =
            (w_acc_ * pow(velocity_[step], 4) +
             9 * w_jerk_ * sqr(velocity_[step]) * sqr(acceleration_[step])) *
            x(kKappa) +
            cross_coeff_[step] * x(kDkappa);
    lx(kDkappa) = cross_coeff_[step] * x(kKappa) +
                   w_jerk_ * pow(velocity_[step], 6) * x(kDkappa);
    return true;
}

template<typename T, unsigned M, unsigned int N>
bool LateralComfortCostFunc<T, M, N>::Hessian(const int step, const State &x, const Control &u,
                                              Eigen::Ref<MatrixLXX> lxx,
                                              Eigen::Ref<MatrixLUU> luu,
                                              Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    if (step > CostFunc<T, M, N>::horizon)
        return false;


    lxx(kKappa, kKappa) =
            w_acc_ * pow(velocity_[step], 4) +
            9 * w_jerk_ * sqr(velocity_[step]) * sqr(acceleration_[step]);
    lxx(kKappa, kDkappa) = cross_coeff_[step];
    lxx(kDkappa, kKappa) = cross_coeff_[step];
    lxx(kDkappa, kDkappa) = w_jerk_ * pow(velocity_[step], 6);
    return true;
}

template class LateralComfortCostFunc<double, 5, 1>;
template class LateralComfortCostFunc<double, 4, 2>;

//template<typename T, unsigned M, unsigned int N>
//double FinalAccCostFunc<T, M, N>::value(const State &state) const {
//    return 0.0;
//}
//
//template<typename T, unsigned M, unsigned int N>
//void FinalAccCostFunc<T, M, N>::gradient(const State &state, VecX &g) const {
//    g.setZero();
//    return;
//}
//
//template<typename T, unsigned M, unsigned int N>
//void FinalAccCostFunc<T, M, N>::hessian(const State &state, MatrixLXX &h) const {
//    h.setZero();
//    return;
//}
//
//template class FinalAccCostFunc<double, 5, 1>;
//template class FinalAccCostFunc<double, 4, 2>;
//
//template<typename T, unsigned M, unsigned int N>
//double FinalYawRateCostFunc<T, M, N>::value(const State &state) const {
//    return 0.0;
//}
//
//template<typename T, unsigned M, unsigned int N>
//void FinalYawRateCostFunc<T, M, N>::gradient(const State &state, VecX &g) const {
//    g.setZero();
//    return;
//}
//
//template<typename T, unsigned M, unsigned int N>
//void FinalYawRateCostFunc<T, M, N>::hessian(const State &state, MatrixLXX &h) const {
//    h.setZero();
//    return;
//}
//
//template class FinalYawRateCostFunc<double, 5, 1>;
//template class FinalYawRateCostFunc<double, 4, 2>;
//
//template<typename T, unsigned M, unsigned int N>
//double FinalOffsetCostFunc<T, M, N>::value(const State &state) const {
//    return 0.5 * w_offset_ * (sqr<double>(state(X) - p_ref_(X)) +
//            sqr<double>(state(Y) - p_ref_(Y)));
//}
//
//template<typename T, unsigned M, unsigned int N>
//void FinalOffsetCostFunc<T, M, N>::gradient(const State &state, VecX &g) const {
//    g.setZero();
//    g(X) = w_offset_ * (state(X) - p_ref_(X));
//    return;
//}
//
//template<typename T, unsigned M, unsigned int N>
//void FinalOffsetCostFunc<T, M, N>::hessian(const State &state, MatrixLXX &h) const {
//    h.setZero();
//    h(X, X) = w_offset_;
//    h(Y, Y) = w_offset_;
//    return;
//}
//
//template class FinalOffsetCostFunc<double, 5, 1>;
//template class FinalOffsetCostFunc<double, 4, 2>;
//
//
//template<typename T, unsigned M, unsigned int N>
//double FinalVelDiffCostFunc<T, M, N>::value(const State &state) const {
//    return 0.5 * w_vel_diff_ * sqr<double>(state(V) - vel);
//}
//
//template<typename T, unsigned M, unsigned int N>
//void FinalVelDiffCostFunc<T, M, N>::gradient(const State &state, VecX &g) const {
//    g.setZero();
//    g(V) = w_vel_diff_ * (state(V) - vel);
//    return;
//}
//
//template<typename T, unsigned M, unsigned int N>
//void FinalVelDiffCostFunc<T, M, N>::hessian(const State &state, MatrixLXX &h) const {
//    h.setZero();
//    h(V, V) = w_vel_diff_;
//    return;
//}
//
//template class FinalVelDiffCostFunc<double, 5, 1>;
//template class FinalVelDiffCostFunc<double, 4, 2>;
//
///*
//  J_comfort = 1/2 * w_acc * (V^2 * kappa)^2
//            + 1/2 * w_jerk * (3 * V * A * kappa + V^3 * dkappa)^2
//  where V is the velocity, A is the acceleration, kappa is the curvature, and
//  dkappa is the curvature's rate of change
// */
//template<typename T, unsigned M, unsigned int N>
//double LateralComfortFinalCostFunc<T, M, N>::value(const State &x) const {
//    // quadratic costs
//    return 0.5 * w_acc_ * sqr(sqr(velocity_) * x(kKappa)) +
//           0.5 * w_jerk_ *
//           sqr<double>(3 * velocity_ * acceleration_ * x(kKappa) +
//               cube<double>(velocity_) * x(kDKappa));
//}
//
//template<typename T, unsigned M, unsigned int N>
//void LateralComfortFinalCostFunc<T, M, N>::gradient(const State &x, VecX &res) const {
//    res.setZero();
//    res(kKappa) = (w_acc_ * pow(velocity_, 4) +
//                   9 * w_jerk_ * sqr(velocity_) * sqr(acceleration_)) *
//                  x(kKappa) +
//                  cross_coeff_ * x(kDKappa);
//    res(kDKappa) =
//            cross_coeff_ * x(kKappa) + w_jerk_ * pow(velocity_, 6) * x(kDKappa);
//}
//
//template<typename T, unsigned M, unsigned int N>
//void LateralComfortFinalCostFunc<T, M, N>::hessian(const State &x, MatrixLXX &res) const {
//    res.setZero();
//    res(kKappa, kKappa) = w_acc_ * pow(velocity_, 4) +
//                          9 * w_jerk_ * sqr(velocity_) * sqr(acceleration_);
//    res(kKappa, kDKappa) = cross_coeff_;
//    res(kDKappa, kKappa) = cross_coeff_;
//    res(kDKappa, kDKappa) = w_jerk_ * pow(velocity_, 6);
//}
//
//template class LateralComfortFinalCostFunc<double, 5, 1>;
//template class LateralComfortFinalCostFunc<double, 4, 2>;
//
///*
//  J_shape = 1/2 * w_kappa * kappa^2 + 1/2 * w_dkappa * w^2
//  where kappa is the curvature, dkappa is the curvature's rate of change
// */
//template<typename T, unsigned M, unsigned int N>
//double CurvatureFinalCostFunc<T, M, N>::value(const State &x) const {
//    // quadratic costs
//    return 0.5 * (w_kappa_ * sqr(x(kKappa)) + w_dkappa_ * sqr(x(kDKappa)));
//}
//
//template<typename T, unsigned M, unsigned int N>
//void CurvatureFinalCostFunc<T, M, N>::gradient(const State &x, VecX &res) const {
//    res.setZero();
//    res(kKappa) = w_kappa_ * x(kKappa);
//    res(kDKappa) = w_dkappa_ * x(kDKappa);
//}
//
//template<typename T, unsigned M, unsigned int N>
//void CurvatureFinalCostFunc<T, M, N>::hessian(const State &x, MatrixLXX &res) const {
//    res.setZero();
//    res(kKappa, kKappa) = w_kappa_;
//    res(kDKappa, kDKappa) = w_dkappa_;
//}
//
//template class CurvatureFinalCostFunc<double, 5, 1>;
//template class CurvatureFinalCostFunc<double, 4, 2>;