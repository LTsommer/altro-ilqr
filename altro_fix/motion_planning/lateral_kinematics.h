//
// Lateral motion quantities of the MixModel state [x, y, theta, kappa, v, a] and control
// [jerk, kappa_dot], shared by the costs and the constraints that act on them.
//

#ifndef CILQR_MOTION_PLANNING_LATERAL_KINEMATICS_H
#define CILQR_MOTION_PLANNING_LATERAL_KINEMATICS_H

#include "mix_model.h"
#include "../common/math_utils.h"

// Lateral (centripetal) acceleration v^2 * kappa.
template<typename State>
inline double LateralAcc(const State &x) {
    return sqr(x(mVel)) * x(mKappa);
}

// Lateral jerk, the time derivative of v^2 * kappa: 2 * v * a * kappa + v^2 * kappa_dot.
template<typename State, typename Control>
inline double LateralJerk(const State &x, const Control &u) {
    return doppel(x(mVel) * x(mAcc) * x(mKappa)) + sqr(x(mVel)) * u(mDotKappa);
}

#endif // CILQR_MOTION_PLANNING_LATERAL_KINEMATICS_H
