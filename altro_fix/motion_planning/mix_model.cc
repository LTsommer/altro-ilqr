#include "mix_model.h"

MixModel::State MixModel::PrepareInitialState(const State &x) const {
    State prepared = x;
    prepared(mTheta) = vehicle_heading::PrepareInitialHeading(x(mTheta));
    return prepared;
}

MixModel::State MixModel::StateDifference(const State &x,
                                          const State &reference) const {
    State delta = x - reference;
    delta(mTheta) = AngleDifference(x(mTheta), reference(mTheta));
    return delta;
}

void MixModel::set_traj_info(const vector<double> &t,
                             const vector<double> &s,
                             const vector<double> &v,
                             const vector<double> &y) {
    sy.set_points(s, y);
    ts.set_points(t, s);
    tv.set_points(t, v);
}

void MixModel::UpdateTimer(const double dt) {
    timer_ += dt;
}

double MixModel::HeadingChange(const State &x,
                               const Control &u,
                               const double dt) {
    return x(mKappa) * x(mVel) * dt +
           0.5 * (x(mKappa) * x(mAcc) + x(mVel) * u(mDotKappa)) *
                   sqr<double>(dt) +
           (1.0 / 3.0) * (0.5 * x(mKappa) * u(mJerk) + x(mAcc) * u(mDotKappa)) *
                   cube<double>(dt) +
           0.125 * u(mDotKappa) * u(mJerk) * cube<double>(dt) * dt;
}

double MixModel::NextVel(const State &x, const Control &u, const double dt) {
    return x(mVel) + x(mAcc) * dt + 0.5 * u(mJerk) * sqr<double>(dt);
}

PlanarPositionPartials MixModel::PositionPartials(const State &x,
                                                  const Control &u,
                                                  const double dt) {
    return IntegratePlanarPositionPartials(
            x(mTheta), HeadingChange(x, u, dt), x(mVel), NextVel(x, u, dt), dt);
}
