#include "smooth_kappa_model.h"

SmoothKappaModel::State SmoothKappaModel::PrepareInitialState(
        const State &state) const {
    State prepared = state;
    prepared(kTheta) = vehicle_heading::PrepareInitialHeading(state(kTheta));
    return prepared;
}

SmoothKappaModel::State SmoothKappaModel::StateDifference(
        const State &state, const State &reference) const {
    State delta = state - reference;
    delta(kTheta) = AngleDifference(state(kTheta), reference(kTheta));
    return delta;
}
