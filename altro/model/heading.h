#ifndef CILQR_MODEL_HEADING_H
#define CILQR_MODEL_HEADING_H

#include "common/math_utils.h"
#include <stdexcept>

namespace vehicle_heading {

// At this magnitude, reducing by the double-precision 2 pi incurs about 4e-14 rad of phase error.
// Applies to the external initial heading and the actual turn of one model step.
inline constexpr double kMaxAngleMagnitude = 1024.0; // rad

inline bool IsSupportedAngle(double angle) {
    return std::abs(angle) <= kMaxAngleMagnitude;
}

inline double PrepareInitialHeading(double angle) {
    if (!IsSupportedAngle(angle)) {
        throw std::invalid_argument("Initial vehicle heading must be finite and within +/-1024 rad");
    }
    return NormalizeAngle(angle);
}

} // namespace vehicle_heading

#endif // CILQR_MODEL_HEADING_H
