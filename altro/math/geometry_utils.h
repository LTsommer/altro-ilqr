#pragma once

#include "math/vec2d.h"
#include <cmath>

namespace altro {

inline constexpr double kGeometryPi = 3.14159265358979323846;

inline double CrossProd(const Vec2d &origin,
                        const Vec2d &first,
                        const Vec2d &second) {
    return (first - origin).CrossProd(second - origin);
}

// Wrap an angle to [0, 2 pi).
inline double WrapAngle(double angle) {
    constexpr double kTwoPi = 2.0 * kGeometryPi;
    const double wrapped = std::fmod(angle, kTwoPi);
    return wrapped < 0.0 ? wrapped + kTwoPi : wrapped;
}

} // namespace altro
