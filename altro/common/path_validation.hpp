#pragma once

#include "math/vec2d.h"
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

class FrenetCoordinateSystem;

namespace altro {

struct RoadBoundarySample {
    double s;
    double left_offset;
    double right_offset;
};

struct PathSceneGeometry {
    std::vector<Vec2d> left_boundary;
    std::vector<Vec2d> right_boundary;
    std::vector<std::vector<Vec2d>> obstacles;
    double vehicle_radius = 0.0;
};

PathSceneGeometry BuildPathSceneGeometry(
        const FrenetCoordinateSystem &fcs,
        double lane_half_width,
        double vehicle_radius,
        const std::vector<RoadBoundarySample> &samples);

struct PathValidationFailure {
    std::optional<std::size_t> segment;
    std::string reason;
};

struct PathClearance {
    std::string object;
    std::optional<std::size_t> segment;
    double center_distance = 0.0;
    Vec2d path_point;
    Vec2d object_point;
};

// Checks the sweep of a circle along the XY path. Boundary contact is a failure.
// When requested, also records the closest approach to each scene object.
std::optional<PathValidationFailure> ValidatePath(
        const PathSceneGeometry &scene,
        const std::vector<Vec2d> &path,
        std::vector<PathClearance> *clearances = nullptr);

bool ReportPathValidation(const std::optional<PathValidationFailure> &failure,
                          bool print_summary);

template<typename States>
bool ValidateSolvedPath(const PathSceneGeometry &scene,
                        const States &states,
                        bool print_summary) {
    std::vector<Vec2d> path;
    path.reserve(states.size());
    for (const auto &state : states) {
        path.emplace_back(state[0], state[1]);
    }
    return ReportPathValidation(ValidatePath(scene, path), print_summary);
}

} // namespace altro
