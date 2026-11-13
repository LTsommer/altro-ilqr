#include "common/path_validation.hpp"

#include "frenet_coordinate/frenet_coordinate_system.h"
#include "math/line_segment2d.h"
#include "math/polygon2d.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace altro {
namespace {

using Points = std::vector<Vec2d>;

Vec2d RoadPoint(const FrenetCoordinateSystem &fcs, double s, double offset) {
    Vec2d point;
    if (fcs.FrenetCoord2CartCoord(Vec2d(s, offset), point) !=
        TRANSFORM_SUCCESS) {
        throw std::invalid_argument("Cannot construct scene road geometry");
    }
    return point;
}

void AppendRoadsideObstacles(const FrenetCoordinateSystem &fcs,
                             const std::vector<RoadBoundarySample> &samples,
                             const Points &boundary,
                             double lane_half_width,
                             bool left,
                             std::vector<Points> &obstacles) {
    const double nominal = left ? lane_half_width : -lane_half_width;
    const auto offset = [left](const RoadBoundarySample &sample) {
        return left ? sample.left_offset : sample.right_offset;
    };

    // Use the same inward edge and taper as the connected samples in the road CSV.
    for (std::size_t i = 0; i < samples.size();) {
        if (offset(samples[i]) == nominal) {
            ++i;
            continue;
        }
        const std::size_t first = i;
        Points polygon;
        if (first > 0) {
            polygon.push_back(boundary[first - 1]);
        }
        while (i < samples.size() && offset(samples[i]) != nominal) {
            polygon.push_back(RoadPoint(fcs, samples[i].s, offset(samples[i])));
            ++i;
        }
        if (i < samples.size()) {
            polygon.push_back(boundary[i]);
        }
        for (std::size_t j = i; j > first; --j) {
            polygon.push_back(boundary[j - 1]);
        }
        obstacles.push_back(std::move(polygon));
    }
}

void RecordClearance(PathClearance &closest,
                     std::size_t segment,
                     double distance,
                     const Vec2d &path_point,
                     const Vec2d &object_point) {
    if (std::isfinite(distance) &&
        (!closest.segment || distance < closest.center_distance)) {
        closest.segment = segment;
        closest.center_distance = distance;
        closest.path_point = path_point;
        closest.object_point = object_point;
    }
}

template<typename Geometry>
bool SweepTouches(const LineSegment2d &segment,
                  const Geometry &geometry,
                  double radius,
                  std::size_t index,
                  PathClearance *closest) {
    if (!closest) {
        return geometry.DistanceTo(segment) <= radius;
    }
    Vec2d path_point, object_point;
    const double distance =
            geometry.DistanceTo(segment, &object_point, &path_point);
    RecordClearance(*closest, index, distance, path_point, object_point);
    return distance <= radius;
}

bool SweepTouchesEdges(const LineSegment2d &segment,
                       const Points &edge,
                       double radius,
                       std::size_t index,
                       PathClearance *closest) {
    bool touches = false;
    for (std::size_t i = 1; i < edge.size(); ++i) {
        if (SweepTouches(segment,
                         LineSegment2d(edge[i - 1], edge[i]),
                         radius,
                         index,
                         closest)) {
            touches = true;
            if (!closest) {
                break;
            }
        }
    }
    return touches;
}

bool FinitePoint(const Vec2d &point) {
    return std::isfinite(point.x()) && std::isfinite(point.y());
}

bool FinitePoints(const Points &points) {
    return std::all_of(points.begin(), points.end(), FinitePoint);
}

} // namespace

PathSceneGeometry BuildPathSceneGeometry(
        const FrenetCoordinateSystem &fcs,
        double lane_half_width,
        double vehicle_radius,
        const std::vector<RoadBoundarySample> &samples) {
    PathSceneGeometry scene;
    scene.vehicle_radius = vehicle_radius;
    if (samples.empty()) {
        return scene;
    }

    // Extend the nominal road beyond the plotted horizon using the full reference curve.
    const double length = fcs.GetSlength();
    constexpr double kExtensionStep = 1.0;
    std::vector<RoadBoundarySample> full_samples;
    for (double s = 0.0; s < samples.front().s; s += kExtensionStep) {
        full_samples.push_back({s, lane_half_width, -lane_half_width});
    }
    full_samples.insert(full_samples.end(), samples.begin(), samples.end());
    for (double s = samples.back().s + kExtensionStep; s < length;
         s += kExtensionStep) {
        full_samples.push_back({s, lane_half_width, -lane_half_width});
    }
    if (samples.back().s < length) {
        full_samples.push_back({length, lane_half_width, -lane_half_width});
    }

    scene.left_boundary.reserve(full_samples.size());
    scene.right_boundary.reserve(full_samples.size());
    for (const auto &sample : full_samples) {
        scene.left_boundary.push_back(
                RoadPoint(fcs, sample.s, lane_half_width));
        scene.right_boundary.push_back(
                RoadPoint(fcs, sample.s, -lane_half_width));
    }
    AppendRoadsideObstacles(fcs,
                            full_samples,
                            scene.left_boundary,
                            lane_half_width,
                            true,
                            scene.obstacles);
    AppendRoadsideObstacles(fcs,
                            full_samples,
                            scene.right_boundary,
                            lane_half_width,
                            false,
                            scene.obstacles);
    return scene;
}

std::optional<PathValidationFailure> ValidatePath(
        const PathSceneGeometry &scene,
        const Points &path,
        std::vector<PathClearance> *clearances) {
    if (clearances) {
        clearances->clear();
    }
    if (path.size() < 2) {
        return PathValidationFailure{std::nullopt,
                                     "path needs at least two points"};
    }
    if (scene.left_boundary.size() < 2 || scene.right_boundary.size() < 2 ||
        !FinitePoints(scene.left_boundary) ||
        !FinitePoints(scene.right_boundary) ||
        !std::isfinite(scene.vehicle_radius) || scene.vehicle_radius <= 0.0) {
        return PathValidationFailure{std::nullopt,
                                     "invalid road geometry or vehicle radius"};
    }
    std::vector<Polygon2d> obstacles;
    obstacles.reserve(scene.obstacles.size());
    for (const auto &vertices : scene.obstacles) {
        if (vertices.size() < 3 || !FinitePoints(vertices)) {
            return PathValidationFailure{std::nullopt,
                                         "invalid obstacle geometry"};
        }
        obstacles.emplace_back(vertices);
    }

    Points corridor_vertices = scene.left_boundary;
    corridor_vertices.insert(corridor_vertices.end(),
                             scene.right_boundary.rbegin(),
                             scene.right_boundary.rend());
    const Polygon2d corridor(corridor_vertices);
    const LineSegment2d entry(scene.left_boundary.front(),
                              scene.right_boundary.front());
    const LineSegment2d exit(scene.left_boundary.back(),
                             scene.right_boundary.back());

    if (clearances) {
        for (const char *name : {"left road boundary",
                                 "right road boundary",
                                 "road start",
                                 "road end"}) {
            clearances->emplace_back();
            clearances->back().object = name;
        }
        for (std::size_t j = 0; j < obstacles.size(); ++j) {
            clearances->emplace_back();
            clearances->back().object = "obstacle " + std::to_string(j + 1);
        }
    }
    const auto closest = [clearances](std::size_t object) -> PathClearance * {
        return clearances ? &(*clearances)[object] : nullptr;
    };
    std::optional<PathValidationFailure> failure;
    constexpr double kContactTolerance = 1e-9; // meters
    const double radius = scene.vehicle_radius + kContactTolerance;
    for (std::size_t i = 0; i + 1 < path.size(); ++i) {
        const auto &start = path[i];
        const auto &end = path[i + 1];
        if (!FinitePoint(start) || !FinitePoint(end)) {
            if (clearances) {
                clearances->clear();
            }
            return failure ? failure
                           : std::optional<PathValidationFailure>{
                                     PathValidationFailure{
                                             i, "non-finite path point"}};
        }
        const auto record_failure = [&](const std::string &reason) {
            if (!failure) {
                failure = PathValidationFailure{i, reason};
            }
        };
        const LineSegment2d segment(start, end);
        if (SweepTouchesEdges(
                    segment, scene.left_boundary, radius, i, closest(0))) {
            record_failure("left road boundary");
        }
        if (SweepTouchesEdges(
                    segment, scene.right_boundary, radius, i, closest(1))) {
            record_failure("right road boundary");
        }
        const bool touches_entry =
                SweepTouches(segment, entry, radius, i, closest(2));
        const bool touches_exit =
                SweepTouches(segment, exit, radius, i, closest(3));
        if (touches_entry || touches_exit || !corridor.IsPointIn(start)) {
            record_failure("outside road corridor or at a road endpoint");
        }

        for (std::size_t j = 0; j < obstacles.size(); ++j) {
            if (SweepTouches(
                        segment, obstacles[j], radius, i, closest(j + 4))) {
                record_failure("obstacle " + std::to_string(j + 1));
            }
        }
        if (failure && !clearances) {
            return failure;
        }
    }
    return failure;
}

bool ReportPathValidation(const std::optional<PathValidationFailure> &failure,
                          bool print_summary) {
    if (failure) {
        std::cerr << "Path validation failed";
        if (failure->segment) {
            const auto i = *failure->segment;
            std::cerr << " at segment [" << i << ", " << i + 1 << "]";
        }
        std::cerr << ": " << failure->reason << '\n';
        return false;
    }
    if (print_summary) {
        std::cout << "Path validation passed\n";
    }
    return true;
}

} // namespace altro
