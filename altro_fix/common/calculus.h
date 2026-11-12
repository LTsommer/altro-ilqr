//
// Created by LTsommer on 2024/6/13.
//

#ifndef CILQR_CALCULUS_H
#define CILQR_CALCULUS_H

#include "../frenet_coordinate/spline.h"
#include "trig_table.h"
#include <iostream>
#include <type_traits>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <algorithm>
#include <array>

using Eigen::Matrix;

namespace gauss_legendre {
inline constexpr int kNumNodes = 10;
inline constexpr std::array<double, kNumNodes> kNodes{
        -0.9739065285171717, -0.8650633666889845,
        -0.6794095682990244, -0.4333953941292472,
        -0.1488743389816312, 0.1488743389816312,
        0.4333953941292472, 0.6794095682990244,
        0.8650633666889845, 0.9739065285171717,
};
inline constexpr std::array<double, kNumNodes> kWeights{
        0.0666713443086881, 0.1494513491505806,
        0.2190863625159820, 0.2692667193099963,
        0.2955242247147529, 0.2955242247147529,
        0.2692667193099963, 0.2190863625159820,
        0.1494513491505806, 0.0666713443086881,
};
} // namespace gauss_legendre

// Displacement over one step of length dt with theta and velocity linear between the two ends.
// Pairs the symmetric quadrature nodes in the frame of the midpoint heading.
// Preconditions (guaranteed by the solver): dt > 0 and all inputs finite.
inline Eigen::Vector2d IntegratePlanarPosition(
        double theta0, double theta1, double velocity0, double velocity1, double dt) {
    const double half_dt = dt / 2.0;
    const double half_theta_change = ((theta1 - theta0) / dt) * half_dt;
    const double half_velocity_change = ((velocity1 - velocity0) / dt) * half_dt;
    const double mid_heading = theta0 + half_theta_change;
    const double mid_velocity = velocity0 + half_velocity_change;
    double weighted_cos = 0.0;
    double weighted_node_sin = 0.0;
    for (int i = gauss_legendre::kNumNodes / 2; i < gauss_legendre::kNumNodes; ++i) {
        const double node = gauss_legendre::kNodes[i];
        const double angle = node * half_theta_change;
        const double weight = gauss_legendre::kWeights[i];
        double sin_angle, cos_angle;
        trig_table::SinCos(angle, &sin_angle, &cos_angle);
        weighted_cos += weight * cos_angle;
        weighted_node_sin += weight * node * sin_angle;
    }
    // Even terms use the sum of the paired velocities, odd terms use their difference.
    const double longitudinal = dt * mid_velocity * weighted_cos;
    const double lateral = dt * half_velocity_change * weighted_node_sin;
    double sin_heading, cos_heading;
    trig_table::SinCos(mid_heading, &sin_heading, &cos_heading);
    return {cos_heading * longitudinal - sin_heading * lateral,
            sin_heading * longitudinal + cos_heading * lateral};
}

// Partials of the IntegratePlanarPosition displacement w.r.t. its end values (theta and
// velocity linear between the two ends, same 10-point Gauss-Legendre rule). Same preconditions
// as IntegratePlanarPosition.
struct PlanarPositionPartials {
    enum Index {
        kTheta0 = 0,
        kTheta1,
        kVel0,
        kVel1,
    };
    std::array<double, 4> dx{};
    std::array<double, 4> dy{};
};

inline PlanarPositionPartials IntegratePlanarPositionPartials(
        double theta0, double theta1, double velocity0, double velocity1, double dt) {
    constexpr int kNum = gauss_legendre::kNumNodes;
    using NodeArray = Eigen::Array<double, kNum, 1>;
    const Eigen::Map<const NodeArray> nodes(gauss_legendre::kNodes.data());
    const Eigen::Map<const NodeArray> weights(gauss_legendre::kWeights.data());

    // Column 0: weight of the start value at each node (1 - s); column 1: of the end value (s).
    Eigen::Matrix<double, kNum, 2> end_weights;
    end_weights.col(1) = (0.5 * (nodes + 1.0)).matrix(); // position of each node in [0, 1]
    end_weights.col(0) = (1.0 - end_weights.col(1).array()).matrix();

    const NodeArray s = end_weights.col(1).array();
    const NodeArray theta = theta0 + (theta1 - theta0) * s;
    const NodeArray velocity = velocity0 + (velocity1 - velocity0) * s;
    NodeArray sin_theta, cos_theta;
    for (int i = 0; i < kNum; ++i) {
        trig_table::SinCos(theta(i), &sin_theta(i), &cos_theta(i));
    }
    const NodeArray w = 0.5 * dt * weights;

    // dx = sum w cos(theta) v, dy = sum w sin(theta) v. Columns: d/dtheta and d/dv of each
    // node's contribution to dx, then to dy.
    Eigen::Matrix<double, kNum, 4> node_partials;
    node_partials.col(0) = (-w * sin_theta * velocity).matrix();
    node_partials.col(1) = (w * cos_theta).matrix();
    node_partials.col(2) = (w * cos_theta * velocity).matrix();
    node_partials.col(3) = (w * sin_theta).matrix();

    // Row 0: w.r.t. the start values; row 1: w.r.t. the end values.
    const Eigen::Matrix<double, 2, 4> end_partials = end_weights.transpose() * node_partials;

    using Index = PlanarPositionPartials::Index;
    PlanarPositionPartials p;
    p.dx[Index::kTheta0] = end_partials(0, 0);
    p.dx[Index::kTheta1] = end_partials(1, 0);
    p.dx[Index::kVel0] = end_partials(0, 1);
    p.dx[Index::kVel1] = end_partials(1, 1);
    p.dy[Index::kTheta0] = end_partials(0, 2);
    p.dy[Index::kTheta1] = end_partials(1, 2);
    p.dy[Index::kVel0] = end_partials(0, 3);
    p.dy[Index::kVel1] = end_partials(1, 3);
    return p;
}

#endif //CILQR_CALCULUS_H
