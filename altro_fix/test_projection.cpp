#include "augmented_ilqr/polished_altro/projection_solver.h"

#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

using Dynamics = altro::LinearDynamics<double, 6, 2>;
using Quadratic = altro::StageQuadratic<double, 6, 2>;
using Step = altro::ProjectionStep<double, 6, 2>;

Quadratic ZeroQuadratic() {
    Quadratic q;
    q.Qxx.setZero();
    q.Quu.setZero();
    q.Qux.setZero();
    q.qx.setZero();
    q.qu.setZero();
    return q;
}

Dynamics MakeDynamics() {
    Dynamics dynamics;
    dynamics.A.setIdentity();
    dynamics.B.setZero();
    return dynamics;
}

bool RejectsActiveRowSaddle() {
    const std::vector<Dynamics> dynamics{MakeDynamics()};
    std::vector<Quadratic> q(2, ZeroQuadratic());
    // H = diag(-1.001, 1), active row [1, 1], violation 0.01, and damping 1e-6.
    q[0].Quu << 1.0e6 - 1.001, 1.0e6, 1.0e6, 1.0e6 + 1.0;
    q[0].qu.setConstant(1.0e4);
    Step step;
    return !altro::SolveProjection(altro::ProjectionMethod::KKTEigen, dynamics, q, step);
}

bool RejectsNegativeCurvatureFromTerminalCost() {
    auto stage = MakeDynamics();
    stage.B.topRows<2>().setIdentity();
    const std::vector<Dynamics> dynamics{stage, stage};
    std::vector<Quadratic> q(3, ZeroQuadratic());
    q[0].Quu.setIdentity();
    q[1].Quu = 3.0 * Eigen::Matrix2d::Identity();
    q[2].Qxx(0, 0) = -1.0;
    q[0].qu(0) = 1.0;
    // The first control component has reduced Hessian [[0, -1], [-1, 2]].
    Step step;
    return !altro::SolveProjection(altro::ProjectionMethod::KKTEigen, dynamics, q, step);
}

bool AcceptsPositiveReducedHessian() {
    auto stage = MakeDynamics();
    stage.B.topRows<2>().setIdentity();
    const std::vector<Dynamics> dynamics{stage, stage};
    std::vector<Quadratic> q(3, ZeroQuadratic());
    q[0].Quu.diagonal() << -1.0, 1.0;
    q[0].qu << 1.0, -1.0;
    q[1].Quu = 3.0 * Eigen::Matrix2d::Identity();
    q[2].Qxx.topLeftCorner<2, 2>() = 2.0 * Eigen::Matrix2d::Identity();
    Step step;
    if (!altro::SolveProjection(altro::ProjectionMethod::KKTEigen, dynamics, q, step)) {
        return false;
    }
    // Substituting x_2 = u_0 + u_1 into the objective gives these two control increments.
    return step.control_delta.size() == 2 && step.feedback.empty() &&
           step.control_delta[0].isApprox(Eigen::Vector2d(-5.0, 5.0 / 11.0), 1.0e-10) &&
           step.control_delta[1].isApprox(Eigen::Vector2d(2.0, -2.0 / 11.0), 1.0e-10);
}

} // namespace

int main() {
    bool passed = true;
    if (!RejectsActiveRowSaddle()) {
        std::cerr << "Eigen KKT accepted a saddle with a small stationarity residual\n";
        passed = false;
    }
    if (!RejectsNegativeCurvatureFromTerminalCost()) {
        std::cerr << "Eigen KKT accepted negative curvature propagated from the terminal cost\n";
        passed = false;
    }
    if (!AcceptsPositiveReducedHessian()) {
        std::cerr << "Eigen KKT failed to solve a positive reduced Hessian with an indefinite stage Hessian\n";
        passed = false;
    }
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
