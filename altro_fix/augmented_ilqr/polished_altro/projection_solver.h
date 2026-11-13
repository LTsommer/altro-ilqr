#ifndef ALILQR_PROJECTION_SOLVER_H
#define ALILQR_PROJECTION_SOLVER_H

#include <Eigen/Core>
#include <optional>
#include <string_view>
#include <vector>

namespace altro {

enum class ProjectionMethod {
    Riccati,
    KKTEigen,
    KKTQdldl,
};

std::string_view ProjectionMethodName(ProjectionMethod method);
std::optional<ProjectionMethod> ParseProjectionMethod(std::string_view name);

// dx_{t+1} = A dx_t + B du_t, with dx_0 fixed at zero.
template<typename Scalar, unsigned StateDim, unsigned ControlDim>
struct LinearDynamics {
    Eigen::Matrix<Scalar, StateDim, StateDim> A;
    Eigen::Matrix<Scalar, StateDim, ControlDim> B;
};

// Hessian and gradient of the quadratic objective at one stage, including active-row penalties.
template<typename Scalar, unsigned StateDim, unsigned ControlDim>
struct StageQuadratic {
    Eigen::Matrix<Scalar, StateDim, StateDim> Qxx;
    Eigen::Matrix<Scalar, ControlDim, ControlDim> Quu;
    Eigen::Matrix<Scalar, ControlDim, StateDim> Qux;
    Eigen::Matrix<Scalar, StateDim, 1> qx;
    Eigen::Matrix<Scalar, ControlDim, 1> qu;
};

template<typename Scalar, unsigned StateDim, unsigned ControlDim>
struct ProjectionStep {
    std::vector<Eigen::Matrix<Scalar, ControlDim, 1>> control_delta;
    // Empty for KKT.
    std::vector<Eigen::Matrix<Scalar, ControlDim, StateDim>> feedback;
};

// The terminal quadratic contains only state terms. The output is valid on success.
// Explicit instantiations match the three ALILQRSolver specializations.
template<typename Scalar, unsigned StateDim, unsigned ControlDim>
bool SolveProjection(
        ProjectionMethod method,
        const std::vector<LinearDynamics<Scalar, StateDim, ControlDim>>
                &dynamics,
        const std::vector<StageQuadratic<Scalar, StateDim, ControlDim>>
                &quadratics,
        ProjectionStep<Scalar, StateDim, ControlDim> &step);

} // namespace altro

#endif // ALILQR_PROJECTION_SOLVER_H
