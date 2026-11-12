#include "projection_solver.h"

#include <Eigen/Cholesky>
#include <Eigen/SparseLU>
#include <qdldl.h>

#include <algorithm>
#include <type_traits>

namespace altro {
namespace {

constexpr ProjectionMethod kMethods[] = {
        ProjectionMethod::Riccati, ProjectionMethod::KKTEigen, ProjectionMethod::KKTQdldl};

using SparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor, QDLDL_int>;
using Triplets = std::vector<Eigen::Triplet<double, QDLDL_int>>;
static_assert(std::is_same_v<QDLDL_float, double>, "Projection requires double-precision QDLDL");

template<typename Scalar, unsigned StateDim, unsigned ControlDim>
bool SolveRiccati(const std::vector<LinearDynamics<Scalar, StateDim, ControlDim>> &dynamics,
                  const std::vector<StageQuadratic<Scalar, StateDim, ControlDim>> &q,
                  ProjectionStep<Scalar, StateDim, ControlDim> &step) {
    using StateHessian = Eigen::Matrix<Scalar, StateDim, StateDim>;
    using ControlHessian = Eigen::Matrix<Scalar, ControlDim, ControlDim>;
    using Feedback = Eigen::Matrix<Scalar, ControlDim, StateDim>;
    using StateVector = Eigen::Matrix<Scalar, StateDim, 1>;
    using ControlVector = Eigen::Matrix<Scalar, ControlDim, 1>;

    step.control_delta.resize(dynamics.size());
    step.feedback.resize(dynamics.size());
    StateHessian V = q.back().Qxx;
    StateVector v = q.back().qx;
    for (int t = static_cast<int>(dynamics.size()) - 1; t >= 0; --t) {
        const auto &[A, B] = dynamics[t];
        const StateHessian Sxx = q[t].Qxx + A.transpose() * V * A;
        const ControlHessian Suu = q[t].Quu + B.transpose() * V * B;
        const Feedback Sux = q[t].Qux + B.transpose() * V * A;
        const StateVector sx = q[t].qx + A.transpose() * v;
        const ControlVector su = q[t].qu + B.transpose() * v;
        const Eigen::LLT<ControlHessian> llt(Suu);
        if (llt.info() != Eigen::Success) {
            return false;
        }
        auto &K = step.feedback[t];
        auto &k = step.control_delta[t];
        K = -llt.solve(Sux);
        k = -llt.solve(su);
        if (!K.allFinite() || !k.allFinite()) {
            return false;
        }
        V = Sxx + Sux.transpose() * K;
        V = 0.5 * (V + V.transpose());
        v = sx + Sux.transpose() * k;
    }
    return true;
}

// Completing the square backward along the dynamics gives the pivots of the reduced Hessian.
// Read the same symmetric blocks as the sparse KKT assembly.
template<typename Scalar, unsigned StateDim, unsigned ControlDim>
bool HasPositiveReducedHessian(const std::vector<LinearDynamics<Scalar, StateDim, ControlDim>> &dynamics,
                               const std::vector<StageQuadratic<Scalar, StateDim, ControlDim>> &q) {
    using StateHessian = Eigen::Matrix<Scalar, StateDim, StateDim>;
    using ControlHessian = Eigen::Matrix<Scalar, ControlDim, ControlDim>;
    using ControlStateBlock = Eigen::Matrix<Scalar, ControlDim, StateDim>;

    StateHessian V = q.back().Qxx.template selfadjointView<Eigen::Upper>();
    for (std::size_t remaining = dynamics.size(); remaining > 0; --remaining) {
        const auto t = remaining - 1;
        const auto &[A, B] = dynamics[t];
        const ControlHessian Quu = q[t].Quu.template selfadjointView<Eigen::Upper>();
        const ControlHessian Suu = Quu + B.transpose() * V * B;
        if (!Suu.allFinite()) {
            return false;
        }
        const Eigen::LLT<ControlHessian, Eigen::Upper> llt(Suu);
        if (llt.info() != Eigen::Success) {
            return false;
        }
        if (t > 0) {
            const StateHessian Qxx = q[t].Qxx.template selfadjointView<Eigen::Upper>();
            const ControlStateBlock Sux = q[t].Qux + B.transpose() * V * A;
            V = (Qxx + A.transpose() * V * A - Sux.transpose() * llt.solve(Sux)).eval();
            V = (0.5 * V + 0.5 * V.transpose()).eval();
        }
    }
    return true;
}

template<typename Derived>
void AddUpperBlock(Triplets &entries, QDLDL_int row, QDLDL_int col, const Eigen::MatrixBase<Derived> &block) {
    for (Eigen::Index j = 0; j < block.cols(); ++j) {
        for (Eigen::Index i = 0; i < block.rows() && row + i <= col + j; ++i) {
            if (block(i, j) != 0.0) {
                entries.emplace_back(row + i, col + j, block(i, j));
            }
        }
    }
}

bool SolveWithEigen(const SparseMatrix &upper, const Eigen::VectorXd &rhs, Eigen::VectorXd &solution) {
    const SparseMatrix matrix = upper.selfadjointView<Eigen::Upper>();
    const Eigen::SparseLU<SparseMatrix> lu(matrix);
    if (lu.info() != Eigen::Success) {
        return false;
    }
    solution = lu.solve(rhs);
    return lu.info() == Eigen::Success;
}

bool SolveWithQdldl(const SparseMatrix &upper,
                    const Eigen::VectorXd &rhs,
                    QDLDL_int primal_size,
                    Eigen::VectorXd &solution) {
    const auto n = static_cast<QDLDL_int>(upper.rows());
    const QDLDL_int *Ap = upper.outerIndexPtr();
    const QDLDL_int *Ai = upper.innerIndexPtr();
    std::vector<QDLDL_int> tree(n), column_counts(n), iwork(3 * n);
    const QDLDL_int nnz = QDLDL_etree(n, Ap, Ai, iwork.data(), column_counts.data(), tree.data());
    if (nnz < 0) {
        return false;
    }

    std::vector<QDLDL_int> Lp(n + 1), Li(nnz);
    std::vector<double> Lx(nnz), D(n), Dinv(n), fwork(n);
    std::vector<QDLDL_bool> bwork(n);
    const auto positive_pivots = QDLDL_factor(n,
                                              Ap,
                                              Ai,
                                              upper.valuePtr(),
                                              Lp.data(),
                                              Li.data(),
                                              Lx.data(),
                                              D.data(),
                                              Dinv.data(),
                                              column_counts.data(),
                                              tree.data(),
                                              bwork.data(),
                                              iwork.data(),
                                              fwork.data());
    // A minimum on the dynamics null space gives one positive pivot per primal variable.
    if (positive_pivots != primal_size || !Eigen::Map<const Eigen::VectorXd>(Dinv.data(), n).allFinite()) {
        return false;
    }
    solution = rhs;
    QDLDL_solve(n, Lp.data(), Li.data(), Lx.data(), Dinv.data(), solution.data());
    return true;
}

template<typename Scalar, unsigned StateDim, unsigned ControlDim>
bool SolveKkt(ProjectionMethod method,
              const std::vector<LinearDynamics<Scalar, StateDim, ControlDim>> &dynamics,
              const std::vector<StageQuadratic<Scalar, StateDim, ControlDim>> &q,
              ProjectionStep<Scalar, StateDim, ControlDim> &step) {
    const bool use_eigen = method == ProjectionMethod::KKTEigen;
    if (use_eigen && !HasPositiveReducedHessian(dynamics, q)) {
        return false;
    }

    // Primal variables are stage ordered: [u_0, x_1, u_1, ..., x_T]. Place the dynamics
    // multipliers last so LDL^T eliminates the stage Hessians before the zero diagonal block.
    const auto horizon = static_cast<QDLDL_int>(dynamics.size());
    const QDLDL_int primal_size = horizon * (StateDim + ControlDim);
    const QDLDL_int size = primal_size + horizon * StateDim;
    const auto control_index = [](QDLDL_int t) { return t * (StateDim + ControlDim); };
    const auto state_index = [&](QDLDL_int t) { return control_index(t - 1) + ControlDim; };
    Triplets entries;
    entries.reserve(horizon * (StateDim + ControlDim) * (StateDim + ControlDim + 1));
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(size);

    // Upper triangle of [Q D^T; D 0], with D dz = dx_{t+1} - A dx_t - B du_t.
    for (QDLDL_int t = 0; t <= horizon; ++t) {
        if (t > 0) {
            const auto x = state_index(t);
            AddUpperBlock(entries, x, x, q[t].Qxx);
            rhs.segment<StateDim>(x) = -q[t].qx;
        }
        if (t == horizon) {
            break;
        }
        const auto u = control_index(t);
        const auto lambda = primal_size + t * StateDim;
        const auto &[A, B] = dynamics[t];
        AddUpperBlock(entries, u, u, q[t].Quu);
        rhs.segment<ControlDim>(u) = -q[t].qu;
        if (t > 0) {
            AddUpperBlock(entries, state_index(t), u, q[t].Qux.transpose());
            AddUpperBlock(entries, state_index(t), lambda, -A.transpose());
        }
        AddUpperBlock(entries, u, lambda, -B.transpose());
        for (QDLDL_int i = 0; i < StateDim; ++i) {
            entries.emplace_back(state_index(t + 1) + i, lambda + i, 1.0);
        }
    }

    SparseMatrix upper(size, size);
    upper.setFromTriplets(entries.begin(), entries.end());
    upper.makeCompressed();
    if (!rhs.allFinite() || !Eigen::Map<const Eigen::VectorXd>(upper.valuePtr(), upper.nonZeros()).allFinite()) {
        return false;
    }

    Eigen::VectorXd solution;
    const bool solved =
            use_eigen ? SolveWithEigen(upper, rhs, solution) : SolveWithQdldl(upper, rhs, primal_size, solution);
    if (!solved || !solution.allFinite()) {
        return false;
    }
    const Eigen::VectorXd residual = upper.selfadjointView<Eigen::Upper>() * solution - rhs;
    constexpr double kResidualTolerance = 1.0e-8;
    if (!residual.allFinite() ||
        residual.head(primal_size).lpNorm<Eigen::Infinity>() >
                kResidualTolerance * std::max(1.0, rhs.lpNorm<Eigen::Infinity>()) ||
        residual.tail(size - primal_size).lpNorm<Eigen::Infinity>() >
                kResidualTolerance * std::max(1.0, solution.head(primal_size).lpNorm<Eigen::Infinity>())) {
        return false;
    }

    step.feedback.clear();
    step.control_delta.resize(horizon);
    for (QDLDL_int t = 0; t < horizon; ++t) {
        step.control_delta[t] = solution.segment<ControlDim>(control_index(t));
    }
    return true;
}

} // namespace

std::string_view ProjectionMethodName(ProjectionMethod method) {
    switch (method) {
        case ProjectionMethod::Riccati:
            return "riccati";
        case ProjectionMethod::KKTEigen:
            return "kkt-eigen";
        case ProjectionMethod::KKTQdldl:
            return "kkt-qdldl";
    }
    return "unknown";
}

std::optional<ProjectionMethod> ParseProjectionMethod(std::string_view name) {
    for (auto method : kMethods) {
        if (ProjectionMethodName(method) == name) {
            return method;
        }
    }
    return std::nullopt;
}

template<typename Scalar, unsigned StateDim, unsigned ControlDim>
bool SolveProjection(ProjectionMethod method,
                     const std::vector<LinearDynamics<Scalar, StateDim, ControlDim>> &dynamics,
                     const std::vector<StageQuadratic<Scalar, StateDim, ControlDim>> &quadratics,
                     ProjectionStep<Scalar, StateDim, ControlDim> &step) {
    if (dynamics.empty() || quadratics.size() != dynamics.size() + 1) {
        return false;
    }
    switch (method) {
        case ProjectionMethod::Riccati:
            return SolveRiccati(dynamics, quadratics, step);
        case ProjectionMethod::KKTEigen:
        case ProjectionMethod::KKTQdldl:
            return SolveKkt(method, dynamics, quadratics, step);
    }
    return false;
}

template bool SolveProjection(ProjectionMethod,
                              const std::vector<LinearDynamics<double, 5, 1>> &,
                              const std::vector<StageQuadratic<double, 5, 1>> &,
                              ProjectionStep<double, 5, 1> &);
template bool SolveProjection(ProjectionMethod,
                              const std::vector<LinearDynamics<double, 4, 2>> &,
                              const std::vector<StageQuadratic<double, 4, 2>> &,
                              ProjectionStep<double, 4, 2> &);
template bool SolveProjection(ProjectionMethod,
                              const std::vector<LinearDynamics<double, 6, 2>> &,
                              const std::vector<StageQuadratic<double, 6, 2>> &,
                              ProjectionStep<double, 6, 2> &);

} // namespace altro
