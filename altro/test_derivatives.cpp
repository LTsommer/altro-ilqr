#include "augmented_ilqr/solver_term_registry.hpp"
#include "model/smooth_kappa_model.h"
#include "motion_planning/cost_functions/cost_terms.hpp"
#include "motion_planning/mix_model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace {

constexpr double kPerturbation = 1e-5;
struct DerivativeTolerance {
    double absolute;
    double relative;
};

constexpr DerivativeTolerance kExactGradient{1e-8, 1e-8};
constexpr DerivativeTolerance kDynamicsJacobian{1e-8, 1e-7};
constexpr DerivativeTolerance kCostHessian{1e-7, 1e-7};
constexpr DerivativeTolerance kAggregateGradient{1e-7, 1e-7};
// The curved fixture has about 0.15% error from the Frenet approximation.
constexpr DerivativeTolerance kCurvedPositionGradient{1e-8, 2.5e-3};
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr int kRunning = 0;
constexpr int kTerminal = 1;
using Scalar = Eigen::Matrix<double, 1, 1>;

template<typename Point, typename Jacobian, typename Evaluate>
bool CheckDerivative(
        const std::string &name,
        const Point &point,
        const Jacobian &analytic,
        Evaluate evaluate,
        DerivativeTolerance tolerance,
        double perturbation = kPerturbation,
        const std::array<DerivativeTolerance, Point::SizeAtCompileTime>
                *column_tolerances = nullptr) {
    bool passed = true;
    for (Eigen::Index col = 0; col < point.size(); ++col) {
        Point plus = point;
        Point minus = point;
        plus(col) += perturbation;
        minus(col) -= perturbation;
        const auto upper = evaluate(plus);
        const auto lower = evaluate(minus);
        const auto numeric = (upper - lower) / (plus(col) - minus(col));
        const auto bounds =
                column_tolerances ? (*column_tolerances)[col] : tolerance;
        for (Eigen::Index row = 0; row < analytic.rows(); ++row) {
            const double expected = analytic(row, col);
            const double actual = numeric(row);
            const double error = std::abs(expected - actual);
            const double limit = bounds.absolute +
                                 bounds.relative * std::max(std::abs(expected),
                                                            std::abs(actual));
            if (!std::isfinite(expected) || !std::isfinite(actual) ||
                error > limit) {
                std::cerr << name << " (" << row << ", " << col
                          << "): analytic=" << std::setprecision(17) << expected
                          << ", central difference=" << actual
                          << ", error=" << error << ", limit=" << limit << '\n';
                passed = false;
            }
        }
    }
    return passed;
}

template<typename Model>
bool CheckDynamics(const std::string &name,
                   const Model &model,
                   const typename Model::State &x,
                   const typename Model::Control &u,
                   double step) {
    const auto nominal = model.ForwardCalculation(x, u, step);
    bool passed = CheckDerivative(
            name + " JacobianX",
            x,
            model.JacobianX(x, u, step),
            [&](const auto &state) {
                return model.StateDifference(
                        model.ForwardCalculation(state, u, step), nominal);
            },
            kDynamicsJacobian);
    passed &= CheckDerivative(
            name + " JacobianU",
            u,
            model.JacobianU(x, u, step),
            [&](const auto &control) {
                return model.StateDifference(
                        model.ForwardCalculation(x, control, step), nominal);
            },
            kDynamicsJacobian);
    return passed;
}

template<typename Model>
bool CheckDynamicsAtHeadingWrap(const std::string &name,
                                const Model &model,
                                typename Model::State x,
                                const typename Model::Control &u,
                                double step,
                                Eigen::Index heading) {
    x(heading) = 0.0;
    x(heading) = kPi - model.ForwardCalculation(x, u, step)(heading);
    auto plus = x;
    auto minus = x;
    plus(heading) += kPerturbation;
    minus(heading) -= kPerturbation;
    if (!(model.ForwardCalculation(plus, u, step)(heading) < 0.0 &&
          model.ForwardCalculation(minus, u, step)(heading) > 0.0)) {
        std::cerr << name << ": heading samples must straddle the wrap\n";
        return false;
    }
    return CheckDynamics(name, model, x, u, step);
}

template<typename Term>
Eigen::Matrix<double, 1, 1> EvaluateValue(const Term &term,
                                          int step,
                                          const typename Term::MatrixZ &point) {
    constexpr int kStateSize = Term::State::RowsAtCompileTime;
    constexpr int kControlSize = Term::Control::RowsAtCompileTime;
    Eigen::Matrix<double, 1, 1> value(kNaN);
    if (!term.Evaluate(step,
                       point.template head<kStateSize>(),
                       point.template tail<kControlSize>(),
                       value(0))) {
        value(0) = kNaN;
    }
    return value;
}

template<typename Term>
typename Term::MatrixZ EvaluateGradient(const Term &term,
                                        int step,
                                        const typename Term::MatrixZ &point) {
    constexpr int kStateSize = Term::State::RowsAtCompileTime;
    constexpr int kControlSize = Term::Control::RowsAtCompileTime;
    typename Term::MatrixZ gradient = Term::MatrixZ::Constant(kNaN);
    if (!term.Gradient(step,
                       point.template head<kStateSize>(),
                       point.template tail<kControlSize>(),
                       gradient.template head<kStateSize>(),
                       gradient.template tail<kControlSize>())) {
        gradient.setConstant(kNaN);
    }
    return gradient;
}

bool CheckCentripetalJerkCost(const MixModel::State &x,
                              const MixModel::Control &u) {
    using Cost = CentripetalJerkCostFunc;
    CostTermConfig config{};
    config.steps = {0.4};
    config.weights.at(WeightLatJerk) = 1.7;
    const Cost cost(config);
    constexpr int kStep = 0;

    Cost::MatrixZ point;
    point << x, u;
    const auto gradient = EvaluateGradient(cost, kStep, point);
    bool passed = CheckDerivative(
            "CentripetalJerkCostFunc gradient",
            point,
            gradient.transpose(),
            [&](const auto &sample) {
                return EvaluateValue(cost, kStep, sample);
            },
            kExactGradient);

    Cost::MatrixLXX hxx = Cost::MatrixLXX::Constant(kNaN);
    Cost::MatrixLUU huu = Cost::MatrixLUU::Constant(kNaN);
    Cost::MatrixLXU hxu = Cost::MatrixLXU::Constant(kNaN);
    if (!cost.Hessian(kStep, x, u, hxx, huu, hxu)) {
        std::cerr
                << "CentripetalJerkCostFunc: Hessian rejected a running step\n";
        return false;
    }
    Cost::MatrixHZ hessian;
    hessian << hxx, hxu, hxu.transpose(), huu;
    passed &= CheckDerivative(
            "CentripetalJerkCostFunc Hessian",
            point,
            hessian,
            [&](const auto &sample) {
                return EvaluateGradient(cost, kStep, sample);
            },
            kCostHessian);
    return passed;
}

CostTermConfig MakeConfig() {
    CostTermConfig config{};
    config.steps = {0.4};
    for (auto &[index, weight] : config.weights) {
        weight = 0.7 + 0.1 * index;
    }
    config.p_ref = {{4.8, 3.7}, {5.4, 4.2}};
    config.velocity = {2.8, 3.1};
    config.acceleration = {0.5, -0.4};
    config.yaw = {0.1, -0.1};
    config.ref_x = {0.0, 3.0, 6.0, 9.0, 12.0, 15.0};
    config.ref_y = {0.0, 4.0, 8.0, 12.0, 16.0, 20.0};
    config.p_bound.assign(2, {{4, 7}, {6, 9}, {4, 3}, {6, 5}});
    config.left_safe_dist = {0.2, 0.25};
    config.right_safe_dist = {0.15, 0.2};
    config.left_comfort_dist = {0.5, 0.55};
    config.right_comfort_dist = {0.4, 0.45};
    config.u_min = {-0.3};
    config.u_max = {0.4};
    config.yaw_diff_min = -0.4;
    config.yaw_diff_max = 0.5;
    config.kappa_min = -0.15;
    config.kappa_max = 0.3;
    config.acc_min = -0.8;
    config.acc_max = 0.9;
    config.jerk_min = -0.9;
    config.jerk_max = 1.1;
    config.vel_min = 2.0;
    config.vel_max = 5.0;
    config.lat_acc_min = -0.25;
    config.lat_acc_max = 3.0;
    config.lat_jerk_min = -0.6;
    config.lat_jerk_max = 3.0;
    return config;
}

template<typename Term>
bool CheckTerm(const Term &term,
               int step,
               const typename Term::MatrixZ &point,
               const std::string &label) {
    const auto gradient = EvaluateGradient(term, step, point);
    return CheckDerivative(
            label,
            point,
            gradient.transpose(),
            [&](const auto &sample) {
                return EvaluateValue(term, step, sample);
            },
            kExactGradient);
}

template<typename Term>
bool CheckRunningAndTerminal(const Term &term,
                             const typename Term::MatrixZ &point) {
    const auto name = term.GetName();
    bool passed = CheckTerm(term, kRunning, point, name + " running");
    passed &= CheckTerm(term, kTerminal, point, name + " terminal");
    return passed;
}

template<typename Penalty>
bool CheckPenalty(const Penalty &penalty,
                  const std::string &name,
                  double residual,
                  double perturbation = kPerturbation) {
    Scalar gradient(kNaN);
    penalty.Gradient(residual, gradient(0));
    return CheckDerivative(
            name,
            Scalar(residual),
            gradient,
            [&](const Scalar &sample) {
                return Scalar(penalty.Evaluate(sample(0)));
            },
            kExactGradient,
            perturbation);
}

bool CheckPenalties() {
    bool passed = true;
    for (const double residual : {-0.5, 0.8}) {
        passed &= CheckPenalty(OneSideCost{}, "OneSideCost", residual);
        passed &= CheckPenalty(
                OneSideQuadraticCost{}, "OneSideQuadraticCost", residual);
        passed &= CheckPenalty(
                OneSideEqualityCost{}, "OneSideEqualityCost", residual);
    }
    passed &= CheckPenalty(
            OneSideEqualityCost{}, "OneSideEqualityCost deadband", 0.0, 1e-7);
    return passed;
}

template<unsigned M, unsigned N>
void LoadCosts(OCPInterface<double, M, N> &problem,
               const CostTermConfig &config,
               const altro::SolverTermRegistry<M, N> &registry) {
    problem.SetConfig(config);
    problem.SetCostUnion(std::make_unique<CostUnion<double, M, N>>(config));
    for (const auto &entry : registry.costs) {
        entry.add(problem);
    }
}

template<unsigned M, unsigned N>
bool CheckRegisteredCosts(
        const CostTermConfig &config,
        const altro::SolverTermRegistry<M, N> &registry,
        const typename OCPInterface<double, M, N>::MatrixZ &point) {
    OCPInterface<double, M, N> problem;
    LoadCosts(problem, config, registry);
    const auto &terms = problem.GetCostUnion().GetCostTerms();
    bool passed = true;
    for (std::size_t i = 0; i < terms.size(); ++i) {
        const auto &term = *terms[i];
        const auto name = term.GetName();
        const auto key = registry.costs[i].name;
        if (key == "comfort_dist") {
            for (const int step : {kRunning, kTerminal}) {
                for (const double y : {3.8, 6.0, 9.0}) {
                    auto sample = point;
                    sample(kYPos) = y;
                    passed &= CheckTerm(term,
                                        step,
                                        sample,
                                        name + " y=" + std::to_string(y));
                }
            }
        } else if (key == "lat_jerk" || key == "dot_kappa" || key == "jerk") {
            passed &= CheckTerm(term, kRunning, point, name + " running");
        } else {
            passed &= CheckRunningAndTerminal(term, point);
        }
    }
    return passed;
}

using LegacyProblem = OCPInterface<double, 4, 2>;

bool CheckUnregisteredCosts(const CostTermConfig &config,
                            const SmoothKappaModel::MatrixZ &smooth,
                            const LegacyProblem::MatrixZ &legacy) {
    bool passed =
            CheckRunningAndTerminal(AccCostFunc<double, 5, 1>(config), smooth);
    passed &= CheckRunningAndTerminal(VelDiffCostFunc<double, 5, 1>(config),
                                      smooth);
    passed &=
            CheckRunningAndTerminal(AccCostFunc<double, 4, 2>(config), legacy);
    passed &= CheckRunningAndTerminal(VelDiffCostFunc<double, 4, 2>(config),
                                      legacy);
    passed &= CheckRunningAndTerminal(OffsetCostFunc<double, 4, 2>(config),
                                      legacy);
    return passed;
}

template<unsigned M, unsigned N>
bool CheckConstraint(
        const Constraint<double, M, N, Inequality> &term,
        int step,
        const typename Constraint<double, M, N, Inequality>::MatrixZ &point,
        const std::string &label,
        Eigen::Index row_count,
        int active_side = -2) {
    const auto x = point.template head<M>();
    const auto u = point.template tail<N>();
    bool passed = CheckTerm(term, step, point, label);
    RawConstraintRows<double, M, N> rows;
    if (!term.RawConstraints(step, x, u, rows) || rows.size() != row_count) {
        std::cerr << label << ": expected " << row_count << " raw rows\n";
        return false;
    }
    Eigen::Matrix<double, Eigen::Dynamic, M + N> jacobian(row_count, M + N);
    for (Eigen::Index i = 0; i < row_count; ++i) {
        jacobian.row(i) << rows[i].gx.transpose(), rows[i].gu.transpose();
        if (active_side >= -1 &&
            (i == active_side ? rows[i].g <= 0.01 : rows[i].g >= -0.01)) {
            std::cerr << label << ": unexpected branch for raw row " << i
                      << " with residual " << rows[i].g << '\n';
            passed = false;
        }
    }
    passed &= CheckDerivative(
            label + " raw rows",
            point,
            jacobian,
            [&](const auto &sample) {
                RawConstraintRows<double, M, N> perturbed;
                Eigen::VectorXd values =
                        Eigen::VectorXd::Constant(row_count, kNaN);
                if (term.RawConstraints(step,
                                        sample.template head<M>(),
                                        sample.template tail<N>(),
                                        perturbed) &&
                    perturbed.size() == row_count) {
                    for (Eigen::Index i = 0; i < row_count; ++i) {
                        values(i) = perturbed[i].g;
                    }
                }
                return values;
            },
            kExactGradient);
    return passed;
}

template<unsigned M, unsigned N>
bool CheckBoundPair(
        const Constraint<double, M, N, Inequality> &term,
        const std::array<typename Constraint<double, M, N, Inequality>::MatrixZ,
                         3> &points,
        bool running_only) {
    bool passed = true;
    const std::array<const char *, 3> cases{" lower", " feasible", " upper"};
    const std::array<int, 3> active_sides{0, -1, 1};
    for (int step = kRunning; step <= (running_only ? kRunning : kTerminal);
         ++step) {
        for (std::size_t i = 0; i < points.size(); ++i) {
            passed &= CheckConstraint(
                    term,
                    step,
                    points[i],
                    term.GetName() + cases[i] + " step=" + std::to_string(step),
                    2,
                    active_sides[i]);
        }
    }
    return passed;
}

template<unsigned M, unsigned N>
bool CheckRegisteredBounds(
        const CostTermConfig &config,
        const altro::SolverTermRegistry<M, N> &registry,
        const std::array<typename OCPInterface<double, M, N>::MatrixZ, 3>
                &points) {
    bool passed = true;
    for (const auto &entry : registry.constraints) {
        if (entry.name == "speed_limit") {
            continue;
        }
        const bool running_only = entry.name == "control" ||
                                  entry.name == "jerk" ||
                                  entry.name == "lat_jerk";
        // Raw rows use left/right order for road boundaries, and lower/upper for scalar bounds.
        if (entry.name == "safe") {
            auto by_side = points;
            std::swap(by_side[0](kYPos), by_side[2](kYPos));
            passed &=
                    CheckBoundPair(*entry.make(config), by_side, running_only);
        } else {
            passed &= CheckBoundPair(*entry.make(config), points, running_only);
        }
    }
    return passed;
}

template<unsigned M, unsigned N>
bool CheckBoundaryMargins(
        CostTermConfig config,
        const typename OCPInterface<double, M, N>::MatrixZ &outside) {
    config.boundary_types[0] = {BoundType::Default, BoundType::Others};
    bool passed = true;
    for (const auto margin : {BoundMargin::Safe, BoundMargin::Comfort}) {
        for (const bool left : {true, false}) {
            const LateralOffsetConstraint<double, M, N> term(
                    config, left, margin);
            passed &= CheckRunningAndTerminal(term, outside);
        }
    }
    const ComfortDistanceCostFunc<double, M, N> comfort(config);
    passed &= CheckTerm(comfort,
                        kRunning,
                        outside,
                        "ComfortDistanceCostFunc disabled left");
    const SafeConstraint<double, M, N> safety(config);
    passed &= CheckConstraint(
            safety, kRunning, outside, "SafeConstraint disabled left", 2);
    return passed;
}

using SpeedLimitTolerances =
        std::array<DerivativeTolerance, MixModel::MatrixZ::SizeAtCompileTime>;

bool CheckSpeedLimit(const SpeedLimitBySConstraint &constraint,
                     const MixModel::MatrixZ &point,
                     const std::string &name,
                     double perturbation = kPerturbation,
                     const SpeedLimitTolerances *column_tolerances = nullptr,
                     double expected_residual = kNaN,
                     double cold_residual_tolerance = 0.0) {
    bool passed = true;
    for (const int step : {kRunning, kTerminal}) {
        auto cold = constraint;
        const double cold_value = EvaluateValue(cold, step, point)(0);
        auto nominal = constraint;
        // Seed the warm path at a nearby position before evaluating the test point.
        auto neighbor = point;
        neighbor(mXPos) += 0.02;
        const double seed_value = EvaluateValue(nominal, step, neighbor)(0);
        const double warm_value = EvaluateValue(nominal, step, point)(0);
        const auto label = name + " step=" + std::to_string(step);
        if (!std::isfinite(seed_value) || !std::isfinite(cold_value) ||
            !std::isfinite(warm_value) ||
            (std::isfinite(expected_residual) &&
             (std::abs(cold_value - expected_residual) >
                      cold_residual_tolerance ||
              std::abs(warm_value - expected_residual) > 1e-7))) {
            std::cerr << label << ": cold=" << cold_value
                      << ", warm=" << warm_value
                      << ", expected=" << expected_residual << '\n';
            return false;
        }
        // The cold search uses a coarse stopping criterion. Differentiate the
        // local value with the same nominal warm cache on both sides.
        const auto gradient = EvaluateGradient(nominal, step, point);
        passed &= CheckDerivative(
                label + " warm",
                point,
                gradient.transpose(),
                [&](const auto &sample) {
                    auto copy = nominal;
                    return EvaluateValue(copy, step, sample);
                },
                kExactGradient,
                perturbation,
                column_tolerances);

        const auto raw_value = [&](const MixModel::MatrixZ &sample,
                                   MixModel::MatrixZ *raw_gradient) {
            auto copy = nominal;
            RawConstraintRows<double, 6, 2> rows;
            if (!copy.RawConstraints(
                        step, sample.head<6>(), sample.tail<2>(), rows) ||
                rows.size() != 1) {
                return Scalar(kNaN);
            }
            if (raw_gradient) {
                *raw_gradient << rows[0].gx, rows[0].gu;
            }
            return Scalar(rows[0].g);
        };
        MixModel::MatrixZ raw_gradient = MixModel::MatrixZ::Constant(kNaN);
        (void)raw_value(point, &raw_gradient);
        passed &= CheckDerivative(
                label + " raw row",
                point,
                raw_gradient.transpose(),
                [&](const auto &sample) { return raw_value(sample, nullptr); },
                kExactGradient,
                perturbation,
                column_tolerances);
    }
    return passed;
}

template<unsigned M, unsigned N>
class AffineEquality final : public Constraint<double, M, N, Equality> {
public:
    OCP_VARIABLES(double, M, N)

    std::string GetName() const override { return "AffineEquality"; }

    bool Evaluate(int,
                  const State &x,
                  const Control &u,
                  double &value) const override {
        value = 0.4 * x(0) - 0.6 * x(1) + 0.7 * u(0) - 0.2;
        return true;
    }

    bool Gradient(int,
                  const State &,
                  const Control &,
                  Eigen::Ref<VecX> lx,
                  Eigen::Ref<VecU> lu) const override {
        lx.setZero();
        lu.setZero();
        lx(0) = 0.4;
        lx(1) = -0.6;
        lu(0) = 0.7;
        return true;
    }

    bool Hessian(int,
                 const State &,
                 const Control &,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override {
        lxx.setZero();
        luu.setZero();
        lxu.setZero();
        return true;
    }
};

template<unsigned M, unsigned N, typename Kind>
auto MakeConstraintValue(ConstraintPtr<double, M, N, Kind> constraint,
                         double lambda,
                         double penalty) {
    auto value =
            std::make_unique<ConstraintValue<double, M, N, Kind>>(kTerminal);
    value->LoadConstraint(std::move(constraint));
    value->SetDual(lambda);
    value->SetPenalty(penalty);
    return value;
}

template<unsigned M, unsigned N>
bool CheckRunningUnion(const CostUnion<double, M, N> &costs,
                       const typename CostUnion<double, M, N>::MatrixZ &point,
                       const std::string &name) {
    using Union = CostUnion<double, M, N>;
    const auto x = point.template head<M>();
    const auto u = point.template tail<N>();
    typename Union::MatrixZ cost_gradient = Union::MatrixZ::Constant(kNaN);
    typename Union::MatrixZ al_gradient = Union::MatrixZ::Constant(kNaN);
    // Refine any cold projection before comparing the local derivatives.
    typename Union::State neighbor = x;
    neighbor(kXPos) += 0.02;
    (void)costs.Evaluate(kRunning, neighbor, u);
    // CalcAugLagConstraintGradient uses the residual cached by the preceding Evaluate.
    (void)costs.Evaluate(kRunning, x, u);
    costs.CalcCostGradient(kRunning,
                           x,
                           u,
                           cost_gradient.template head<M>(),
                           cost_gradient.template tail<N>());
    costs.CalcAugLagConstraintGradient(kRunning,
                                       x,
                                       u,
                                       al_gradient.template head<M>(),
                                       al_gradient.template tail<N>());

    bool passed = CheckDerivative(
            name + " costs",
            point,
            cost_gradient.transpose(),
            [&](const auto &sample) {
                double total = 0.0;
                for (const auto &term : costs.GetCostTerms()) {
                    total += EvaluateValue(*term, kRunning, sample)(0);
                }
                return Scalar(total);
            },
            kAggregateGradient);
    passed &= CheckDerivative(
            name + " augmented constraints",
            point,
            al_gradient.transpose(),
            [&](const auto &sample) {
                double total = 0.0;
                const auto add_values = [&](const auto &constraints) {
                    for (const auto &constraint : constraints) {
                        if (!constraint->Evaluate(kRunning,
                                                  sample.template head<M>(),
                                                  sample.template tail<N>())) {
                            total = kNaN;
                            return;
                        }
                        total += constraint->AugLagVal();
                    }
                };
                (void)costs.Evaluate(kRunning, x, u);
                add_values(costs.GetImmutableEqConstraints());
                add_values(costs.GetImmutableIneqConstraints());
                return Scalar(total);
            },
            kAggregateGradient);
    const typename Union::MatrixZ combined = cost_gradient + al_gradient;
    passed &= CheckDerivative(
            name + " augmented objective",
            point,
            combined.transpose(),
            [&](const auto &sample) {
                (void)costs.Evaluate(kRunning, x, u);
                return Scalar(costs.Evaluate(kRunning,
                                             sample.template head<M>(),
                                             sample.template tail<N>()));
            },
            kAggregateGradient);
    return passed;
}

template<unsigned M, unsigned N>
bool CheckUnion(const CostTermConfig &config,
                const altro::SolverTermRegistry<M, N> &registry,
                const typename OCPInterface<double, M, N>::MatrixZ &inside,
                const typename OCPInterface<double, M, N>::MatrixZ &outside) {
    using Union = CostUnion<double, M, N>;
    OCPInterface<double, M, N> problem;
    LoadCosts(problem, config, registry);
    auto &costs = *problem.GetCostUnionPtr();
    for (const auto &entry : registry.constraints) {
        costs.AddIneqConstraint(
                MakeConstraintValue(entry.make(config), 0.3, 1.7));
    }
    costs.AddEqConstraint(MakeConstraintValue<M, N, Equality>(
            std::make_unique<AffineEquality<M, N>>(), -0.4, 1.9));
    auto signed_bound = MakeConstraintValue<M, N, Inequality>(
            std::make_unique<LateralOffsetConstraint<double, M, N>>(config,
                                                                    true),
            0.0,
            2.3);
    auto &bound = *signed_bound;
    costs.AddIneqConstraint(std::move(signed_bound));

    const auto name =
            "CostUnion " + std::to_string(M) + "x" + std::to_string(N);
    bool passed = true;
    const auto check_running = [&](const auto &point,
                                   double lambda,
                                   bool violated,
                                   bool active,
                                   const std::string &scenario) {
        bound.SetDual(lambda);
        if (!bound.Evaluate(kRunning,
                            point.template head<M>(),
                            point.template tail<N>()) ||
            !(violated ? bound.ConsVal() > 0.01 : bound.ConsVal() < -0.01) ||
            bound.ActivePenalty(kRunning) != (active ? 2.3 : 0.0)) {
            std::cerr << name << ' ' << scenario
                      << ": unexpected inequality branch\n";
            return false;
        }
        return CheckRunningUnion(costs, point, name + ' ' + scenario);
    };
    passed &= check_running(inside, 0.0, false, false, "inactive inequality");
    passed &= check_running(outside, 0.0, true, true, "violated inequality");
    passed &= check_running(
            inside, 0.6, false, true, "feasible inequality with positive dual");

    const typename Union::State terminal = outside.template head<M>();
    const typename Union::Control zero = Union::Control::Zero();
    typename Union::State neighbor = terminal;
    neighbor(kXPos) += 0.02;
    (void)costs.Evaluate(kTerminal, neighbor, zero);
    (void)costs.Evaluate(kTerminal, terminal, zero);
    costs.CalcTerminalCostToGo(terminal);
    const auto gradient = costs.GetFinalCostToGoGradient();
    passed &= CheckDerivative(
            name + " terminal",
            terminal,
            gradient.transpose(),
            [&](const auto &sample) {
                (void)costs.Evaluate(kTerminal, terminal, zero);
                return Scalar(costs.Evaluate(kTerminal, sample, zero));
            },
            kAggregateGradient);
    return passed;
}

bool RunGradientChecks() {
    const auto config = MakeConfig();
    const auto smooth_registry = altro::CurvatureTermRegistry();
    FrenetCoordinateSystemParameters params;
    params.Init();
    const FrenetCoordinateSystem fcs(config.ref_x, config.ref_y, params);
    const std::vector<double> limit_s{2.0, 12.0, 24.0};
    const std::vector<double> limit_v{6.0, 4.0, 4.0};
    const auto mix_registry =
            altro::MotionPlanningTermRegistry(fcs, limit_s, limit_v);
    const altro::SolverTermRegistry<4, 2> legacy_registry{
            {{"acc",
              altro::detail::AddRegisteredCost<AccCostFunc<double, 4, 2>,
                                               LegacyProblem>},
             {"offset",
              altro::detail::AddRegisteredCost<OffsetCostFunc<double, 4, 2>,
                                               LegacyProblem>},
             {"velocity",
              altro::detail::AddRegisteredCost<VelDiffCostFunc<double, 4, 2>,
                                               LegacyProblem>}},
            {}};

    // Each array orders samples by lower violation, feasible, then upper violation.
    std::array<SmoothKappaModel::MatrixZ, 3> smooth;
    smooth[0] << 5.0, 3.8, -0.8, -0.25, -0.08, -0.6;
    smooth[1] << 5.0, 6.0, 0.25, 0.2, 0.1, 0.25;
    smooth[2] << 5.0, 9.0, 0.9, 0.4, 0.13, 0.7;
    std::array<MixModel::MatrixZ, 3> mix;
    mix[0] << 5.0, 3.8, -0.8, -0.25, 1.4, -1.2, -1.4, -1.0;
    mix[1] << 5.0, 6.0, 0.25, 0.2, 3.4, 0.7, 0.3, 0.12;
    mix[2] << 5.0, 9.0, 0.9, 0.4, 5.7, 1.4, 1.5, 0.3;
    LegacyProblem::MatrixZ legacy;
    legacy << 5.0, 6.0, 3.5, 0.4, 0.6, 0.75;
    auto legacy_outside = legacy;
    legacy_outside(kYPos) = 9.0;

    bool passed = CheckPenalties();
    passed &= CheckRegisteredCosts(config, smooth_registry, smooth[1]);
    passed &= CheckRegisteredCosts(config, mix_registry, mix[1]);
    passed &= CheckUnregisteredCosts(config, smooth[1], legacy);
    // The raw rows call Evaluate and Gradient on each lower and upper primitive.
    passed &= CheckRegisteredBounds(config, smooth_registry, smooth);
    passed &= CheckRegisteredBounds(config, mix_registry, mix);
    auto by_side = mix;
    std::swap(by_side[0](mYPos), by_side[2](mYPos));
    passed &= CheckBoundPair(
            SafeDistanceToBoundaryConstraint(config), by_side, false);
    passed &= CheckBoundaryMargins<5, 1>(config, smooth[2]);
    passed &= CheckBoundaryMargins<6, 2>(config, mix[2]);

    const auto speed_factory = mix_registry.FindConstraint("speed_limit");
    if (!speed_factory) {
        std::cerr
                << "speed_limit is missing from the motion-planning registry\n";
        return false;
    }
    const auto speed = (*speed_factory)(config);
    const auto &speed_limit =
            static_cast<const SpeedLimitBySConstraint &>(*speed);
    passed &= CheckSpeedLimit(
            speed_limit, mix[1], "SpeedLimitByS sloped segment");
    auto flat = mix[1];
    flat(mXPos) = 9.1;
    flat(mYPos) = 11.7;
    passed &= CheckSpeedLimit(speed_limit, flat, "SpeedLimitByS flat segment");

    passed &= CheckUnion(config, smooth_registry, smooth[1], smooth[2]);
    passed &= CheckUnion(config, mix_registry, mix[1], mix[2]);
    passed &= CheckUnion(config, legacy_registry, legacy, legacy_outside);
    return passed;
}

bool CheckYawRateCost() {
    LegacyProblem::MatrixZ point;
    point << 5.0, 6.0, 3.5, 0.4, 0.6, 0.75;
    return CheckRunningAndTerminal(YawRateCostFunc<double, 4, 2>(MakeConfig()),
                                   point);
}

bool CheckCurvedSpeedLimit() {
    const auto config = MakeConfig();
    FrenetCoordinateSystemParameters params;
    params.Init();
    const FrenetCoordinateSystem fcs(
            config.ref_x,
            std::vector<double>{0.0, 0.2, 0.8, 1.8, 3.2, 5.0},
            params);
    constexpr double s = 7.4;
    const auto reference = fcs.GetRefCurvePoint(s);
    const auto tangent = fcs.GetRefCurveUnitTangent(s);
    MixModel::MatrixZ point;
    point << reference.x - 0.5 * tangent.y, reference.y + 0.5 * tangent.x, 0.25,
            0.2, 3.4, 0.7, 0.3, 0.12;
    const SpeedLimitBySConstraint limit(config, &fcs, {2.0, 12.0}, {6.0, 4.0});
    // The point is a normal offset from s=7.4, inside the sloped speed segment.
    constexpr double kLimitSlope = -0.2;
    const double expected_residual =
            point(mVel) - (6.0 + kLimitSlope * (s - 2.0));
    // The global search stops on a damped parameter step and returns the previous
    // iterate. On this regular fixture, allow three undamped step tolerances.
    const double cold_residual_tolerance = 3.0 * std::abs(kLimitSlope) *
                                           params.coord_transform_precision /
                                           params.optimization_gamma;
    SpeedLimitTolerances tolerances;
    tolerances.fill(kExactGradient);
    tolerances[mXPos] = kCurvedPositionGradient;
    tolerances[mYPos] = kCurvedPositionGradient;
    bool passed = CheckSpeedLimit(limit,
                                  point,
                                  "SpeedLimitByS curved offset (h=1e-5)",
                                  kPerturbation,
                                  &tolerances,
                                  expected_residual,
                                  cold_residual_tolerance);
    passed &= CheckSpeedLimit(limit,
                              point,
                              "SpeedLimitByS curved offset (h=5e-6)",
                              5e-6,
                              &tolerances,
                              expected_residual,
                              cold_residual_tolerance);

    // A straight route gives an independent endpoint distance and direction.
    const FrenetCoordinateSystem straight(
            config.ref_x,
            std::vector<double>(config.ref_x.size(), 0.0),
            params);
    const double end_s = straight.GetSlength();
    const SpeedLimitBySConstraint endpoint_limit(
            config, &straight, {0.0, end_s}, {6.0, 6.0 + kLimitSlope * end_s});
    for (double endpoint : {0.0, end_s}) {
        const double outward = endpoint == 0.0 ? -1.0 : 1.0;
        point(mYPos) = 0.0;
        for (int step : {kRunning, kTerminal}) {
            for (double distance : {0.3, 1.2}) {
                auto cold = endpoint_limit;
                auto warm = endpoint_limit;
                point(mXPos) = endpoint - 0.2 * outward;
                const double seed = EvaluateValue(warm, step, point)(0);
                point(mXPos) = endpoint + distance * outward;
                const double cold_value = EvaluateValue(cold, step, point)(0);
                const double warm_value = EvaluateValue(warm, step, point)(0);
                const double expected =
                        point(mVel) - (6.0 + kLimitSlope * endpoint);
                const bool accepted = distance < 1.0;
                const bool matches =
                        accepted ? std::abs(cold_value - expected) <=
                                                   cold_residual_tolerance &&
                                           std::abs(warm_value - expected) <=
                                                   cold_residual_tolerance
                                 : std::isnan(cold_value) &&
                                           std::isnan(warm_value);
                if (!std::isfinite(seed) || !matches) {
                    std::cerr << "SpeedLimitByS endpoint=" << endpoint
                              << ", step=" << step << ", distance=" << distance
                              << ", cold=" << cold_value
                              << ", warm=" << warm_value
                              << ", expected acceptance=" << accepted << '\n';
                    passed = false;
                }
            }
        }
    }
    return passed;
}

} // namespace

int main(int argc, char **argv) {
    if (argc == 2) {
        const std::string test = argv[1];
        if (test == "yaw-rate") {
            return CheckYawRateCost() ? EXIT_SUCCESS : EXIT_FAILURE;
        }
        if (test == "curved-speed-limit") {
            return CheckCurvedSpeedLimit() ? EXIT_SUCCESS : EXIT_FAILURE;
        }
        return EXIT_FAILURE;
    }
    const SmoothKappaModel smooth_model{};
    SmoothKappaModel::State smooth_x;
    smooth_x << 1.0, -0.5, 0.6, 0.2, 0.1;
    const SmoothKappaModel::Control smooth_u =
            SmoothKappaModel::Control::Constant(0.3);

    const MixModel mix_model{};
    MixModel::State mix_x;
    mix_x << 1.0, -0.5, 0.6, 0.2, 3.4, 0.7;
    MixModel::Control mix_u;
    mix_u << 0.3, 0.15;

    bool passed = CheckDynamics(
            "SmoothKappaModel", smooth_model, smooth_x, smooth_u, 0.4);
    passed &= CheckDynamicsAtHeadingWrap("SmoothKappaModel heading wrap",
                                         smooth_model,
                                         smooth_x,
                                         smooth_u,
                                         0.4,
                                         kTheta);
    passed &= CheckDynamics("MixModel", mix_model, mix_x, mix_u, 0.4);
    passed &= CheckDynamicsAtHeadingWrap(
            "MixModel heading wrap", mix_model, mix_x, mix_u, 0.4, mTheta);
    passed &= CheckCentripetalJerkCost(mix_x, mix_u);
    passed &= RunGradientChecks();
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
