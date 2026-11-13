#pragma once

#include "augmented_ilqr/polished_altro/altro_solver.hpp"
#include "augmented_ilqr/solver_config.hpp"
#include "augmented_ilqr/solver_term_registry.hpp"
#include "common/math_utils.h"
#include "frenet_coordinate/frenet_coordinate_system.h"
#include "math/vec2d.h"
#include "model/smooth_kappa_model.h"
#include "motion_planning/mix_model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace altro {

struct CurvatureStart {
    Vec2d position;
    double curvature;
    double curvature_rate;
};

struct MotionPlanningStart {
    Vec2d position;
    double curvature;
    double speed;
};

namespace detail {

template<class ConstraintFactory>
struct ResolvedConstraint {
    ConstraintFactory make;
    double dual;
    double penalty;
};

template<unsigned StateDim, unsigned ControlDim>
bool ResolveTerms(
        const SolverSettings &settings,
        const SolverTermRegistry<StateDim, ControlDim> &registry,
        const std::string &config_path,
        std::vector<typename SolverTermRegistry<StateDim, ControlDim>::AddCost>
                &costs,
        std::vector<ResolvedConstraint<
                typename SolverTermRegistry<StateDim,
                                            ControlDim>::ConstraintFactory>>
                &constraints) {
    for (const std::string &name : settings.costs) {
        const auto add_cost = registry.FindCost(name);
        if (!add_cost) {
            std::cerr << "Solver config " << config_path
                      << ": unsupported item '" << name << "' in 'costs'\n";
            return false;
        }
        costs.push_back(add_cost);
    }
    for (const SolverSettings::Constraint &term : settings.constraints) {
        const auto *make = registry.FindConstraint(term.name);
        if (!make) {
            std::cerr << "Solver config " << config_path
                      << ": unsupported item '" << term.name
                      << "' in 'constraints'\n";
            return false;
        }
        constraints.push_back({*make, term.dual, term.penalty});
    }
    return true;
}

// Conservative gains avoid oscillation in the initial rollout.
// Curvature per metre of offset (1/m^2).
constexpr double kInitLatGain = 0.02;
// Curvature per radian of heading error (1/m).
constexpr double kInitHeadingGain = 0.3;

inline bool InitializeTrajectory(const CurvatureStart &start,
                                 const FrenetCoordinateSystem &fcs,
                                 const CostTermConfig &config,
                                 const SmoothKappaModel &model,
                                 SmoothKappaModel::State &x0,
                                 SmoothKappaModel::Controls &u0) {
    const auto &ds = config.steps;
    x0(kXPos) = start.position.x();
    x0(kYPos) = start.position.y();
    x0(kTheta) = config.yaw.front();
    x0(kKappa) = start.curvature;
    x0(kDkappa) = start.curvature_rate;
    // Initial guess: instead of all-zero controls (whose rollout keeps the initial curvature and
    // quickly leaves the road), roll the model forward with a simple curvature-tracking law so the
    // starting trajectory already follows the reference line:
    //     kappa_cmd = kappa_ref(s + ds) - k_lat * l - k_heading * (theta - theta_ref)
    // where (s, l) is the Frenet projection of the current position. kappa is steered towards
    // kappa_cmd over two steps through dkappa and the control ddkappa.
    const double k_lat = kInitLatGain;
    const double k_heading = kInitHeadingGain;
    SmoothKappaModel::State x = model.PrepareInitialState(x0);
    for (int t = 0; t < static_cast<int>(ds.size()); ++t) {
        Vec2d sl;
        (void)fcs.CartCoord2FrenetCoord(Vec2d(x(kXPos), x(kYPos)), sl);
        const double s_next = std::min(sl.x() + ds[t], fcs.GetSlength());
        const double heading_err =
                AngleDifference(x(kTheta), fcs.GetRefCurveHeading(sl.x()));
        const double kappa_cmd = fcs.GetRefCurveCurvature(s_next) -
                                 k_lat * sl.y() - k_heading * heading_err;
        const double dkappa_des = (kappa_cmd - x(kKappa)) / (2.0 * ds[t]);
        u0[t](kDDkappa) = std::min(
                config.u_max[t],
                std::max(config.u_min[t], (dkappa_des - x(kDkappa)) / ds[t]));
        x = model.ForwardCalculation(x, u0[t], ds[t]);
        if (!x.allFinite()) {
            std::cerr << "Invalid curvature-tracking initial guess at step "
                      << t << '\n';
            return false;
        }
    }
    return true;
}

inline bool InitializeTrajectory(const MotionPlanningStart &start,
                                 const FrenetCoordinateSystem &fcs,
                                 const CostTermConfig &config,
                                 const MixModel &model,
                                 MixModel::State &x0,
                                 MixModel::Controls &u0) {
    const auto &dt = config.steps;
    x0(mXPos) = start.position.x();
    x0(mYPos) = start.position.y();
    x0(mTheta) = config.yaw.front();
    x0(mKappa) = start.curvature;
    x0(mVel) = start.speed;
    x0(mAcc) = config.acceleration.front();
    // Initial guess: instead of all-zero controls (whose rollout keeps the initial curvature and
    // acceleration and quickly leaves the road), roll the model forward with simple tracking laws:
    //  - curvature: kappa_cmd = kappa_ref(s + v dt) - k_lat * l - k_heading * (theta - theta_ref),
    //    clamped to the curvature limits, reached in one step through kappa_dot;
    //  - speed: follow the reference speed, capped by what the lateral-acceleration limit allows
    //    on the upcoming curvature, through an acceleration command and the jerk control,
    //    both clamped to their limits.
    const double k_lat = kInitLatGain;
    const double k_heading = kInitHeadingGain;
    constexpr double k_speed = 1.0; // 1/s
    auto clamp = [](double v, double lo, double hi) {
        return std::min(hi, std::max(lo, v));
    };
    MixModel::State x = model.PrepareInitialState(x0);
    for (int t = 0; t < static_cast<int>(dt.size()); ++t) {
        Vec2d sl;
        (void)fcs.CartCoord2FrenetCoord(Vec2d(x(mXPos), x(mYPos)), sl);
        const double s_next = std::min(sl.x() + std::max(x(mVel), 0.0) * dt[t],
                                       fcs.GetSlength());
        const double kappa_ref = fcs.GetRefCurveCurvature(s_next);
        const double heading_err =
                AngleDifference(x(mTheta), fcs.GetRefCurveHeading(sl.x()));
        const double kappa_cmd =
                clamp(kappa_ref - k_lat * sl.y() - k_heading * heading_err,
                      config.kappa_min,
                      config.kappa_max);
        double v_ref = config.velocity[std::min(
                t + 1, static_cast<int>(config.velocity.size()) - 1)];
        v_ref = std::min(v_ref,
                         std::sqrt(config.lat_acc_max /
                                   std::max(std::fabs(kappa_ref), 1e-6)));
        const double acc_cmd = clamp(
                k_speed * (v_ref - x(mVel)), config.acc_min, config.acc_max);
        u0[t](mDotKappa) = (kappa_cmd - x(mKappa)) / dt[t];
        u0[t](mJerk) = clamp(
                (acc_cmd - x(mAcc)) / dt[t], config.jerk_min, config.jerk_max);
        x = model.ForwardCalculation(x, u0[t], dt[t]);
        if (!x.allFinite()) {
            std::cerr << "Invalid motion-planning initial guess at step " << t
                      << '\n';
            return false;
        }
    }
    return true;
}

} // namespace detail

// Before calling BuildALTROSolver(), use BuildScene() to prepare the
// constraint boundaries, reference trajectory, starting conditions, and
// Frenet reference. For curvature smoothing, config.steps contains arc
// length increments (ds).
inline std::unique_ptr<ALTROSolver<double, 5, 1>> BuildALTROSolver(
        const SolverBuildOptions &options,
        const CostTermConfig &scene_config,
        const CurvatureStart &start,
        const FrenetCoordinateSystem &fcs,
        SmoothKappaModel::State &x0,
        SmoothKappaModel::Controls &u0) {
    // 1. Get the term registry for this model.
    const auto registry = CurvatureTermRegistry();
    using Registry = SolverTermRegistry<5, 1>;
    using Problem = Registry::Problem;

    // 2. Load and validate the ordered term lists and tuning.
    using Solver = ALTROSolver<double, 5, 1>;
    using ConstraintValues = ConstraintValue<double, 5, 1, Inequality>;
    assert(!scene_config.steps.empty());
    SolverConfig file(options.config_path);
    const std::optional<SolverSettings> settings =
            file.Load(scene_config, ProblemKind::Curvature);
    if (!settings) {
        return nullptr;
    }
    std::vector<Registry::AddCost> costs;
    std::vector<detail::ResolvedConstraint<Registry::ConstraintFactory>>
            constraints;
    if (!detail::ResolveTerms(
                *settings, registry, options.config_path, costs, constraints)) {
        return nullptr;
    }
    const auto &config = settings->tuning;
    const auto &steps = config.steps;

    // 3. Build the OCP, create the registered terms, and set duals and penalties.
    auto problem = std::make_unique<Problem>();
    problem->SetConfig(config);
    problem->SetCostUnion(std::make_unique<CostUnion<double, 5, 1>>(config));
    for (const Registry::AddCost add_cost : costs) {
        add_cost(*problem);
    }
    for (const auto &term : constraints) {
        auto values = std::make_unique<ConstraintValues>(steps.size());
        values->LoadConstraint(term.make(config));
        values->SetDual(term.dual);
        values->SetPenalty(term.penalty);
        problem->AddIneqConstraint(std::move(values));
    }

    auto model = std::make_unique<SmoothKappaModel>();
    const SmoothKappaModel &initial_model = *model;
    problem->SetModel(std::move(model));
    std::cout << "all cost functions and forward model are loaded\n";

    // 4. Initialize x0 and u0 with the same tuned configuration as the OCP.
    Solver::State initial_state;
    Solver::Controls initial_controls(steps.size(), Solver::Control::Zero());
    if (!detail::InitializeTrajectory(start,
                                      fcs,
                                      config,
                                      initial_model,
                                      initial_state,
                                      initial_controls)) {
        return nullptr;
    }
    assert(initial_controls.size() == steps.size());

    // 5. Construct the solver and apply the selected projection method.
    auto solver = std::make_unique<Solver>(std::move(problem), steps);
    solver->SetPrintSummary(options.print_summary);
    solver->SetUpdateConfig(options.update_config);
    solver->SetProjectionMethod(
            options.projection_override.value_or(settings->projection_method));

    x0 = std::move(initial_state);
    u0 = std::move(initial_controls);
    return solver;
}

// For motion planning, BuildScene() also prepares the arc-length-indexed
// boundaries and speed limits. config.steps contains time increments (dt).
// SpeedLimitBySConstraint borrows fcs, so the scene must keep it alive through
// the caller's Solve() (step 7).
inline std::unique_ptr<ALTROSolver<double, 6, 2>> BuildALTROSolver(
        const SolverBuildOptions &options,
        const CostTermConfig &scene_config,
        const MotionPlanningStart &start,
        const FrenetCoordinateSystem &fcs,
        const std::vector<double> &speed_limit_s,
        const std::vector<double> &speed_limit_v,
        MixModel::State &x0,
        MixModel::Controls &u0) {
    // 1. Get the term registry for this model.
    const auto registry =
            MotionPlanningTermRegistry(fcs, speed_limit_s, speed_limit_v);
    using Registry = SolverTermRegistry<6, 2>;
    using Problem = Registry::Problem;

    // 2. Load and validate the ordered term lists and tuning.
    using Solver = ALTROSolver<double, 6, 2>;
    using ConstraintValues = ConstraintValue<double, 6, 2, Inequality>;
    assert(!scene_config.steps.empty());
    SolverConfig file(options.config_path);
    const std::optional<SolverSettings> settings =
            file.Load(scene_config, ProblemKind::MotionPlanning);
    if (!settings) {
        return nullptr;
    }
    std::vector<Registry::AddCost> costs;
    std::vector<detail::ResolvedConstraint<Registry::ConstraintFactory>>
            constraints;
    if (!detail::ResolveTerms(
                *settings, registry, options.config_path, costs, constraints)) {
        return nullptr;
    }
    const auto &config = settings->tuning;
    const auto &steps = config.steps;

    // 3. Build the OCP, create the registered terms, and set duals and penalties.
    auto problem = std::make_unique<Problem>();
    problem->SetConfig(config);
    problem->SetCostUnion(std::make_unique<CostUnion<double, 6, 2>>(config));
    for (const Registry::AddCost add_cost : costs) {
        add_cost(*problem);
    }
    for (const auto &term : constraints) {
        auto values = std::make_unique<ConstraintValues>(steps.size());
        values->LoadConstraint(term.make(config));
        values->SetDual(term.dual);
        values->SetPenalty(term.penalty);
        problem->AddIneqConstraint(std::move(values));
    }

    auto model = std::make_unique<MixModel>();
    const MixModel &initial_model = *model;
    problem->SetModel(std::move(model));
    const auto &cost_union = *problem->GetCostUnionPtr();
    std::cout << "Nums of Inequality Constraints: "
              << cost_union.GetImmutableIneqConstraints().size() << "\n";
    std::cout << "Nums of Equality Constraints: "
              << cost_union.GetImmutableEqConstraints().size() << "\n";
    std::cout << "all cost functions, constraints and forward "
                 "model are "
                 "loaded\n";

    // 4. Initialize x0 and u0 with the same tuned configuration as the OCP.
    Solver::State initial_state;
    Solver::Controls initial_controls(steps.size(), Solver::Control::Zero());
    if (!detail::InitializeTrajectory(start,
                                      fcs,
                                      config,
                                      initial_model,
                                      initial_state,
                                      initial_controls)) {
        return nullptr;
    }
    assert(initial_controls.size() == steps.size());

    // 5. Construct the solver and apply the selected projection method.
    auto solver = std::make_unique<Solver>(std::move(problem), steps);
    solver->SetPrintSummary(options.print_summary);
    solver->SetUpdateConfig(options.update_config);
    solver->SetProjectionMethod(
            options.projection_override.value_or(settings->projection_method));

    x0 = std::move(initial_state);
    u0 = std::move(initial_controls);
    return solver;
}

} // namespace altro
