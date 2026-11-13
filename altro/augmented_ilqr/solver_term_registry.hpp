#pragma once

#include "augmented_ilqr/ocp_problem.hpp"
#include "basic_constraint/basic_constraint.hpp"
#include "cost_function/cost_terms.h"
#include "motion_planning/constraints/constraints.hpp"
#include "motion_planning/cost_functions/cost_terms.hpp"

#include <algorithm>
#include <functional>
#include <memory>
#include <string_view>
#include <vector>

namespace altro {

// The registry maps file names to constructors. The config file selects
// which entries to load and in what order.
template<unsigned StateDim, unsigned ControlDim>
struct SolverTermRegistry {
    using Problem = OCPInterface<double, StateDim, ControlDim>;
    using AddCost = void (*)(Problem &);
    using ConstraintValues =
            ConstraintValue<double, StateDim, ControlDim, Inequality>;
    using ConstraintFactory = std::function<
            ConstraintPtr<double, StateDim, ControlDim, Inequality>(
                    const CostTermConfig &)>;

    struct CostEntry {
        std::string_view name;
        AddCost add;
    };
    struct ConstraintEntry {
        std::string_view name;
        ConstraintFactory make;
    };

    std::vector<CostEntry> costs;
    std::vector<ConstraintEntry> constraints;

    AddCost FindCost(std::string_view name) const {
        const auto it = std::find_if(
                costs.begin(), costs.end(), [name](const CostEntry &entry) {
                    return entry.name == name;
                });
        return it == costs.end() ? nullptr : it->add;
    }

    const ConstraintFactory *FindConstraint(std::string_view name) const {
        const auto it = std::find_if(constraints.begin(),
                                     constraints.end(),
                                     [name](const ConstraintEntry &entry) {
                                         return entry.name == name;
                                     });
        return it == constraints.end() ? nullptr : &it->make;
    }
};

namespace detail {

template<class CostType, class Problem>
void AddRegisteredCost(Problem &problem) {
    problem.template AddCostFunc<CostType>();
}

template<class ConstraintType>
auto MakeConstraint(const CostTermConfig &config) {
    return std::make_unique<ConstraintType>(config);
}

} // namespace detail

inline SolverTermRegistry<5, 1> CurvatureTermRegistry() {
    using Problem = SolverTermRegistry<5, 1>::Problem;
    return {
            {
                    {"offset",
                     detail::AddRegisteredCost<OffsetCostFunc<double, 5, 1>,
                                               Problem>},
                    {"curvature",
                     detail::AddRegisteredCost<CurvatureCostFunc<double, 5, 1>,
                                               Problem>},
                    {"lateral_comfort",
                     detail::AddRegisteredCost<
                             LateralComfortCostFunc<double, 5, 1>,
                             Problem>},
                    {"comfort_dist",
                     detail::AddRegisteredCost<
                             ComfortDistanceCostFunc<double, 5, 1>,
                             Problem>},
            },
            {
                    {"safe",
                     detail::MakeConstraint<SafeConstraint<double, 5, 1>>},
                    {"control",
                     detail::MakeConstraint<ControlConstraint<double, 5, 1>>},
                    {"heading",
                     detail::MakeConstraint<HeadingConstraint<double, 5, 1>>},
                    {"kappa",
                     detail::MakeConstraint<KappaConstraint<double, 5, 1>>},
            },
    };
}

// Scene data must outlive factory use. SpeedLimitBySConstraint copies the
// limit samples and borrows the Frenet reference through Solve().
inline SolverTermRegistry<6, 2> MotionPlanningTermRegistry(
        const FrenetCoordinateSystem &fcs,
        const std::vector<double> &speed_limit_s,
        const std::vector<double> &speed_limit_v) {
    using Problem = SolverTermRegistry<6, 2>::Problem;
    return {
            {
                    {"lat_acc",
                     detail::AddRegisteredCost<CentripetalAccCostFunc,
                                               Problem>},
                    {"lat_jerk",
                     detail::AddRegisteredCost<CentripetalJerkCostFunc,
                                               Problem>},
                    {"dot_kappa",
                     detail::AddRegisteredCost<DotCurveCostFunc, Problem>},
                    {"jerk",
                     detail::AddRegisteredCost<LinearJerkCostFunc, Problem>},
                    {"offset",
                     detail::AddRegisteredCost<OffsetCostFunc<double, 6, 2>,
                                               Problem>},
                    {"target_speed",
                     detail::AddRegisteredCost<TargetSpeedCostFunc, Problem>},
                    {"comfort_dist",
                     detail::AddRegisteredCost<
                             ComfortDistanceCostFunc<double, 6, 2>,
                             Problem>},
            },
            {
                    {"acc", detail::MakeConstraint<AccConstraint>},
                    {"jerk", detail::MakeConstraint<JerkConstraint>},
                    {"curve", detail::MakeConstraint<CurveConstraint>},
                    {"lat_acc",
                     detail::MakeConstraint<CentripetalAccConstraint>},
                    {"lat_jerk",
                     detail::MakeConstraint<CentripetalJerkConstraint>},
                    {"safe",
                     detail::MakeConstraint<SafeConstraint<double, 6, 2>>},
                    {"heading", detail::MakeConstraint<HeadingDiffConstraint>},
                    {"vel", detail::MakeConstraint<VelConstraint>},
                    {"speed_limit",
                     [&fcs, &speed_limit_s, &speed_limit_v](
                             const CostTermConfig &config) {
                         return std::make_unique<SpeedLimitBySConstraint>(
                                 config, &fcs, speed_limit_s, speed_limit_v);
                     }},
            },
    };
}

} // namespace altro
