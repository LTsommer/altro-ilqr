#include "augmented_ilqr/solver_config.hpp"
#include "constraint_definition/constraint_values.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iostream>
#include <utility>

namespace altro {
namespace {

std::string_view Trim(std::string_view value) {
    const auto first = value.find_first_not_of(" \t\r");
    if (first == std::string_view::npos) {
        return {};
    }
    return value.substr(first, value.find_last_not_of(" \t\r") - first + 1);
}

} // namespace

bool SolverConfig::Error(const std::string &message) const {
    std::cerr << "Solver config " << path_ << ": " << message << '\n';
    return false;
}

std::optional<SolverSettings> SolverConfig::Load(CostTermConfig config,
                                                 ProblemKind kind) {
    if (!ReadFile()) {
        return std::nullopt;
    }
    auto costs = TakeTermList("costs");
    if (!costs) {
        return std::nullopt;
    }
    auto constraints = TakeTermList("constraints");
    if (!constraints || !ApplyTuning(config, kind)) {
        return std::nullopt;
    }
    const auto projection = TakeProjectionMethod();
    if (!projection) {
        return std::nullopt;
    }

    SolverSettings settings{
            std::move(config), *projection, std::move(*costs), {}};
    settings.constraints.reserve(constraints->size());
    const double max_penalty =
            kind == ProblemKind::Curvature
                    ? ConstraintValue<double, 5, 1, Inequality>::kMaxPenalty
                    : ConstraintValue<double, 6, 2, Inequality>::kMaxPenalty;
    for (auto &name : *constraints) {
        const std::string prefix = "constraint." + name;
        double dual;
        double penalty;
        if (!TakeNumber(prefix + ".dual", dual, 0.0) ||
            !TakeNumber(
                    prefix + ".penalty", penalty, 0.0, max_penalty, false)) {
            return std::nullopt;
        }
        settings.constraints.push_back({std::move(name), dual, penalty});
    }
    if (!CheckUnused()) {
        return std::nullopt;
    }
    return settings;
}

bool SolverConfig::ReadFile() {
    std::ifstream input(path_);
    if (!input) {
        return Error("cannot open file");
    }

    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto entry =
                Trim(std::string_view(line).substr(0, line.find('#')));
        if (entry.empty()) {
            continue;
        }
        const auto separator = entry.find('=');
        const auto key = Trim(entry.substr(0, separator));
        if (separator == std::string_view::npos || key.empty()) {
            return Error("expected key=value at line " +
                         std::to_string(line_number));
        }
        const auto value = Trim(entry.substr(separator + 1));
        if (!entries_.emplace(key, value).second) {
            return Error("duplicate key '" + std::string(key) + "' at line " +
                         std::to_string(line_number));
        }
    }
    return (!input.bad() && input.eof()) || Error("cannot read file");
}

std::optional<std::string> SolverConfig::Take(std::string_view key) {
    const auto it = entries_.find(std::string(key));
    if (it == entries_.end()) {
        Error("missing key '" + std::string(key) + "'");
        return std::nullopt;
    }
    auto value = std::move(it->second);
    entries_.erase(it);
    return value;
}

std::optional<std::vector<std::string>> SolverConfig::TakeTermList(
        std::string_view key) {
    const auto text = Take(key);
    if (!text) {
        return std::nullopt;
    }
    std::vector<std::string> names;
    std::string_view remaining = *text;
    for (;;) {
        const auto comma = remaining.find(',');
        const auto name = Trim(remaining.substr(0, comma));
        if (name.empty()) {
            Error("empty item in '" + std::string(key) + "'");
            return std::nullopt;
        }
        if (std::find(names.begin(), names.end(), name) != names.end()) {
            Error("duplicate item '" + std::string(name) + "' in '" +
                  std::string(key) + "'");
            return std::nullopt;
        }
        names.emplace_back(name);
        if (comma == std::string_view::npos) {
            return names;
        }
        remaining.remove_prefix(comma + 1);
    }
}

bool SolverConfig::TakeNumber(std::string_view key,
                              double &value,
                              double minimum,
                              double maximum,
                              bool minimum_inclusive) {
    const auto text = Take(key);
    if (!text) {
        return false;
    }

    const char *end = text->data() + text->size();
    double parsed = 0;
    const auto result = std::from_chars(text->data(), end, parsed);
    if (result.ec != std::errc{} || result.ptr != end ||
        !std::isfinite(parsed) || parsed < minimum || parsed > maximum ||
        (!minimum_inclusive && parsed == minimum)) {
        return Error("invalid number for '" + std::string(key) + "': " + *text);
    }
    value = parsed;
    return true;
}

std::optional<ProjectionMethod> SolverConfig::TakeProjectionMethod() {
    const auto name = Take("projection_method");
    if (!name) {
        return std::nullopt;
    }
    const auto method = ParseProjectionMethod(*name);
    if (!method) {
        Error("invalid projection_method '" + *name +
              "' (use riccati, kkt-eigen, or kkt-qdldl)");
    }
    return method;
}

bool SolverConfig::TakeBounds(std::string_view name,
                              double &lower,
                              double &upper,
                              double minimum_upper) {
    const std::string key = "limit." + std::string(name);
    return TakeNumber(key + "_min", lower) &&
           TakeNumber(key + "_max", upper, minimum_upper) &&
           (lower <= upper ||
            Error(key + "_min must not exceed " + key + "_max"));
}

bool SolverConfig::ApplyTuning(CostTermConfig &config, ProblemKind kind) {
    if (kind == ProblemKind::Curvature) {
        return ApplyTuning(config,
                           {{"ref_offset", WeightRefOffset},
                            {"kappa", WeightKappa},
                            {"dkappa", WeightDkappa},
                            {"u", WeightU},
                            {"clb", WeightClb},
                            {"cub", WeightCub},
                            {"plb", WeightPlb},
                            {"prb", WeightPrb},
                            {"jerk", WeightJerk},
                            {"acc", WeightAcc},
                            {"comfort_dist", WeightComfortDist}},
                           {{"yaw_diff",
                             &CostTermConfig::yaw_diff_min,
                             &CostTermConfig::yaw_diff_max},
                            {"kappa",
                             &CostTermConfig::kappa_min,
                             &CostTermConfig::kappa_max}});
    }
    // The motion planning initial guess takes the square root of lat_acc_max.
    return ApplyTuning(
            config,
            {{"ref_offset", WeightRefOffset},
             {"vel_diff", WeightVelDiff},
             {"u", WeightU},
             {"clb", WeightClb},
             {"cub", WeightCub},
             {"plb", WeightPlb},
             {"prb", WeightPrb},
             {"lat_jerk", WeightLatJerk},
             {"lat_acc", WeightLatAcc},
             {"jerk", WeightJerk},
             {"dot_kappa", WeightDotKappa},
             {"comfort_dist", WeightComfortDist}},
            {{"yaw_diff",
              &CostTermConfig::yaw_diff_min,
              &CostTermConfig::yaw_diff_max},
             {"kappa", &CostTermConfig::kappa_min, &CostTermConfig::kappa_max},
             {"acc", &CostTermConfig::acc_min, &CostTermConfig::acc_max},
             {"lat_jerk",
              &CostTermConfig::lat_jerk_min,
              &CostTermConfig::lat_jerk_max},
             {"lat_acc",
              &CostTermConfig::lat_acc_min,
              &CostTermConfig::lat_acc_max,
              0.0},
             {"jerk", &CostTermConfig::jerk_min, &CostTermConfig::jerk_max},
             {"vel", &CostTermConfig::vel_min, &CostTermConfig::vel_max}});
}

bool SolverConfig::ApplyTuning(CostTermConfig &config,
                               std::initializer_list<WeightSetting> weights,
                               std::initializer_list<LimitSetting> limits) {
    for (const auto &weight : weights) {
        if (!TakeNumber("weight." + std::string(weight.name),
                        config.weights.at(weight.index),
                        0.0)) {
            return false;
        }
    }
    for (const auto &limit : limits) {
        if (!TakeBounds(limit.name,
                        config.*limit.lower,
                        config.*limit.upper,
                        limit.minimum_upper)) {
            return false;
        }
    }
    double u_min;
    double u_max;
    if (!TakeBounds(
                "u", u_min, u_max, -std::numeric_limits<double>::infinity())) {
        return false;
    }
    config.u_min.assign(config.velocity.size(), u_min);
    config.u_max.assign(config.velocity.size(), u_max);
    return true;
}

bool SolverConfig::CheckUnused() const {
    return entries_.empty() ||
           Error("unknown key '" + entries_.begin()->first + "'");
}

} // namespace altro
