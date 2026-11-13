#pragma once

#include "augmented_ilqr/polished_altro/projection_solver.h"
#include "cost_function/cost_term_config.hpp"

#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace altro {

struct SolverBuildOptions {
    std::string config_path;
    std::optional<ProjectionMethod> projection_override;
    bool print_summary = false;
    bool update_config = false;
};

enum class ProblemKind {
    Curvature,
    MotionPlanning
};

struct SolverSettings {
    struct Constraint {
        std::string name;
        double dual;
        double penalty;
    };

    CostTermConfig tuning;
    ProjectionMethod projection_method;
    std::vector<std::string> costs;
    std::vector<Constraint> constraints;
};

// Each required key is consumed once. Leftover keys indicate a typo or a
// configuration for a different problem.
class SolverConfig {
public:
    explicit SolverConfig(const std::string &path) : path_(path) {}

    std::optional<SolverSettings> Load(CostTermConfig config, ProblemKind kind);

private:
    struct WeightSetting {
        std::string_view name;
        WeightIndex index;
    };

    struct LimitSetting {
        std::string_view name;
        double CostTermConfig::*lower;
        double CostTermConfig::*upper;
        double minimum_upper = -std::numeric_limits<double>::infinity();
    };

    bool ReadFile();
    std::optional<std::vector<std::string>> TakeTermList(std::string_view key);
    bool TakeNumber(std::string_view key,
                    double &value,
                    double minimum = -std::numeric_limits<double>::infinity(),
                    double maximum = std::numeric_limits<double>::infinity(),
                    bool minimum_inclusive = true);
    std::optional<ProjectionMethod> TakeProjectionMethod();
    bool ApplyTuning(CostTermConfig &config, ProblemKind kind);
    bool ApplyTuning(CostTermConfig &config,
                     std::initializer_list<WeightSetting> weights,
                     std::initializer_list<LimitSetting> limits);
    bool CheckUnused() const;
    std::optional<std::string> Take(std::string_view key);
    bool Error(const std::string &message) const;
    bool TakeBounds(std::string_view name,
                    double &lower,
                    double &upper,
                    double minimum_upper);

    std::string path_;
    std::unordered_map<std::string, std::string> entries_;
};

} // namespace altro
