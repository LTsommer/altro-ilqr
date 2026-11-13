#pragma once

#include "augmented_ilqr/solver_config.hpp"

namespace altro {

struct SolverCliOptions {
    std::string output_dir = ".";
    SolverBuildOptions solver;
};

std::optional<SolverCliOptions> ParseSolverCli(int argc,
                                               char **argv,
                                               const char *default_config);

} // namespace altro
