#include "common/solver_cli.hpp"

#include <iostream>

namespace altro {

std::optional<SolverCliOptions> ParseSolverCli(int argc,
                                               char **argv,
                                               const char *default_config) {
    SolverCliOptions options;
    options.solver.config_path = default_config;
    int position = 0;
    bool has_config = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--summary") {
            options.solver.print_summary = true;
        } else if (arg == "--config") {
            if (has_config || i + 1 == argc ||
                std::string_view(argv[i + 1]).empty() ||
                std::string_view(argv[i + 1]).substr(0, 2) == "--") {
                std::cerr << "--config requires exactly one file path\n";
                return std::nullopt;
            }
            options.solver.config_path = argv[++i];
            has_config = true;
        } else if (arg.substr(0, 2) == "--" || position == 2) {
            std::cerr << "Unexpected argument: " << arg << '\n';
            return std::nullopt;
        } else if (position++ == 0) {
            options.output_dir = arg;
        } else {
            options.solver.projection_override = ParseProjectionMethod(arg);
            if (!options.solver.projection_override) {
                std::cerr << "Unknown projection method: " << arg
                          << " (use riccati, kkt-eigen, or kkt-qdldl)\n";
                return std::nullopt;
            }
        }
    }
    return options;
}

} // namespace altro
