#include "common/path_validation.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

int main() {
    const altro::PathSceneGeometry scene{{{0, 4}, {20, 4}},
                                         {{0, -4}, {20, -4}},
                                         {{{9, 1}, {9, -1}, {11, -1}, {11, 1}}},
                                         1.0};

    const double nan = std::numeric_limits<double>::quiet_NaN();
    struct TestCase {
        const char *name;
        std::vector<Vec2d> path;
        const char *expected_failure;
    };
    const std::vector<TestCase> cases{
            {"clear", {{2, 2.5}, {18, 2.5}}, ""},
            {"crosses between nodes", {{2, 0}, {18, 0}}, "obstacle 1"},
            {"entirely inside obstacle", {{9.5, 0}, {10.5, 0}}, "obstacle 1"},
            {"sweep touches obstacle", {{2, 2}, {18, 2}}, "obstacle 1"},
            {"sweep touches road", {{2, 3}, {18, 3}}, "left road boundary"},
            {"entirely outside road",
             {{2, -6}, {18, -6}},
             "outside road corridor or at a road endpoint"},
            {"past road endpoint",
             {{18, 0}, {21, 0}},
             "outside road corridor or at a road endpoint"},
            {"stationary clear path", {{3, 0}, {3, 0}}, ""},
            {"stationary in obstacle", {{10, 0}, {10, 0}}, "obstacle 1"},
            {"non-finite point", {{2, 0}, {nan, 0}}, "non-finite path point"},
            {"empty path", {}, "path needs at least two points"}};

    for (const auto &test : cases) {
        const auto failure = altro::ValidatePath(scene, test.path);
        const std::string result = failure ? failure->reason : "";
        if (result != test.expected_failure) {
            std::cerr << test.name << ": expected '" << test.expected_failure
                      << "', got '" << result << "'\n";
            return EXIT_FAILURE;
        }
    }

    const auto failure = altro::ValidatePath(scene, {{2, 0}, {6, 0}, {18, 0}});
    if (!failure || failure->segment != 1 || failure->reason != "obstacle 1") {
        std::cerr << "Expected the second segment to hit obstacle 1\n";
        return EXIT_FAILURE;
    }

    const altro::PathSceneGeometry small_scene{{{-1, 1}, {1, 1}},
                                               {{-1, -1}, {1, -1}},
                                               {{{0.0003, -0.001},
                                                 {0.0006, -0.001},
                                                 {0.0006, 0.001},
                                                 {0.0003, 0.001}}},
                                               0.0001};
    const auto short_crossing =
            altro::ValidatePath(small_scene, {{-0.001, 0}, {0.001, 0}});
    if (!short_crossing || short_crossing->segment != 0 ||
        short_crossing->reason != "obstacle 1") {
        std::cerr << "Expected the short path segment to cross obstacle 1\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
