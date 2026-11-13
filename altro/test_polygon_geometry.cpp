#include "math/polygon2d.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
    const altro::Polygon2d square({{0, 0}, {2, 0}, {2, 2}, {0, 2}});
    const Vec2d contact{0, 1};
    const Vec2d outside{-1, 1};
    bool passed = true;
    for (const auto &segment : {altro::LineSegment2d(contact, outside),
                                altro::LineSegment2d(outside, contact)}) {
        Vec2d first{-10, -10};
        Vec2d last{-10, -10};
        if (!square.GetOverlap(segment, &first, &last) || !(first == contact) ||
            !(last == contact)) {
            std::cerr
                    << "Endpoint contact must produce both overlap endpoints\n";
            passed = false;
            break;
        }
    }

    const Vec2d a{0, 2};
    const Vec2d b{0.75 * kMathEpsilon, 1};
    const Vec2d c{1.5 * kMathEpsilon, 0};
    const Vec2d d{3, 0};
    const Vec2d e{3, 2};
    const std::array<Vec2d, 5> input{a, b, c, d, e};
    const std::array<Vec2d, 4> expected{a, c, d, e};
    std::array<int, 5> order{0, 1, 2, 3, 4};
    do {
        std::vector<Vec2d> points;
        for (const int index : order) {
            points.push_back(input[index]);
        }
        altro::Polygon2d hull;
        if (!altro::Polygon2d::ComputeConvexHull(points, &hull) ||
            !std::is_permutation(hull.points().begin(),
                                 hull.points().end(),
                                 expected.begin(),
                                 expected.end())) {
            std::cerr << "Convex hull depends on the order of near-aligned "
                         "points\n";
            passed = false;
            break;
        }
    } while (std::next_permutation(order.begin(), order.end()));
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
