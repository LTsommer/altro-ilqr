#include "math/aabox2d.h"
#include "math/box2d.h"
#include "math/polygon2d.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>

int main() {
    const altro::AABox2d aabox({1, 2}, {7, 5});
    altro::Box2d box(aabox);
    const altro::Polygon2d polygon(box);
    const std::array<Vec2d, 4> expected{
            Vec2d{1, 2}, Vec2d{7, 2}, Vec2d{7, 5}, Vec2d{1, 5}};

    const auto corners = box.GetAllCorners();
    if (!std::is_permutation(corners.begin(),
                             corners.end(),
                             expected.begin(),
                             expected.end()) ||
        box.min_x() != 1.0 || box.max_x() != 7.0 || box.min_y() != 2.0 ||
        box.max_y() != 5.0) {
        std::cerr << "AABox2d conversion produced incorrect Box2d corners or "
                     "bounds\n";
        return EXIT_FAILURE;
    }

    const auto &vertices = polygon.points();
    if (!std::is_permutation(vertices.begin(),
                             vertices.end(),
                             expected.begin(),
                             expected.end()) ||
        polygon.area() != 18.0 || !polygon.IsPointIn({4, 3.5}) ||
        polygon.IsPointIn({8, 3.5})) {
        std::cerr << "Box2d conversion produced incorrect Polygon2d geometry\n";
        return EXIT_FAILURE;
    }
    box.Shift({10, -10});
    if (box.min_x() != 11.0 || box.max_x() != 17.0 || box.min_y() != -8.0 ||
        box.max_y() != -5.0) {
        std::cerr << "Shifted Box2d retained its old bounds\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
