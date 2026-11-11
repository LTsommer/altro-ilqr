//
// Created by LTsommer on 2024/5/31.
//

#ifndef CILQR_COST_TERM_CONFIG_HPP
#define CILQR_COST_TERM_CONFIG_HPP

#include "../ilqr_system_declaration.hpp"
#include <unordered_map>
#include <vector>
#include <typeindex>

typedef std::unordered_map<std::type_index, std::string> Type_names;

enum BoundType { Default, Others };

enum WeightIndex {
    WeightAcc,    // acceleration term
    WeightYawRate, // yaw rate term
    WeightRefOffset,    // reference point term
    WeightVelDiff, // velocity difference
    WeightKappa,  // Curvature
    WeightDkappa, // Curvature change rate
    WeightJerk,   // Jerk
    WeightU,      // Control input
    WeightClb,    // Control lower bound
    WeightCub,    // Control upper bound
    WeightPlb,    // Path left bound
    WeightPrb,     // Path right bound
    WeightLatAcc,
    WeightLatJerk,
    WeightDotKappa,
};

struct CostTermConfig {
//    bool use_finite_diff{true};
    std::unordered_map<WeightIndex, double, std::hash<int>> weights{
            {WeightIndex::WeightAcc, 0.0}, {WeightIndex::WeightYawRate, 0.0},
            {WeightIndex::WeightRefOffset, 0.0}, {WeightIndex::WeightVelDiff, 0.0},
            {WeightKappa, 0.0}, {WeightDkappa, 0.0}, {WeightU, 0.0},
            {WeightClb, 0.0},   {WeightCub, 0.0},    {WeightPlb, 0.0},
            {WeightPrb, 0.0}, {WeightJerk, 0.0}, {WeightLatAcc, 0.0},
            {WeightLatJerk, 0.0}, {WeightDotKappa, 0.0}};
    std::vector<double> velocity{};
    std::vector<double> acceleration{};
    std::vector<double> yaw;
    std::vector<double> s;
    std::vector<double> t;
    std::vector<Vector2d> p_ref;     // reference points: p_ref[i] = [x, y]
    std::vector<double> steps;
    std::vector<double> ref_factors; // weight discount factor for each p_ref[i]
    std::vector<double> u_min{};
    std::vector<double> u_max{};
    std::vector<std::vector<Vector2d>> p_bound; // left bound p1->p2 and right
    // bound p3->p4 of each ref point
    // p_bound[i] = {p1, p2, p3, p4}
    std::array<std::vector<BoundType>, 2> boundary_types;
    double safe_dist = 1.0;
    double final_yaw_rate;
    double final_yaw;
    double kappa_max;
    double acc_max;
    double lat_acc_max;
    double j_max;
    double yaw_diff_max;
    double lambda;
    double mu;
    double mu_factor;
    std::string name{""};
    Type_names type_names;
};

#endif //CILQR_COST_TERM_CONFIG_HPP
