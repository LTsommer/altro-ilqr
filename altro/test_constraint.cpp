//
// Created by LTsommer on 2024/7/7.
//
#include <iostream>
#include "model/smooth_kappa_model.h"
#include "ilqr_system_declaration.hpp"
#include "cost_function/cost_function.hpp"
#include "cost_function/cost_terms.h"
#include "frenet_coordinate/vec2d.h"
#include "frenet_coordinate/frenet_coordinate_system.h"
#include "augmented_ilqr/ocp_problem.hpp"
#include "augmented_ilqr/alilqr_solver.h"
#include "basic_constraint/basic_constraint.hpp"
#include "augmented_ilqr/polished_altro/altro_solver.hpp"
#include "clock.hpp"
#include <iostream>
#include <typeindex>
#include <unordered_map>
#include <string>
#include <memory>
#include <cmath>

using std::cout;
using std::endl;

void vector_norm(vector<double> &rx) {
    auto iter = std::min_element(std::begin(rx), std::end(rx));
    int index = std::distance(std::begin(rx), iter);
    auto min_ele = rx[index];
    std::for_each(std::begin(rx), std::end(rx), [&min_ele](double &val) { val -= min_ele; });
    return;
}

void test_calculus() {
    OCP_VARIABLES(double, 6, 2)
    double lb = 1.2;
    double ub = 2.0;
    vector<double> x{-0.9739065285171717,
                     -0.8650633666889845,
                     -0.6794095682990244,
                     -0.4333953941292472,
                     -0.1488743389816312,
                     0.1488743389816312,
                     0.4333953941292472,
                     0.6794095682990244,
                     0.8650633666889845,
                     0.9739065285171717};
    double dt = 0.1;
    double d_rate = (ub - lb) / dt;
    double coef_m = dt / 2.0;
    double coef_p = dt / 2.0;
    Eigen::Map<Matrix<double, 10, 1>> xi(x.data());
    Eigen::Matrix<double, 10, 1> Coef_p;
    Coef_p.setConstant(coef_p);
    Eigen::Matrix<double, 10, 1> T = coef_m * xi + Coef_p;
    Eigen::Matrix<double, 10, 1> LB;
    LB.setConstant(lb);
    Eigen::Matrix<double, 10, 1> UB = LB + T * d_rate;
    cout << UB << endl;
}

void test_model() {}

int main() {
    test_calculus();
    bool visualize = false;
    vector<double> centerline_x{
            458826.4544207164,  458828.53126436437, 458830.61546950776, 458832.7064482603,  458834.79742701043,
            458836.88840576046, 458838.97938451247, 458841.07036325993, 458843.16134201374, 458845.252320763,
            458847.34329951426, 458849.4342782653,  458851.5236426195,  458853.6099811353,  458855.28036619513,
            458856.3180312012,  458857.35569667764, 458858.39336168376, 458859.43102715584, 458860.39632774034,
            458862.47631880885, 458864.5563098777,  458866.63630094484, 458868.7157781697,  458870.7828980542,
            458872.85001793865, 458873.8835776474,  458874.91713782307, 458875.9506975317,  458876.9842577073,
            458877.98604426556, 458878.98783128127, 458879.5912861882,  458880.1947412999,  458880.2913817152,
            458880.38802213175, 458880.4846625475,  458880.5813029623,  458880.6779433763,  458880.77304845705,
            458880.82541630074, 458880.8179323919,  458880.7511199279,  458880.6246768579,  458880.43791652855,
            458880.1906926864,  458879.88172611984, 458879.51037498855, 458879.07589149766, 458878.5254063772,
            458877.9256015499,  458877.32579672226, 458876.72599189205, 458875.89673135534, 458874.2032267701,
            458872.5097217082,  458870.8162171199,  458869.1227120615,  458867.4292074746,  458865.73570241313,
            458864.0421978275,  458862.34869276744, 458860.65518817876, 458858.96168312,    458855.5746734753,
            458852.1876638267,  458848.80065417854, 458845.4136445309,  458842.0266348838,  458838.63963219716,
            458836.88095205155};
    vector<double> centerline_y{
            4404758.26058061,   4404754.841996041,  4404751.427904584,  4404748.0179474205, 4404744.607990254,
            4404741.198033092,  4404737.788075925,  4404734.378118761,  4404730.968161597,  4404727.558204432,
            4404724.148247269,  4404720.738290103,  4404717.327344503,  4404713.914546299,  4404711.182157907,
            4404709.472406139,  4404707.7626545215, 4404706.052902755,  4404704.343151138,  4404702.752634045,
            4404699.335963569,  4404695.919293091,  4404692.502622611,  4404689.085640642,  4404685.661167634,
            4404682.236694625,  4404680.524458046,  4404678.8122216165, 4404677.099985038,  4404675.387748607,
            4404673.6584856855, 4404671.929222899,  4404670.649989229,  4404669.370755611,  4404668.37543625,
            4404667.380116888,  4404666.384797528,  4404665.3894781675, 4404664.394158806,  4404663.398702398,
            4404662.4002310075, 4404661.400414442,  4404660.4028074015, 4404659.41099599,   4404658.428759468,
            4404657.459983135,  4404656.509103856,  4404655.580833947,  4404654.680393556,  4404653.848514664,
            4404653.048368321,  4404652.24822198,   4404651.448075639,  4404650.341832818,  4404649.277858284,
            4404648.213884123,  4404647.14990959,   4404646.085935428,  4404645.021960894,  4404643.957986732,
            4404642.894012198,  4404641.830038036,  4404640.766063504,  4404639.702089342,  4404637.5741406465,
            4404635.446191951,  4404633.318243254,  4404631.19029456,   4404629.062345863,  4404626.934386091,
            4404625.829429328};
    vector_norm(centerline_x);
    vector_norm(centerline_y);
    vector<double> rx, ry;
    int start_node = 10;
    int end_node = 40;
    rx.assign(centerline_x.begin() + start_node, centerline_x.begin() + end_node);
    ry.assign(centerline_y.begin() + start_node, centerline_y.begin() + end_node);
    vector<double> kappa, yaw, velocity, acceleration, ds, s, dt;
    vector<Vector2d> p_ref;

    FrenetCoordinateSystemParameters fcs_params;
    fcs_params.Init();
    std::unique_ptr<FrenetCoordinateSystem> fcs(new FrenetCoordinateSystem(centerline_x, centerline_y, fcs_params));
    vector<double> left_safe_dist, right_safe_dist;
    double vel_max = 15.0;
    double vel_min = 2.0;
    double g = 9.8;
    double acc_max = 0.1 * g;
    double s_old = 0.0;
    double s_new = 0.0;
    double kappa_ = 0.0;
    double dkappa_ = 0.0;
    double lane_width = 1.8;
    double buffer_dis = 0.2;
    double safe_dist = 1.5;
    double half_veh_width = 1.0;
    double max_safe_buffer = 0.3;
    double min_safe_buffer = 0.1;
    vector<double> lb_x, lb_y, rb_x, rb_y;
    vector<vector<Vector2d>> p_bounds;
    // One entry per reference point (ds / dt / acceleration have one fewer).
    p_ref.reserve(rx.size());
    p_bounds.reserve(rx.size());
    left_safe_dist.reserve(rx.size());
    right_safe_dist.reserve(rx.size());
    yaw.reserve(rx.size());
    kappa.reserve(rx.size());
    velocity.reserve(rx.size());
    s.reserve(rx.size());
    ds.reserve(rx.size());
    dt.reserve(rx.size());
    acceleration.reserve(rx.size());
    for (int i = 0; i < rx.size(); ++i) {
        Vector2d ref;
        double x = rx[i];
        double y = ry[i];
        ref << x, y;
        p_ref.emplace_back(ref);
        Vec2d frenet_pt;
        (void)fcs->CartCoord2FrenetCoord(Vec2d(x, y), frenet_pt);
        double h = fcs->GetRefCurveHeading(frenet_pt.x());
        double k = fcs->GetRefCurveCurvature(frenet_pt.x());
        double v = std::min(vel_max, std::max(std::sqrt(acc_max / k), vel_min));
        double s_front = std::max(frenet_pt.x() - buffer_dis, 0.0);
        double s_back = std::min(frenet_pt.x() + buffer_dis, fcs->GetSlength());
        vector<Vec2d> bound_points;
        if (i >= 5 and i <= 8) {
            Vec2d left_lane_boundary_1(s_front, lane_width - 1.2);
            Vec2d left_lane_boundary_2(s_back, lane_width - 1.2);
            Vec2d right_lane_boundary_1(s_front, -lane_width);
            Vec2d right_lane_boundary_2(s_back, -lane_width);
            left_safe_dist.emplace_back(half_veh_width + max_safe_buffer);
            right_safe_dist.emplace_back(half_veh_width + min_safe_buffer);
            bound_points = vector<Vec2d>{
                    left_lane_boundary_1, left_lane_boundary_2, right_lane_boundary_1, right_lane_boundary_2};
        } else if (i >= 15 and i <= 18) {
            Vec2d left_lane_boundary_1(s_front, lane_width);
            Vec2d left_lane_boundary_2(s_back, lane_width);
            Vec2d right_lane_boundary_1(s_front, -lane_width + 0.8);
            Vec2d right_lane_boundary_2(s_back, -lane_width + 0.8);
            left_safe_dist.emplace_back(half_veh_width + min_safe_buffer);
            right_safe_dist.emplace_back(half_veh_width + max_safe_buffer);
            bound_points = vector<Vec2d>{
                    left_lane_boundary_1, left_lane_boundary_2, right_lane_boundary_1, right_lane_boundary_2};
        } else {
            Vec2d left_lane_boundary_1(s_front, lane_width);
            Vec2d left_lane_boundary_2(s_back, lane_width);
            Vec2d right_lane_boundary_1(s_front, -lane_width);
            Vec2d right_lane_boundary_2(s_back, -lane_width);
            left_safe_dist.emplace_back(half_veh_width + max_safe_buffer);
            right_safe_dist.emplace_back(half_veh_width + max_safe_buffer);
            bound_points = vector<Vec2d>{
                    left_lane_boundary_1, left_lane_boundary_2, right_lane_boundary_1, right_lane_boundary_2};
        }

        vector<Vector2d> p_bound;
        p_bound.reserve(bound_points.size());
        std::for_each(bound_points.begin(), bound_points.end(), [&p_bound, &fcs](const Vec2d &f_point) {
            Vec2d c_point;
            TRANSFORM_STATUS status = fcs->FrenetCoord2CartCoord(f_point, c_point);
            assert(status == TRANSFORM_SUCCESS);
            Vector2d p_b(c_point.x(), c_point.y());
            p_bound.emplace_back(p_b);
        });
        p_bounds.emplace_back(p_bound);
        yaw.emplace_back(h);
        kappa.emplace_back(k);
        velocity.emplace_back(v);
        s_new = frenet_pt.x();
        if (i == 0) {
            s.emplace_back(0.0);
            kappa_ = fcs->GetRefCurveCurvature(s.back());
            dkappa_ = fcs->GetRefCurveDCurvature(s.back());
        } else {
            Vec2d prev(rx[i - 1], ry[i - 1]);
            Vec2d curr(x, y);
            s.emplace_back((curr - prev).Length() + s.back());
        }
        if (i > 0) {
            double step = s_new - s_old;
            double t = (s_new - s_old) / velocity.back();
            double acc = (velocity[i] - velocity[i - 1]) / t;
            ds.emplace_back(step);
            dt.emplace_back(t);
            acceleration.emplace_back(acc);
        }
        s_old = s_new;
    }
    // The cost terms read the reference acceleration at every node. The loop above gives one value
    // per interval; the terminal node uses 0.
    acceleration.emplace_back(0.0);
    lb_x.reserve(p_bounds.size());
    lb_y.reserve(p_bounds.size());
    rb_x.reserve(p_bounds.size());
    rb_y.reserve(p_bounds.size());
    for (const auto &bound : p_bounds) {
        lb_x.emplace_back(bound.front()(0));
        lb_y.emplace_back(bound.front()(1));
        rb_x.emplace_back(bound[2](0));
        rb_y.emplace_back(bound[2](1));
    }

    double w_ref_offset = 3000.0;
    double w_kappa = 1000.0;
    double w_dkappa = 1000.0;
    double w_u = 1.0;
    double w_clb = 1.0;
    double w_cub = 1.0;
    double w_plb = 1.0;
    double w_prb = 1.0;
    double w_jerk = 5.0;
    double w_acc = 10.0;

    double u_min = -5.0;
    double u_max = 5.0;
    int size = velocity.size();
    vector<double> u_min_(size, u_min);
    vector<double> u_max_(size, u_max);

    vector<BoundType> l_bound(size, BoundType::Others);
    vector<BoundType> r_bound(size, BoundType::Others);

    TypeNames type_names;
    type_names[std::type_index(typeid(OffsetCostFunc<double, 5, 1>))] = std::string("OffsetCostFunc");
    type_names[std::type_index(typeid(CurvatureCostFunc<double, 5, 1>))] = std::string("CurvatureCostFunc");
    type_names[std::type_index(typeid(LateralComfortCostFunc<double, 5, 1>))] = std::string("LateralComfortCostFunc");
    type_names[std::type_index(typeid(SafeConstraint<double, 5, 1>))] = std::string("SafeConstraint");
    type_names[std::type_index(typeid(ControlConstraint<double, 5, 1>))] = std::string("ControlConstraint");
    type_names[std::type_index(typeid(HeadingConstraint<double, 5, 1>))] = std::string("HeadingConstraint");

    CostTermConfig config;
    config.velocity = velocity;
    config.acceleration = acceleration;
    config.yaw = yaw;
    config.final_yaw_rate = 0.0;
    config.final_yaw = yaw.back();
    config.u_min.swap(u_min_);
    config.u_max.swap(u_max_);
    config.boundary_types.at(0) = l_bound;
    config.boundary_types.at(1) = r_bound;
    config.p_ref = p_ref;
    config.weights[WeightRefOffset] = w_ref_offset;
    config.weights[WeightKappa] = w_kappa;
    config.weights[WeightDkappa] = w_dkappa;
    config.weights[WeightU] = w_u;
    config.weights[WeightClb] = w_clb;
    config.weights[WeightCub] = w_cub;
    config.weights[WeightPlb] = w_plb;
    config.weights[WeightPrb] = w_prb;
    config.weights[WeightJerk] = w_jerk;
    config.weights[WeightAcc] = w_acc;
    config.steps = ds;
    config.p_bound = p_bounds;
    config.left_safe_dist = left_safe_dist;
    config.right_safe_dist = right_safe_dist;
    config.safe_dist = safe_dist;
    config.ref_x = rx;
    config.ref_y = ry;
    config.yaw_diff_max = 0.1;
    config.kappa_max = 0.15;
    config.kappa_min = -0.15;
    config.type_names = type_names;

    std::unique_ptr<OCPInterface<double, 5, 1>> ocp_interface(new OCPInterface<double, 5, 1>());
    OCP_VARIABLES(double, 5, 1)
    int horizon = config.steps.size();
    ConstraintValuePtr<double, 5, 1, Inequality> safe_cons_val(new ConstraintValue<double, 5, 1, Inequality>(horizon));
    ConstraintPtr<double, 5, 1, Inequality> safe_cons(new SafeConstraint<double, 5, 1>(config));
    int step_num = config.steps.size();
    States x_seq(step_num + 1, State::Zero());
    for (int i = 0; i <= step_num; ++i) {
        x_seq[i][kXPos] = config.ref_x[i];
        x_seq[i][kYPos] = config.ref_y[i];
    }
    LateralOffsetConstraint<double, 5, 1> lateral_cons_lb(config, true);
    LateralOffsetConstraint<double, 5, 1> lateral_cons_rb(config, false);
    for (int t = 0; t < step_num; ++t) {
        double left_dis;
        (void)lateral_cons_lb.Evaluate(t, x_seq[t], Control::Zero(), left_dis);
        double right_dis;
        (void)lateral_cons_rb.Evaluate(t, x_seq[t], Control::Zero(), right_dis);
        double safe_dis;
        (void)safe_cons->Evaluate(t, x_seq[t], Control::Zero(), safe_dis);
        if ((t >= 5 and t <= 8) or (t >= 15 and t <= 18)) {
            cout << "left distance = " << left_dis << "\n";
            cout << "right distance = " << right_dis << "\n";
            cout << "safe distance = " << safe_dis << "\n";
            cout << "==================\n";
        }
    }

    //
    //
    //

    return 0;
}
