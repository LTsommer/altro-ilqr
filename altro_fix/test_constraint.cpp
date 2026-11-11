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
#include "alilqr_solver.h"
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
    //    test_kinematics();
    //    OCP_PROBLEM_MATRIX(double, M, N);
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
    //    vector<double> centerline_x{-29, -28.073, -27.1461, -26.2195, -25.2932, -24.3675, -23.4423, -22.5179, -21.5944, -20.6719, -19.7506, -18.8305, -17.9115, -16.9932, -16.0749, -15.156, -14.2358, -13.3138, -12.3894, -11.4619, -10.5307, -9.59494, -8.6546, -7.70999, -6.76301, -5.8163, -4.87223, -3.93307, -3.00116, -2.07885, -1.16851, -0.27247, 0.606931, 1.46722, 2.30408, 3.11193, 3.88402, 4.61319, 5.29673, 5.93554, 6.53708, 7.11347, 7.67582, 8.23435, 8.79929, 9.38099, 9.9907, 10.6403, 11.3369, 12.0837, 12.8782, 13.7133, 14.5829, 15.4816, 16.4041, 17.3444, 18.297, 19.2564, 20.2168, 21.1722, 22.1195, 23.059, 23.9914, 24.918, 25.84, 26.7585, 27.6745, 28.589, 29.5032, 30.4184, 31.3354, 32.2546, 33.1761, 34.0994, 35.0242, 35.95, 36.8767, 37.8039, 38.7312, 39.6583, 40.5848, 41.5108, 42.4362, 43.3612, 44.2858, 45.2101, 46.1342, 47.0581, 47.9819, 48.9057, 49.8297, 50.7537, 51.6779, 52.6022, 53.5267, 54.4512, 55.3759, 56.3006, 57.2254, 58.1503, 59.0751, 60};
    //    vector<double> centerline_y{5, 4.98539, 4.97153, 4.95919, 4.94911, 4.94205, 4.93878, 4.94008, 4.94661, 4.9591, 4.97858, 5.00585, 5.0399, 5.07889, 5.12007, 5.16039, 5.19696, 5.22699, 5.24779, 5.25664, 5.25017, 5.22511, 5.18078, 5.11855, 5.0466, 4.97622, 4.91764, 4.88062, 4.87524, 4.91171, 5.00002, 5.1501, 5.37243, 5.67685, 6.06321, 6.52507, 7.05561, 7.64774, 8.29326, 8.9831, 9.70631, 10.4506, 11.2041, 11.9548, 12.6912, 13.4019, 14.0742, 14.6951, 15.256, 15.7527, 16.186, 16.5618, 16.8847, 17.1588, 17.3879, 17.5765, 17.729, 17.8493, 17.9419, 18.011, 18.0602, 18.0921, 18.1092, 18.1137, 18.1081, 18.0948, 18.076, 18.0542, 18.0317, 18.0112, 17.9948, 17.9831, 17.9764, 17.9737, 17.974, 17.9768, 17.9814, 17.987, 17.9929, 17.9982, 18.0025, 18.0056, 18.0074, 18.0082, 18.0083, 18.0076, 18.0065, 18.005, 18.0034, 18.0017, 18.0003, 17.9991, 17.9982, 17.9977, 17.9974, 17.9973, 17.9975, 17.9978, 17.9982, 17.9988, 17.9994, 18};
    //    vector<double> ref_x{-19.7506, -18.8305, -17.9109, -16.9913, -16.0712, -15.1504, -14.229, -13.3071, -12.3846, -11.4614, -10.5363, -9.60804, -8.67375, -7.73174, -6.78169, -5.82796, -4.87636, -3.93259, -3.00292, -2.09295, -1.20701, -0.348906, 0.480788, 1.28814, 2.07545, 2.84559, 3.59578, 4.32201, 5.02497, 5.70644, 6.37277, 7.03031, 7.6826, 8.33153, 8.97809, 9.6238, 10.2733, 10.9333, 11.6132, 12.325, 13.0742, 13.8641, 14.6916, 15.554, 16.448, 17.3678, 18.3068, 19.2581, 20.2143, 21.1682, 22.1156, 23.056, 23.9899, 24.9182, 25.8417, 26.7613, 27.6781, 28.5931, 29.5072, 30.4217, 31.3378, 32.256, 33.1767, 34.0995, 35.024, 35.9497, 36.8764, 37.8036, 38.731, 39.6582, 40.5848, 41.5108, 42.4363, 43.3612, 44.2858, 45.2101, 46.1342, 47.0581, 47.9819, 48.9057, 49.8296, 50.7537, 51.6779, 52.6022, 53.5267, 54.4512, 55.3759, 56.3006, 57.2254, 58.1503};
    //    vector<double> ref_y{4.97858, 5.00484, 5.0246, 5.03615, 5.03729, 5.02823, 5.01108, 4.98817, 4.96164, 4.93309, 4.90358, 4.87374, 4.84474, 4.81969, 4.80483, 4.80917, 4.83898, 4.89869, 4.99269, 5.1247, 5.29817, 5.5176, 5.78959, 6.11839, 6.50542, 6.94828, 7.4409, 7.97929, 8.56151, 9.18289, 9.83737, 10.5135, 11.199, 11.8818, 12.5506, 13.1939, 13.8004, 14.3629, 14.8784, 15.3528, 15.7894, 16.1923, 16.562, 16.8959, 17.1911, 17.446, 17.66, 17.8343, 17.9717, 18.0765, 18.1531, 18.2059, 18.2389, 18.2556, 18.2593, 18.2528, 18.2388, 18.2192, 18.1962, 18.172, 18.1484, 18.1263, 18.106, 18.0873, 18.0703, 18.0551, 18.0419, 18.031, 18.022, 18.0147, 18.0087, 18.0039, 18.0002, 17.9974, 17.9956, 17.9944, 17.9938, 17.9934, 17.9933, 17.9934, 17.9936, 17.994, 17.9945, 17.995, 17.9956, 17.9962, 17.9968, 17.9975, 17.9982, 17.9989};
    vector<double> rx, ry;
    int start_node = 10;
    int end_node = 40;
    rx.assign(centerline_x.begin() + start_node, centerline_x.begin() + end_node);
    ry.assign(centerline_y.begin() + start_node, centerline_y.begin() + end_node);
    //    vector<Vec2d> rxy;
    //    rxy.resize(rx.size());
    //    for (int i = 0; i < rx.size(); ++i) {
    //        rxy[i] = std::move(Vec2d(rx[i], ry[i]));
    //    }
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
    //    double half_veh_width = 0.0;
    //    double max_safe_buffer = 0.0;
    //    double min_safe_buffer = 0.0;
    vector<double> lb_x, lb_y, rb_x, rb_y;
    vector<vector<Vector2d>> p_bounds;
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
        std::for_each(bound_points.begin(), bound_points.end(), [&p_bound, &fcs](const Vec2d &f_point) {
            Vec2d c_point;
            TRANSFORM_STATUS status = fcs->FrenetCoord2CartCoord(f_point, c_point);
            assert(status == TRANSFORM_SUCCESS);
            Vector2d p_b(c_point.x(), c_point.y());
            p_bound.emplace_back(p_b);
        });
        p_bounds.emplace_back(p_bound);
        //        cout << "heading = " << h << "\n";
        //        cout << "kappa = " << k << "\n";
        //        cout << "vel = " << v << "\n";
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
            //            cout << "delta vel = " << velocity[i] - velocity[i - 1] << "\n";
            double acc = (velocity[i] - velocity[i - 1]) / t;
            //            cout << "step = " << step << "\n";
            //            cout << "acc = " << acc << "\n";
            ds.emplace_back(step);
            dt.emplace_back(t);
            acceleration.emplace_back(acc);
            //            if (i == rx.size() - 1)
            //                acceleration.emplace_back(acceleration.back());
        }
        s_old = s_new;
        //        cout << "\n";
    }
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
    //    ocp_interface->SetConfig(config);
    //    ocp_interface->SetCostUnion(std::move(std::make_unique<CostUnion<double, 5, 1>>(config)));
    //    ocp_interface->AddCostFunc<OffsetCostFunc>();
    //    ocp_interface->AddCostFunc<CurvatureCostFunc>();
    //    ocp_interface->AddCostFunc<LateralComfortCostFunc>();
    OCP_VARIABLES(double, 5, 1)
    int horizon = config.steps.size();
    ConstraintValuePtr<double, 5, 1, Inequality> safe_cons_val(new ConstraintValue<double, 5, 1, Inequality>(horizon));
    ConstraintPtr<double, 5, 1, Inequality> safe_cons(new SafeConstraint<double, 5, 1>(config));
    int step_num = config.steps.size();
    //    Controls u_seq(step_num, Control::Zero());
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

    //    safe_cons_val->LoadConstraint(std::move(safe_cons));
    //    safe_cons_val->SetDual(10.0);
    //    safe_cons_val->SetPenalty(10.0);
    //    ocp_interface->AddIneqConstraint(std::move(safe_cons_val));

    //    ConstraintValuePtr<double, 5, 1, Inequality>
    //            control_cons_val(new ConstraintValue<double, 5, 1, Inequality>(horizon));
    //    ConstraintPtr<double, 5, 1, Inequality>
    //            control_cons(new ControlConstraint<double, 5, 1>(config));
    //    control_cons_val->LoadConstraint(std::move(control_cons));
    //    control_cons_val->SetDual(1);
    //    control_cons_val->SetPenalty(1);
    //    ocp_interface->AddIneqConstraint(std::move(control_cons_val));
    //
    //    ConstraintValuePtr<double, 5, 1, Inequality>
    //            heading_track_val(new ConstraintValue<double, 5, 1, Inequality>(horizon));
    //    ConstraintPtr<double, 5, 1, Inequality>
    //            heading_cons(new HeadingConstraint<double, 5, 1>(config));
    //    heading_track_val->LoadConstraint(std::move(heading_cons));
    //    heading_track_val->SetDual(0.01);
    //    heading_track_val->SetPenalty(0.01);
    //    ocp_interface->AddIneqConstraint(std::move(heading_track_val));
    //
    //    ConstraintValuePtr<double, 5, 1, Inequality>
    //            curve_cons_val(new ConstraintValue<double, 5, 1, Inequality>(horizon));
    //    ConstraintPtr<double, 5, 1, Inequality>
    //            curve_cons(new KappaConstraint<double, 5, 1>(config));
    //    curve_cons_val->LoadConstraint(std::move(curve_cons));
    //    curve_cons_val->SetDual(1);
    //    curve_cons_val->SetPenalty(1);
    //    ocp_interface->AddIneqConstraint(std::move(curve_cons_val));
    //
    //    ocp_interface->SetModel(std::make_unique<SmoothKappaModel>());
    //    cout << "all cost functions and forward model are loaded\n";

    //    std::unique_ptr<ILQRSolver<double, 5, 1>> solver(new ILQRSolver(std::move(ocp_interface), ds));
    //    OCP_VARIABLES(double, 5, 1)
    //    State x0;
    //    x0(kXPos) = rx.front();
    //    x0(kYPos) = ry.front();
    //    x0(kTheta) = yaw.front();
    //    x0(kKappa) = kappa_;
    //    x0(kDkappa) = dkappa_;
    //    Controls u0(ds.size(), Control::Zero());
    //    States x_res_seq;
    //    Controls u_res_seq;
    //    solver->Solve(x0, u0, x_res_seq, u_res_seq);
    ////
    //    std::unique_ptr<ALILQRSolver<double, 5, 1>> altro_solver(new ALILQRSolver(std::move(ocp_interface), ds));
    //    altro_solver->Solve(x0, u0, x_res_seq, u_res_seq);
    //    vector<double> ilqr_x, ilqr_y, ilqr_h, ilqr_k, ilqr_dk, ilqr_ddk;
    //    for (int i = 0; i < x_res_seq.size(); ++i) {
    //        ilqr_x.emplace_back(x_res_seq[i][kXPos]);
    //        ilqr_y.emplace_back(x_res_seq[i][kYPos]);
    //        ilqr_h.emplace_back(x_res_seq[i][kTheta]);
    //        ilqr_k.emplace_back(x_res_seq[i][kKappa]);
    //        ilqr_dk.emplace_back(x_res_seq[i][kDkappa]);
    //        ilqr_ddk.emplace_back(u_res_seq[i][kDDkappa]);
    //    }

    //    vector<double> smooth_v;
    //    std::for_each(ilqr_k.begin(), ilqr_k.end(), [&vel_max, &vel_min, &smooth_v, &acc_max](const double k){
    //        double v = std::min(vel_max, std::max(std::sqrt(acc_max / abs(k)), vel_min));
    //        smooth_v.emplace_back(v);
    //    });

    //
    //
    //

    return 0;
}