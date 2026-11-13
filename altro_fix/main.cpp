//
// Created by LTsommer on 2024/6/6.
//
#include <iostream>
#include "model/smooth_kappa_model.h"
#include "ilqr_system_declaration.hpp"
#include "common/math_utils.h"
#include "cost_function/cost_function.hpp"
#include "cost_function/cost_terms.h"
#include "csv_writer.hpp"
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
#include <string_view>
#include <memory>
#include <cmath>

using std::cout;
using std::endl;

void vector_norm(vector<double> &rx) {
    auto iter = std::min_element(std::begin(rx), std::end(rx));
    int index = std::distance(std::begin(rx), iter);
    auto min_ele = rx[index];
    std::for_each(std::begin(rx), std::end(rx), [&min_ele](double &val) {
        val -= min_ele;
    });
    return;
}

// Initial guess for the solver (see where u0 is built in main).
constexpr bool kUseTrackingInitialGuess = true;
// Conservative gains: larger ones (e.g. k_heading = 1.0) make the initial rollout oscillate and
// can lead the solver to a much worse local optimum.
// curvature per metre of lateral offset (1/m^2)
constexpr double kInitLatGain = 0.02;
// curvature per radian of heading error (1/m)
constexpr double kInitHeadingGain = 0.3;

// Weight of the soft comfort distance to the boundaries (cost per squared meter short of it),
// the same order as the reference-offset weight.
constexpr double kComfortDistWeight = 3000.0;

int main(int argc, char **argv) {
    const bool print_summary =
            argc > 1 && std::string_view(argv[argc - 1]) == "--summary";
    if (print_summary) {
        --argc;
    }

    // Directory the result CSV files are written to (must exist); defaults to the working directory.
    const std::string output_dir = argc > 1 ? argv[1] : ".";
    const auto projection_method =
            argc > 2 ? altro::ParseProjectionMethod(argv[2]) : std::nullopt;
    if (argc > 2 && !projection_method) {
        std::cerr << "Unknown projection method: " << argv[2]
                  << " (use riccati, kkt-eigen, or kkt-qdldl)\n";
        return 1;
    }
    bool visualize = false;
    vector<double> centerline_x{
            458826.4544207164,  458828.53126436437, 458830.61546950776,
            458832.7064482603,  458834.79742701043, 458836.88840576046,
            458838.97938451247, 458841.07036325993, 458843.16134201374,
            458845.252320763,   458847.34329951426, 458849.4342782653,
            458851.5236426195,  458853.6099811353,  458855.28036619513,
            458856.3180312012,  458857.35569667764, 458858.39336168376,
            458859.43102715584, 458860.39632774034, 458862.47631880885,
            458864.5563098777,  458866.63630094484, 458868.7157781697,
            458870.7828980542,  458872.85001793865, 458873.8835776474,
            458874.91713782307, 458875.9506975317,  458876.9842577073,
            458877.98604426556, 458878.98783128127, 458879.5912861882,
            458880.1947412999,  458880.2913817152,  458880.38802213175,
            458880.4846625475,  458880.5813029623,  458880.6779433763,
            458880.77304845705, 458880.82541630074, 458880.8179323919,
            458880.7511199279,  458880.6246768579,  458880.43791652855,
            458880.1906926864,  458879.88172611984, 458879.51037498855,
            458879.07589149766, 458878.5254063772,  458877.9256015499,
            458877.32579672226, 458876.72599189205, 458875.89673135534,
            458874.2032267701,  458872.5097217082,  458870.8162171199,
            458869.1227120615,  458867.4292074746,  458865.73570241313,
            458864.0421978275,  458862.34869276744, 458860.65518817876,
            458858.96168312,    458855.5746734753,  458852.1876638267,
            458848.80065417854, 458845.4136445309,  458842.0266348838,
            458838.63963219716, 458836.88095205155};
    vector<double> centerline_y{
            4404758.26058061,   4404754.841996041,  4404751.427904584,
            4404748.0179474205, 4404744.607990254,  4404741.198033092,
            4404737.788075925,  4404734.378118761,  4404730.968161597,
            4404727.558204432,  4404724.148247269,  4404720.738290103,
            4404717.327344503,  4404713.914546299,  4404711.182157907,
            4404709.472406139,  4404707.7626545215, 4404706.052902755,
            4404704.343151138,  4404702.752634045,  4404699.335963569,
            4404695.919293091,  4404692.502622611,  4404689.085640642,
            4404685.661167634,  4404682.236694625,  4404680.524458046,
            4404678.8122216165, 4404677.099985038,  4404675.387748607,
            4404673.6584856855, 4404671.929222899,  4404670.649989229,
            4404669.370755611,  4404668.37543625,   4404667.380116888,
            4404666.384797528,  4404665.3894781675, 4404664.394158806,
            4404663.398702398,  4404662.4002310075, 4404661.400414442,
            4404660.4028074015, 4404659.41099599,   4404658.428759468,
            4404657.459983135,  4404656.509103856,  4404655.580833947,
            4404654.680393556,  4404653.848514664,  4404653.048368321,
            4404652.24822198,   4404651.448075639,  4404650.341832818,
            4404649.277858284,  4404648.213884123,  4404647.14990959,
            4404646.085935428,  4404645.021960894,  4404643.957986732,
            4404642.894012198,  4404641.830038036,  4404640.766063504,
            4404639.702089342,  4404637.5741406465, 4404635.446191951,
            4404633.318243254,  4404631.19029456,   4404629.062345863,
            4404626.934386091,  4404625.829429328};
    vector_norm(centerline_x);
    vector_norm(centerline_y);
    vector<double> rx, ry;
    int start_node = 10;
    int end_node = 70;
    rx.assign(centerline_x.begin() + start_node,
              centerline_x.begin() + end_node);
    ry.assign(centerline_y.begin() + start_node,
              centerline_y.begin() + end_node);
    vector<double> kappa, yaw, velocity, acceleration, ds, s, dt;
    vector<Vector2d> p_ref;

    FrenetCoordinateSystemParameters fcs_params;
    fcs_params.Init();
    std::unique_ptr<FrenetCoordinateSystem> fcs(
            new FrenetCoordinateSystem(centerline_x, centerline_y, fcs_params));
    vector<double> left_safe_dist, right_safe_dist;
    vector<double> left_comfort_dist, right_comfort_dist;
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
    left_comfort_dist.reserve(rx.size());
    right_comfort_dist.reserve(rx.size());
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
            bound_points = vector<Vec2d>{left_lane_boundary_1,
                                         left_lane_boundary_2,
                                         right_lane_boundary_1,
                                         right_lane_boundary_2};
        } else if (i >= 15 and i <= 18) {
            Vec2d left_lane_boundary_1(s_front, lane_width);
            Vec2d left_lane_boundary_2(s_back, lane_width);
            Vec2d right_lane_boundary_1(s_front, -lane_width);
            Vec2d right_lane_boundary_2(s_back, -lane_width);
            bound_points = vector<Vec2d>{left_lane_boundary_1,
                                         left_lane_boundary_2,
                                         right_lane_boundary_1,
                                         right_lane_boundary_2};
        } else {
            Vec2d left_lane_boundary_1(s_front, lane_width);
            Vec2d left_lane_boundary_2(s_back, lane_width);
            Vec2d right_lane_boundary_1(s_front, -lane_width);
            Vec2d right_lane_boundary_2(s_back, -lane_width);
            bound_points = vector<Vec2d>{left_lane_boundary_1,
                                         left_lane_boundary_2,
                                         right_lane_boundary_1,
                                         right_lane_boundary_2};
        }

        vector<Vector2d> p_bound;
        p_bound.reserve(bound_points.size());
        std::for_each(bound_points.begin(),
                      bound_points.end(),
                      [&p_bound, &fcs](const Vec2d &f_point) {
                          Vec2d c_point;
                          TRANSFORM_STATUS status =
                                  fcs->FrenetCoord2CartCoord(f_point, c_point);
                          assert(status == TRANSFORM_SUCCESS);
                          Vector2d p_b(c_point.x(), c_point.y());
                          p_bound.emplace_back(p_b);
                      });
        p_bounds.emplace_back(p_bound);
        // Minimum distance everywhere (hard); the preferred distance is a soft cost that gives
        // way where the passage is narrow (see ComfortDistanceCostFunc).
        left_safe_dist.emplace_back(half_veh_width + min_safe_buffer);
        right_safe_dist.emplace_back(half_veh_width + min_safe_buffer);
        left_comfort_dist.emplace_back(half_veh_width + max_safe_buffer);
        right_comfort_dist.emplace_back(half_veh_width + max_safe_buffer);
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
            // The speed profile comes from a curvature-based speed limit and can jump between
            // samples; keep the reference acceleration (used by the lateral comfort cost) within
            // [-2, 4] m/s^2.
            acc = std::min(4.0, std::max(-2.0, acc));
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
    double w_u = 1000.0;
    double w_clb = 1.0;
    double w_cub = 1.0;
    double w_plb = 2.0;
    double w_prb = 2.0;
    double w_jerk = 10.0;
    double w_acc = 10.0;

    double u_min = -5.0;
    double u_max = 5.0;
    int size = velocity.size();
    vector<double> u_min_(size, u_min);
    vector<double> u_max_(size, u_max);

    vector<BoundType> l_bound(size, BoundType::Others);
    vector<BoundType> r_bound(size, BoundType::Others);

    TypeNames type_names;
    type_names[std::type_index(typeid(OffsetCostFunc<double, 5, 1>))] =
            std::string("OffsetCostFunc");
    type_names[std::type_index(typeid(CurvatureCostFunc<double, 5, 1>))] =
            std::string("CurvatureCostFunc");
    type_names[std::type_index(typeid(LateralComfortCostFunc<double, 5, 1>))] =
            std::string("LateralComfortCostFunc");
    type_names[std::type_index(typeid(SafeConstraint<double, 5, 1>))] =
            std::string("SafeConstraint");
    type_names[std::type_index(typeid(ControlConstraint<double, 5, 1>))] =
            std::string("ControlConstraint");
    type_names[std::type_index(typeid(HeadingConstraint<double, 5, 1>))] =
            std::string("HeadingConstraint");

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
    config.left_comfort_dist = left_comfort_dist;
    config.right_comfort_dist = right_comfort_dist;
    config.weights[WeightComfortDist] = kComfortDistWeight;
    config.safe_dist = safe_dist;
    config.ref_x = rx;
    config.ref_y = ry;
    config.yaw_diff_max = 0.1;
    config.yaw_diff_min = -0.1;
    config.kappa_max = 0.15;
    config.kappa_min = -0.15;
    config.type_names = type_names;

    std::unique_ptr<OCPInterface<double, 5, 1>> ocp_interface(
            new OCPInterface<double, 5, 1>());
    ocp_interface->SetConfig(config);
    ocp_interface->SetCostUnion(
            std::move(std::make_unique<CostUnion<double, 5, 1>>(config)));
    ocp_interface->AddCostFunc<OffsetCostFunc>();
    ocp_interface->AddCostFunc<CurvatureCostFunc>();
    ocp_interface->AddCostFunc<LateralComfortCostFunc>();
    ocp_interface->AddCostFunc<ComfortDistanceCostFunc>();

    int horizon = config.steps.size();
    ConstraintValuePtr<double, 5, 1, Inequality> safe_cons_val(
            new ConstraintValue<double, 5, 1, Inequality>(horizon));
    ConstraintPtr<double, 5, 1, Inequality> safe_cons(
            new SafeConstraint<double, 5, 1>(config));
    safe_cons_val->LoadConstraint(std::move(safe_cons));
    safe_cons_val->SetDual(10.0);
    safe_cons_val->SetPenalty(10.0);
    ocp_interface->AddIneqConstraint(std::move(safe_cons_val));

    ConstraintValuePtr<double, 5, 1, Inequality> control_cons_val(
            new ConstraintValue<double, 5, 1, Inequality>(horizon));
    ConstraintPtr<double, 5, 1, Inequality> control_cons(
            new ControlConstraint<double, 5, 1>(config));
    control_cons_val->LoadConstraint(std::move(control_cons));
    control_cons_val->SetDual(1);
    control_cons_val->SetPenalty(1);
    ocp_interface->AddIneqConstraint(std::move(control_cons_val));

    ConstraintValuePtr<double, 5, 1, Inequality> heading_track_val(
            new ConstraintValue<double, 5, 1, Inequality>(horizon));
    ConstraintPtr<double, 5, 1, Inequality> heading_cons(
            new HeadingConstraint<double, 5, 1>(config));
    heading_track_val->LoadConstraint(std::move(heading_cons));
    heading_track_val->SetDual(10);
    heading_track_val->SetPenalty(20);
    ocp_interface->AddIneqConstraint(std::move(heading_track_val));

    ConstraintValuePtr<double, 5, 1, Inequality> curve_cons_val(
            new ConstraintValue<double, 5, 1, Inequality>(horizon));
    ConstraintPtr<double, 5, 1, Inequality> curve_cons(
            new KappaConstraint<double, 5, 1>(config));
    curve_cons_val->LoadConstraint(std::move(curve_cons));
    curve_cons_val->SetDual(10);
    curve_cons_val->SetPenalty(100);
    ocp_interface->AddIneqConstraint(std::move(curve_cons_val));

    ocp_interface->SetModel(std::make_unique<SmoothKappaModel>());
    cout << "all cost functions and forward model are loaded\n";

    OCP_VARIABLES(double, 5, 1)
    State x0;
    x0(kXPos) = rx.front();
    x0(kYPos) = ry.front();
    x0(kTheta) = yaw.front();
    x0(kKappa) = kappa_;
    x0(kDkappa) = dkappa_;
    Controls u0(ds.size(), Control::Zero());
    // Initial guess: instead of all-zero controls (whose rollout keeps the initial curvature and
    // quickly leaves the road), roll the model forward with a simple curvature-tracking law so the
    // starting trajectory already follows the reference line:
    //     kappa_cmd = kappa_ref(s + ds) - k_lat * l - k_heading * (theta - theta_ref)
    // where (s, l) is the Frenet projection of the current position. kappa is steered towards
    // kappa_cmd over two steps through dkappa and the control ddkappa.
    if (kUseTrackingInitialGuess) {
        const double k_lat = kInitLatGain;
        const double k_heading = kInitHeadingGain;
        SmoothKappaModel model;
        State x = model.PrepareInitialState(x0);
        for (int t = 0; t < static_cast<int>(ds.size()); ++t) {
            Vec2d sl;
            (void)fcs->CartCoord2FrenetCoord(Vec2d(x(kXPos), x(kYPos)), sl);
            const double s_next = std::min(sl.x() + ds[t], fcs->GetSlength());
            const double heading_err =
                    AngleDifference(x(kTheta), fcs->GetRefCurveHeading(sl.x()));
            const double kappa_cmd = fcs->GetRefCurveCurvature(s_next) -
                                     k_lat * sl.y() - k_heading * heading_err;
            const double dkappa_des = (kappa_cmd - x(kKappa)) / (2.0 * ds[t]);
            u0[t](kDDkappa) = std::min(
                    u_max, std::max(u_min, (dkappa_des - x(kDkappa)) / ds[t]));
            x = model.ForwardCalculation(x, u0[t], ds[t]);
            if (!x.allFinite()) {
                std::cerr << "Invalid curvature-tracking initial guess at step "
                          << t << '\n';
                return 1;
            }
        }
    }
    States x_res_seq;
    Controls u_res_seq;
    // AL-iLQR followed by ALTRO solution polishing (runs only once AL-iLQR is close to feasible).
    std::unique_ptr<ALTROSolver<double, 5, 1>> altro_solver(
            new ALTROSolver<double, 5, 1>(std::move(ocp_interface), ds));
    altro_solver->SetPrintSummary(print_summary);
    if (projection_method) {
        altro_solver->SetProjectionMethod(*projection_method);
    }
    const auto projection_status =
            altro_solver->Solve(x0, u0, x_res_seq, u_res_seq);
    vector<double> ilqr_x, ilqr_y, ilqr_h, ilqr_k, ilqr_dk, ilqr_ddk;
    ilqr_x.reserve(x_res_seq.size());
    ilqr_y.reserve(x_res_seq.size());
    ilqr_h.reserve(x_res_seq.size());
    ilqr_k.reserve(x_res_seq.size());
    ilqr_dk.reserve(x_res_seq.size());
    ilqr_ddk.reserve(x_res_seq.size());
    for (int i = 0; i < x_res_seq.size(); ++i) {
        ilqr_x.emplace_back(x_res_seq[i][kXPos]);
        ilqr_y.emplace_back(x_res_seq[i][kYPos]);
        ilqr_h.emplace_back(x_res_seq[i][kTheta]);
        ilqr_k.emplace_back(x_res_seq[i][kKappa]);
        ilqr_dk.emplace_back(x_res_seq[i][kDkappa]);
        // One control fewer than states: the ddkappa column is one row shorter.
        if (i < static_cast<int>(u_res_seq.size())) {
            ilqr_ddk.emplace_back(u_res_seq[i][kDDkappa]);
        }
    }

    vector<double> smooth_v;
    smooth_v.reserve(ilqr_k.size());
    std::for_each(ilqr_k.begin(),
                  ilqr_k.end(),
                  [&vel_max, &vel_min, &smooth_v, &acc_max](const double k) {
                      double v = std::min(
                              vel_max,
                              std::max(std::sqrt(acc_max / abs(k)), vel_min));
                      smooth_v.emplace_back(v);
                  });

    // Export the result; plot it with scripts/plot_results.py.
    const bool projection_incomplete =
            altro::ProjectionIncomplete(projection_status);
    if (projection_incomplete) {
        std::cerr << "ALTRO projection did not reach tolerance. Exporting the "
                     "approximate result.\n";
    }
    WriteCsv(output_dir + "/alilqr_trajectory.csv",
             {{"s", s},
              {"ref_yaw", yaw},
              {"x", ilqr_x},
              {"y", ilqr_y},
              {"theta", ilqr_h},
              {"kappa", ilqr_k},
              {"dkappa", ilqr_dk},
              {"ddkappa", ilqr_ddk}});
    WriteCsv(output_dir + "/alilqr_road.csv",
             {{"centerline_x", centerline_x},
              {"centerline_y", centerline_y},
              {"left_x", lb_x},
              {"left_y", lb_y},
              {"right_x", rb_x},
              {"right_y", rb_y}});

    return projection_incomplete ? 1 : 0;
}
