//
// Created by LTsommer on 2024/6/28.
//
#include <iostream>
#include "mix_model.h"
#include "../ilqr_system_declaration.hpp"
#include "../common/math_utils.h"
#include "../cost_function/cost_function.hpp"
#include "cost_functions/cost_terms.hpp"
#include "../cost_function/cost_terms.h"
#include "../csv_writer.hpp"
#include "../frenet_coordinate/vec2d.h"
#include "../frenet_coordinate/frenet_coordinate_system.h"
#include "../augmented_ilqr/ocp_problem.hpp"
#include "../augmented_ilqr/alilqr_solver.h"
#include "../basic_constraint/basic_constraint.hpp"
#include "../augmented_ilqr/polished_altro/altro_solver.hpp"
#include "constraints/constraints.hpp"
#include "../clock.hpp"
#include <iostream>
#include <typeindex>
#include <unordered_map>
#include <string>
#include <memory>
#include <cmath>

#define M 6
#define N 2
#define N_MAX 999

using std::cout;
using std::endl;

void vector_norm(vector<double> &rx) {
    auto iter = std::min_element(std::begin(rx), std::end(rx));
    int index = std::distance(std::begin(rx), iter);
    auto min_ele = rx[index] + 50;
    std::for_each(std::begin(rx), std::end(rx), [&min_ele](double &val) { val -= min_ele; });
    return;
}

double random(double lb, double ub) {
    double r = rand() % (N_MAX + 1) / (float)(N_MAX + 1);
    return lb + (ub - lb) * r;
}

// Initial guess for the solver (see where u0 is built in main).
constexpr bool kUseTrackingInitialGuess = true;
// Conservative gains: larger ones (e.g. k_heading = 1.0) make the initial rollout oscillate and
// can lead the solver to a much worse local optimum.
constexpr double kInitLatGain = 0.02;    // curvature per metre of lateral offset (1/m^2)
constexpr double kInitHeadingGain = 0.3; // curvature per radian of heading error (1/m)

// Weight of the soft comfort distance (cost per squared meter short of it), the same order as
// the reference-offset weight so neither term drowns the other.
constexpr double kComfortDistWeight = 3000.0;

// Lower speed where the passage is narrow: v_lim = kNarrowSpeedBase + kNarrowSpeedGain * free,
// free = passage width - vehicle width - both minimum buffers (lateral room to maneuver).
// e.g. free = 0.2 m -> 5 m/s, 0.6 m -> 9 m/s; the normal lane (1.4 m) is not limited.
constexpr double kNarrowSpeedBase = 3.0;  // m/s
constexpr double kNarrowSpeedGain = 10.0; // (m/s) per meter of free width
// The limit is never set below what the vehicle can reach from its initial speed with this
// (comfortable) deceleration, otherwise a passage close ahead would make the problem infeasible.
constexpr double kNarrowBrakeDecel = 1.5;  // m/s^2, below the |acc_min| = 2 limit to leave room for jerk
constexpr double kSpeedProfileAccel = 1.0; // m/s^2, comfortable acceleration of the target speed profile
constexpr double kInitSpeedGain = 1.0;     // 1/s

// Test scene: initial speed, and the reference points (about 2 m apart) next to which the
// passage narrows. The first narrowing starts about 20 m ahead, so there is room to slow down
// and move sideways before it (it used to start about 8 m ahead at 10 m/s).
constexpr double kStartSpeed = 8.0;                         // m/s
constexpr int kNarrowLeftBegin = 10, kNarrowLeftEnd = 13;   // left boundary moved in by 1.2 m
constexpr int kNarrowRightBegin = 20, kNarrowRightEnd = 23; // right boundary moved in by 0.8 m

int main(int argc, char **argv) {
    // Directory the result CSV files are written to (must exist); defaults to the working directory.
    const std::string output_dir = argc > 1 ? argv[1] : ".";
    const auto projection_method = argc > 2 ? altro::ParseProjectionMethod(argv[2]) : std::nullopt;
    if (argc > 2 && !projection_method) {
        std::cerr << "Unknown projection method: " << argv[2] << " (use riccati, kkt-eigen, or kkt-qdldl)\n";
        return 1;
    }
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
    int start_node = 20;
    int end_node = 70;
    rx.assign(centerline_x.begin() + start_node, centerline_x.begin() + end_node);
    ry.assign(centerline_y.begin() + start_node, centerline_y.begin() + end_node);
    vector<double> kappa, yaw, velocity, acceleration, ds, s, dt, T;
    vector<Vector2d> p_ref;

    FrenetCoordinateSystemParameters fcs_params;
    fcs_params.Init();
    std::unique_ptr<FrenetCoordinateSystem> fcs(new FrenetCoordinateSystem(centerline_x, centerline_y, fcs_params));
    double vel_max = 10.0;
    double vel_min = 6.0;
    double g = 9.8;
    double acc_max = 0.1 * g;
    double s_old = 0.0;
    double s_new = 0.0;
    double kappa_ = 0.0;
    double dkappa_ = 0.0;
    double lane_width = 1.8;
    double buffer_dis = 0.2;
    double start_s = 0.0;
    for (int i = 1; i < start_node; ++i) {
        Vec2d prev(centerline_x[i - 1], centerline_y[i - 1]);
        Vec2d curr(centerline_x[i], centerline_y[i]);
        start_s += (curr - prev).Length();
    }
    vector<vector<Vector2d>> p_bounds;
    Vec2d start_point(rx[0], ry[0]);
    Vec2d start_frenet_point;
    (void)fcs->CartCoord2FrenetCoord(start_point, start_frenet_point);
    double ss = start_frenet_point.x();
    double cur_x = rx[0];
    double cur_y = ry[0];
    double delta_t = 0.2;
    double acc = 0.0;
    double v0 = kStartSpeed;
    double safe_dist = 1.0;
    vector<double> ref_x, ref_y;
    double half_veh_width = 1.0;
    double max_safe_buffer = 0.3;
    double min_safe_buffer = 0.1;
    vector<double> lb_x, lb_y, rb_x, rb_y;
    vector<double> left_safe_dist, right_safe_dist;
    vector<double> left_comfort_dist, right_comfort_dist;
    vector<double> speed_limit_s, speed_limit_v; // narrow-passage speed limit along s
    vector<Vector2d> left_boundary, right_boundary;
    vector<double> bound_s;
    // One entry per reference point (dt / acceleration have one fewer).
    p_ref.reserve(rx.size());
    p_bounds.reserve(rx.size());
    ref_x.reserve(rx.size());
    ref_y.reserve(rx.size());
    left_safe_dist.reserve(rx.size());
    right_safe_dist.reserve(rx.size());
    left_comfort_dist.reserve(rx.size());
    right_comfort_dist.reserve(rx.size());
    speed_limit_s.reserve(rx.size());
    speed_limit_v.reserve(rx.size());
    left_boundary.reserve(rx.size());
    right_boundary.reserve(rx.size());
    bound_s.reserve(rx.size());
    yaw.reserve(rx.size());
    kappa.reserve(rx.size());
    velocity.reserve(rx.size());
    s.reserve(rx.size());
    T.reserve(rx.size());
    dt.reserve(rx.size());
    acceleration.reserve(rx.size());
    for (int i = 0; i < rx.size(); ++i) {
        Vec2d frenet_pt;
        (void)fcs->CartCoord2FrenetCoord(Vec2d(cur_x, cur_y), frenet_pt);
        double h = fcs->GetRefCurveHeading(frenet_pt.x());
        double k = fcs->GetRefCurveCurvature(frenet_pt.x());
        double v = std::min(vel_max, std::max(std::sqrt(acc_max / k), vel_min));
        double s_front = std::max(frenet_pt.x() - buffer_dis, 0.0);
        double s_back = std::min(frenet_pt.x() + buffer_dis, fcs->GetSlength());
        vector<Vec2d> bound_points;
        if (i >= kNarrowLeftBegin and i <= kNarrowLeftEnd) {
            Vec2d left_lane_boundary_1(s_front, lane_width - 1.2);
            Vec2d left_lane_boundary_2(s_back, lane_width - 1.2);
            Vec2d right_lane_boundary_1(s_front, -lane_width);
            Vec2d right_lane_boundary_2(s_back, -lane_width);
            Vec2d left_ref, right_ref;
            (void)fcs->FrenetCoord2CartCoord(left_lane_boundary_1, left_ref);
            Vector2d lf;
            lf << left_ref.x(), left_ref.y();
            (void)fcs->FrenetCoord2CartCoord(right_lane_boundary_1, right_ref);
            Vector2d rf;
            rf << right_ref.x(), right_ref.y();
            left_boundary.emplace_back(lf);
            right_boundary.emplace_back(rf);
            bound_s.emplace_back(s_front);
            bound_points = vector<Vec2d>{
                    left_lane_boundary_1, left_lane_boundary_2, right_lane_boundary_1, right_lane_boundary_2};
        } else if (i >= kNarrowRightBegin and i <= kNarrowRightEnd) {
            Vec2d left_lane_boundary_1(s_front, lane_width);
            Vec2d left_lane_boundary_2(s_back, lane_width);
            Vec2d right_lane_boundary_1(s_front, -lane_width + 0.8);
            Vec2d right_lane_boundary_2(s_back, -lane_width + 0.8);
            Vec2d left_ref, right_ref;
            (void)fcs->FrenetCoord2CartCoord(left_lane_boundary_1, left_ref);
            Vector2d lf;
            lf << left_ref.x(), left_ref.y();
            (void)fcs->FrenetCoord2CartCoord(right_lane_boundary_1, right_ref);
            Vector2d rf;
            rf << right_ref.x(), right_ref.y();
            left_boundary.emplace_back(lf);
            right_boundary.emplace_back(rf);
            bound_s.emplace_back(s_front);
            bound_points = vector<Vec2d>{
                    left_lane_boundary_1, left_lane_boundary_2, right_lane_boundary_1, right_lane_boundary_2};
        } else {
            Vec2d left_lane_boundary_1(s_front, lane_width);
            Vec2d left_lane_boundary_2(s_back, lane_width);
            Vec2d right_lane_boundary_1(s_front, -lane_width);
            Vec2d right_lane_boundary_2(s_back, -lane_width);
            Vec2d left_ref, right_ref;
            (void)fcs->FrenetCoord2CartCoord(left_lane_boundary_1, left_ref);
            Vector2d lf;
            lf << left_ref.x(), left_ref.y();
            (void)fcs->FrenetCoord2CartCoord(right_lane_boundary_1, right_ref);
            Vector2d rf;
            rf << right_ref.x(), right_ref.y();
            left_boundary.emplace_back(lf);
            right_boundary.emplace_back(rf);
            bound_s.emplace_back(s_front);
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
        // Minimum distance everywhere (hard); the preferred distance is a soft cost that
        // gives way where the passage is narrow.
        left_safe_dist.emplace_back(half_veh_width + min_safe_buffer);
        right_safe_dist.emplace_back(half_veh_width + min_safe_buffer);
        left_comfort_dist.emplace_back(half_veh_width + max_safe_buffer);
        right_comfort_dist.emplace_back(half_veh_width + max_safe_buffer);
        const double free_width = bound_points[0].y() - bound_points[2].y() - 2.0 * (half_veh_width + min_safe_buffer);
        speed_limit_s.emplace_back(frenet_pt.x());
        speed_limit_v.emplace_back(kNarrowSpeedBase + kNarrowSpeedGain * std::max(0.0, free_width));
        yaw.emplace_back(h);
        kappa.emplace_back(k);
        velocity.emplace_back(v);
        s_new = frenet_pt.x();
        if (i == 0) {
            s.emplace_back(0.0);
            T.emplace_back(0.0);
            kappa_ = fcs->GetRefCurveCurvature(s.back());
            dkappa_ = fcs->GetRefCurveDCurvature(s.back());
            Vector2d ref;
            ref << cur_x, cur_y;
            ref_x.emplace_back(cur_x);
            ref_y.emplace_back(cur_y);
            p_ref.emplace_back(ref);
        } else {
            Vec2d prev(rx[i - 1], ry[i - 1]);
            Vec2d curr(cur_x, cur_y);
            s.emplace_back((curr - prev).Length() + s.back());
        }
        if (i > 0) {
            double jerk = random(-1.0, 1.0);
            acc += jerk * delta_t;
            v0 = v0 + acc * delta_t + 0.5 * jerk * sqr(delta_t);
            v0 = std::min(vel_max, std::max(std::min(v0, v), vel_min));
            s_new = s_old + v0 * delta_t + 0.5 * acc * sqr(delta_t) + (1.0 / 6.0) * jerk * cube(delta_t);
            if (s_new > fcs->GetSlength()) {
                break;
            }
            Point2D cur_point = fcs->GetRefCurvePoint(s_new);
            cur_x = cur_point.x;
            cur_y = cur_point.y;
            Vector2d ref;
            ref << cur_x, cur_y;
            p_ref.emplace_back(ref);
            ref_x.emplace_back(cur_x);
            ref_y.emplace_back(cur_y);

            dt.emplace_back(delta_t);
            acceleration.emplace_back(acc);

            T.emplace_back(T.back() + delta_t);
        }
        s_old = s_new;
    }

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

    double w_lat_acc = 50.0;
    double w_lat_jerk = 50.0;
    double w_dot_kappa = 1000.0;
    double w_jerk = 1000.0;
    double w_ref_offset = 3000.0;
    double w_ref_vel = 1000.0;
    double w_u = 1.0;
    double w_clb = 1.0;
    double w_cub = 1.0;
    double w_plb = 2.0;
    double w_prb = 2.0;

    double u_min = -0.1;
    double u_max = 0.1;
    int size = velocity.size();
    vector<double> u_min_(size, u_min);
    vector<double> u_max_(size, u_max);

    vector<BoundType> l_bound(size, BoundType::Others);
    vector<BoundType> r_bound(size, BoundType::Others);

    // Speed limit along s: never below what the vehicle can reach from its initial speed with a
    // comfortable deceleration, v^2 = v0^2 - 2 * a * (s - s0), so the constraint stays feasible.
    for (int i = 0; i < static_cast<int>(speed_limit_s.size()); ++i) {
        const double v_reach_sq =
                sqr(kStartSpeed) - 2.0 * kNarrowBrakeDecel * std::max(0.0, speed_limit_s[i] - speed_limit_s.front());
        speed_limit_v[i] = std::max(speed_limit_v[i], std::sqrt(std::max(0.0, v_reach_sq)));
    }
    // Target speed per time step that respects the limit ahead: a backward pass gives the highest
    // speed at each s from which the vehicle can still brake down to every limit ahead, then a
    // forward rollout from the initial speed with bounded acceleration turns it into a time
    // profile. Without this the target speed (10 m/s everywhere) pulls the vehicle up to speed
    // right before the narrow passage and it has to brake hard.
    vector<double> target_speed(velocity.size());
    {
        vector<double> envelope = speed_limit_v;
        for (int i = static_cast<int>(envelope.size()) - 2; i >= 0; --i) {
            const double ds_i = std::max(0.0, speed_limit_s[i + 1] - speed_limit_s[i]);
            envelope[i] = std::min(envelope[i], std::sqrt(sqr(envelope[i + 1]) + 2.0 * kNarrowBrakeDecel * ds_i));
        }
        double v_cur = kStartSpeed;
        double s_cur = speed_limit_s.front();
        for (int t = 0; t < static_cast<int>(target_speed.size()); ++t) {
            target_speed[t] = v_cur;
            if (t >= static_cast<int>(dt.size())) {
                continue;
            }
            const double v_envelope = InterpolatePiecewiseLinear(speed_limit_s, envelope, s_cur + v_cur * dt[t]);
            const double v_next = std::max(v_cur - kNarrowBrakeDecel * dt[t],
                                           std::min({velocity[std::min(t + 1, static_cast<int>(velocity.size()) - 1)],
                                                     v_envelope,
                                                     v_cur + kSpeedProfileAccel * dt[t]}));
            s_cur += 0.5 * (v_cur + v_next) * dt[t];
            v_cur = v_next;
        }
    }

    CostTermConfig config;
    config.velocity = target_speed;
    config.acceleration = acceleration;
    config.yaw = yaw;
    config.final_yaw_rate = 0.0;
    config.final_yaw = yaw.back();
    config.u_min.swap(u_min_);
    config.u_max.swap(u_max_);
    config.ref_x = centerline_x;
    config.ref_y = centerline_y;
    config.boundary_types.at(0) = l_bound;
    config.boundary_types.at(1) = r_bound;
    config.p_ref = p_ref;
    config.weights[WeightRefOffset] = w_ref_offset;
    config.weights[WeightVelDiff] = w_ref_vel;
    config.weights[WeightU] = w_u;
    config.weights[WeightClb] = w_clb;
    config.weights[WeightCub] = w_cub;
    config.weights[WeightPlb] = w_plb;
    config.weights[WeightPrb] = w_prb;
    config.weights[WeightLatJerk] = w_lat_jerk;
    config.weights[WeightLatAcc] = w_lat_acc;
    config.weights[WeightJerk] = w_jerk;
    config.weights[WeightDotKappa] = w_dot_kappa;
    config.steps = dt;
    config.start_s = start_s;
    config.p_bound = p_bounds;
    config.safe_dist = safe_dist;
    config.buffer_dis = buffer_dis;
    config.lane_width = lane_width;
    config.yaw_diff_max = 0.1;
    config.yaw_diff_min = -0.1;
    config.kappa_max = 0.15;
    config.kappa_min = -0.15;
    config.acc_max = 4.0;
    config.acc_min = -2.0;
    config.lat_jerk_min = -5.0;
    config.lat_jerk_max = 5.0;
    config.lat_acc_min = -2.0;
    config.lat_acc_max = 2.0;
    config.jerk_min = -7.0;
    config.jerk_max = 7.0;
    config.vel_min = 0.0;
    config.vel_max = 100.0;
    config.left_safe_dist = left_safe_dist;
    config.bound_s = bound_s;
    config.bound_segments = p_bounds;
    config.bound_left_safe_dist = left_safe_dist;
    config.bound_right_safe_dist = right_safe_dist;
    config.left_comfort_dist = left_comfort_dist;
    config.right_comfort_dist = right_comfort_dist;
    config.weights[WeightComfortDist] = kComfortDistWeight;
    config.right_safe_dist = right_safe_dist;
    config.half_veh_length = half_veh_width;
    config.max_safe_buffer = max_safe_buffer;
    config.min_safe_buffer = min_safe_buffer;

    std::unique_ptr<OCPInterface<double, M, N>> ocp_interface(new OCPInterface<double, M, N>());
    ocp_interface->SetConfig(config);
    ocp_interface->SetCostUnion(std::move(std::make_unique<CostUnion<double, M, N>>(config)));
    /*-------------add cost functions-------------*/
    ocp_interface->AddCostFunc<CentripetalAccCostFunc>();
    ocp_interface->AddCostFunc<CentripetalJerkCostFunc>();
    ocp_interface->AddCostFunc<DotCurveCostFunc>();
    ocp_interface->AddCostFunc<LinearJerkCostFunc>();
    ocp_interface->AddCostFunc<OffsetCostFunc>();
    ocp_interface->AddCostFunc<TargetSpeedCostFunc>();
    ocp_interface->AddCostFunc<ComfortDistanceCostFunc>();

    /*-------------add penalty constraint functions-------------*/
    int horizon = config.steps.size();
    ConstraintValuePtr<double, M, N, Inequality> acc_cons_val(new ConstraintValue<double, M, N, Inequality>(horizon));
    ConstraintPtr<double, M, N, Inequality> acc_cons(new AccConstraint(config));
    acc_cons_val->LoadConstraint(std::move(acc_cons));
    acc_cons_val->SetDual(0.1);
    acc_cons_val->SetPenalty(1);
    ocp_interface->AddIneqConstraint(std::move(acc_cons_val));

    ConstraintValuePtr<double, M, N, Inequality> jerk_cons_val(new ConstraintValue<double, M, N, Inequality>(horizon));
    ConstraintPtr<double, M, N, Inequality> jerk_cons(new JerkConstraint(config));
    jerk_cons_val->LoadConstraint(std::move(jerk_cons));
    jerk_cons_val->SetDual(0.0);
    jerk_cons_val->SetPenalty(1);
    ocp_interface->AddIneqConstraint(std::move(jerk_cons_val));

    ConstraintValuePtr<double, M, N, Inequality> curve_cons_val(new ConstraintValue<double, M, N, Inequality>(horizon));
    ConstraintPtr<double, M, N, Inequality> curve_cons(new CurveConstraint(config));
    curve_cons_val->LoadConstraint(std::move(curve_cons));
    curve_cons_val->SetDual(2.0);
    curve_cons_val->SetPenalty(10.0);
    ocp_interface->AddIneqConstraint(std::move(curve_cons_val));

    ConstraintValuePtr<double, M, N, Inequality> lat_acc_cons_val(
            new ConstraintValue<double, M, N, Inequality>(horizon));
    ConstraintPtr<double, M, N, Inequality> lat_acc_cons(new CentripetalAccConstraint(config));
    lat_acc_cons_val->LoadConstraint(std::move(lat_acc_cons));
    lat_acc_cons_val->SetDual(0.0);
    lat_acc_cons_val->SetPenalty(1);
    ocp_interface->AddIneqConstraint(std::move(lat_acc_cons_val));

    ConstraintValuePtr<double, M, N, Inequality> lat_jerk_cons_val(
            new ConstraintValue<double, M, N, Inequality>(horizon));
    ConstraintPtr<double, M, N, Inequality> lat_jerk_cons(new CentripetalJerkConstraint(config));
    lat_jerk_cons_val->LoadConstraint(std::move(lat_jerk_cons));
    lat_jerk_cons_val->SetDual(0.0);
    lat_jerk_cons_val->SetPenalty(1.0);
    ocp_interface->AddIneqConstraint(std::move(lat_jerk_cons_val));

    ConstraintValuePtr<double, M, N, Inequality> safe_cons_val(new ConstraintValue<double, M, N, Inequality>(horizon));
    ConstraintPtr<double, M, N, Inequality> safe_cons(new SafeConstraint<double, M, N>(config));
    safe_cons_val->LoadConstraint(std::move(safe_cons));
    safe_cons_val->SetDual(10.0);
    safe_cons_val->SetPenalty(20.0);
    ocp_interface->AddIneqConstraint(std::move(safe_cons_val));

    ConstraintValuePtr<double, M, N, Inequality> heading_cons_val(
            new ConstraintValue<double, M, N, Inequality>(horizon));
    ConstraintPtr<double, M, N, Inequality> heading_cons(new HeadingDiffConstraint(config));
    heading_cons_val->LoadConstraint(std::move(heading_cons));
    heading_cons_val->SetDual(1.0);
    heading_cons_val->SetPenalty(10.0);
    ocp_interface->AddIneqConstraint(std::move(heading_cons_val));

    ConstraintValuePtr<double, M, N, Inequality> vel_cons_val(new ConstraintValue<double, M, N, Inequality>(horizon));
    ConstraintPtr<double, M, N, Inequality> vel_cons(new VelConstraint(config));
    vel_cons_val->LoadConstraint(std::move(vel_cons));
    vel_cons_val->SetDual(0.0);
    vel_cons_val->SetPenalty(1.0);
    ocp_interface->AddIneqConstraint(std::move(vel_cons_val));

    ConstraintValuePtr<double, M, N, Inequality> speed_limit_cons_val(
            new ConstraintValue<double, M, N, Inequality>(horizon));
    ConstraintPtr<double, M, N, Inequality> speed_limit_cons(
            new SpeedLimitBySConstraint(config, fcs.get(), speed_limit_s, speed_limit_v));
    speed_limit_cons_val->LoadConstraint(std::move(speed_limit_cons));
    speed_limit_cons_val->SetDual(0.0);
    speed_limit_cons_val->SetPenalty(10.0);
    ocp_interface->AddIneqConstraint(std::move(speed_limit_cons_val));

    ocp_interface->SetModel(std::make_unique<MixModel>());
    cout << "Nums of Inequality Constraints: " << ocp_interface->GetCostUnionPtr()->GetImmutableIneqConstraints().size()
         << "\n";
    cout << "Nums of Equality Constraints: " << ocp_interface->GetCostUnionPtr()->GetImmutableEqConstraints().size()
         << "\n";
    cout << "all cost functions, constraints and forward model are loaded\n";

    OCP_VARIABLES(double, M, N)
    State x0;
    x0(mXPos) = rx.front();
    x0(mYPos) = ry.front();
    x0(mTheta) = yaw.front();
    x0(mKappa) = kappa_;
    x0(mVel) = kStartSpeed;
    x0(mAcc) = acceleration.front();
    Controls u0(dt.size(), Control::Zero());
    // Initial guess: instead of all-zero controls (whose rollout keeps the initial curvature and
    // acceleration and quickly leaves the road), roll the model forward with simple tracking laws:
    //  - curvature: kappa_cmd = kappa_ref(s + v dt) - k_lat * l - k_heading * (theta - theta_ref),
    //    clamped to the curvature limits, reached in one step through kappa_dot;
    //  - speed: follow the reference speed, capped by what the lateral-acceleration limit allows
    //    on the upcoming curvature, through an acceleration command and the jerk control,
    //    both clamped to their limits.
    if (kUseTrackingInitialGuess) {
        const double k_lat = kInitLatGain;
        const double k_heading = kInitHeadingGain;
        const double k_speed = kInitSpeedGain;
        auto clamp = [](double v, double lo, double hi) { return std::min(hi, std::max(lo, v)); };
        MixModel model;
        State x = model.PrepareInitialState(x0);
        for (int t = 0; t < static_cast<int>(dt.size()); ++t) {
            Vec2d sl;
            (void)fcs->CartCoord2FrenetCoord(Vec2d(x(mXPos), x(mYPos)), sl);
            const double s_next = std::min(sl.x() + std::max(x(mVel), 0.0) * dt[t], fcs->GetSlength());
            const double kappa_ref = fcs->GetRefCurveCurvature(s_next);
            const double heading_err = AngleDifference(x(mTheta), fcs->GetRefCurveHeading(sl.x()));
            const double kappa_cmd =
                    clamp(kappa_ref - k_lat * sl.y() - k_heading * heading_err, config.kappa_min, config.kappa_max);
            double v_ref = config.velocity[std::min(t + 1, static_cast<int>(config.velocity.size()) - 1)];
            v_ref = std::min(v_ref, std::sqrt(config.lat_acc_max / std::max(std::fabs(kappa_ref), 1e-6)));
            const double acc_cmd = clamp(k_speed * (v_ref - x(mVel)), config.acc_min, config.acc_max);
            u0[t](mDotKappa) = (kappa_cmd - x(mKappa)) / dt[t];
            u0[t](mJerk) = clamp((acc_cmd - x(mAcc)) / dt[t], config.jerk_min, config.jerk_max);
            x = model.ForwardCalculation(x, u0[t], dt[t]);
            if (!x.allFinite()) {
                std::cerr << "Invalid motion-planning initial guess at step " << t << '\n';
                return 1;
            }
        }
    }
    States x_res_seq;
    Controls u_res_seq;
    // AL-iLQR followed by ALTRO solution polishing (runs only once AL-iLQR is close to feasible).
    std::unique_ptr<ALTROSolver<double, M, N>> altro_solver(
            new ALTROSolver<double, M, N>(std::move(ocp_interface), dt));
    altro_solver->SetUpdateConfig(true);
    if (projection_method) {
        altro_solver->SetProjectionMethod(*projection_method);
    }
    const auto projection_status = altro_solver->Solve(x0, u0, x_res_seq, u_res_seq);
    vector<double> ilqr_x, ilqr_y, ilqr_h, ilqr_k, ilqr_dk, ilqr_v, ilqr_a, ilqr_j;
    ilqr_x.reserve(x_res_seq.size());
    ilqr_y.reserve(x_res_seq.size());
    ilqr_h.reserve(x_res_seq.size());
    ilqr_k.reserve(x_res_seq.size());
    ilqr_dk.reserve(x_res_seq.size());
    ilqr_v.reserve(x_res_seq.size());
    ilqr_a.reserve(x_res_seq.size());
    ilqr_j.reserve(x_res_seq.size());
    for (int i = 0; i < x_res_seq.size(); ++i) {
        ilqr_x.emplace_back(x_res_seq[i][mXPos]);
        ilqr_y.emplace_back(x_res_seq[i][mYPos]);
        ilqr_h.emplace_back(x_res_seq[i][mTheta]);
        ilqr_k.emplace_back(x_res_seq[i][mKappa]);
        ilqr_v.emplace_back(x_res_seq[i][mVel]);
        ilqr_a.emplace_back(x_res_seq[i][mAcc]);
        if (i < u_res_seq.size()) {
            ilqr_j.emplace_back(u_res_seq[i][mJerk]);
            ilqr_dk.emplace_back(u_res_seq[i][mDotKappa]);
        } else {
            ilqr_j.emplace_back(ilqr_j.back());
            ilqr_dk.emplace_back(ilqr_dk.back());
        }
    }

    vector<double> smooth_v;
    smooth_v.reserve(ilqr_k.size());
    std::for_each(ilqr_k.begin(), ilqr_k.end(), [&vel_max, &vel_min, &smooth_v, &acc_max](const double k) {
        double v = std::min(vel_max, std::max(std::sqrt(acc_max / abs(k)), vel_min));
        smooth_v.emplace_back(v);
    });

    // Export the result; plot it with scripts/plot_results.py.
    const bool projection_incomplete = altro::ProjectionIncomplete(projection_status);
    if (projection_incomplete) {
        std::cerr << "ALTRO projection did not reach tolerance. Exporting the approximate result.\n";
    }
    WriteCsv(output_dir + "/mp_trajectory.csv",
             {{"t", T},
              {"ref_yaw", yaw},
              {"ref_v", velocity},
              {"x", ilqr_x},
              {"y", ilqr_y},
              {"theta", ilqr_h},
              {"kappa", ilqr_k},
              {"v", ilqr_v},
              {"a", ilqr_a},
              {"jerk", ilqr_j},
              {"dkappa", ilqr_dk}});
    WriteCsv(output_dir + "/mp_road.csv",
             {{"ref_x", ref_x},
              {"ref_y", ref_y},
              {"left_x", lb_x},
              {"left_y", lb_y},
              {"right_x", rb_x},
              {"right_y", rb_y}});
    //
    return projection_incomplete ? 1 : 0;
}
