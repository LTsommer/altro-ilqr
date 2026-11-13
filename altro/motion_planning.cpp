//
// Created by LTsommer on 2024/6/28.
//
#include "motion_planning/mix_model.h"
#include "ilqr_system_declaration.hpp"
#include "common/math_utils.h"
#include "cost_function/cost_function.hpp"
#include "motion_planning/cost_functions/cost_terms.hpp"
#include "cost_function/cost_terms.h"
#include "csv_writer.hpp"
#include "frenet_coordinate/vec2d.h"
#include "frenet_coordinate/frenet_coordinate_system.h"
#include "basic_constraint/basic_constraint.hpp"
#include "motion_planning/constraints/constraints.hpp"
#include "build_altro_solver.hpp"
#include "common/solver_cli.hpp"
#include <iostream>
#include <typeindex>
#include <unordered_map>
#include <string>
#include <memory>
#include <cmath>

#define N_MAX 999

namespace {

using altro::BuildALTROSolver;
using altro::ParseSolverCli;

void vector_norm(vector<double> &rx) {
    auto iter = std::min_element(std::begin(rx), std::end(rx));
    int index = std::distance(std::begin(rx), iter);
    auto min_ele = rx[index] + 50;
    std::for_each(std::begin(rx), std::end(rx), [&min_ele](double &val) {
        val -= min_ele;
    });
    return;
}

double random(double lb, double ub) {
    double r = rand() % (N_MAX + 1) / (float)(N_MAX + 1);
    return lb + (ub - lb) * r;
}

// Lower speed where the passage is narrow: v_lim = kNarrowSpeedBase + kNarrowSpeedGain * free,
// free = passage width - vehicle width - both minimum buffers (lateral room to maneuver).
// e.g. free = 0.2 m -> 5 m/s, 0.6 m -> 9 m/s; the normal lane (1.4 m) is not limited.
constexpr double kNarrowSpeedBase = 3.0;  // m/s
constexpr double kNarrowSpeedGain = 10.0; // (m/s) per meter of free width
// The limit is never set below what the vehicle can reach from its initial speed with this
// (comfortable) deceleration, otherwise a passage close ahead would make the problem infeasible.
// m/s^2, below the |acc_min| = 2 limit to leave room for jerk
constexpr double kNarrowBrakeDecel = 1.5;
// m/s^2, comfortable acceleration of the target speed profile
constexpr double kSpeedProfileAccel = 1.0;

// Test scene: initial speed, and the reference points (about 2 m apart) next to which the
// passage narrows. The first narrowing starts about 20 m ahead, so there is room to slow down
// and move sideways before it (it used to start about 8 m ahead at 10 m/s).
constexpr double kStartSpeed = 8.0; // m/s
// left boundary moved in by 1.2 m
constexpr int kNarrowLeftBegin = 10, kNarrowLeftEnd = 13;
// right boundary moved in by 0.8 m
constexpr int kNarrowRightBegin = 20, kNarrowRightEnd = 23;

struct Scene {
    CostTermConfig config{};
    std::unique_ptr<FrenetCoordinateSystem> fcs;
    altro::MotionPlanningStart start;
    double ref_vel_min, ref_vel_max, ref_lat_acc;
    vector<double> lb_x, lb_y, rb_x, rb_y;
    vector<double> ref_x, ref_y, ref_velocity, time;
    vector<double> speed_limit_s, speed_limit_v;
};

Scene BuildScene() {
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
    int start_node = 20;
    int end_node = 70;
    rx.assign(centerline_x.begin() + start_node,
              centerline_x.begin() + end_node);
    ry.assign(centerline_y.begin() + start_node,
              centerline_y.begin() + end_node);
    vector<double> kappa, yaw, velocity, acceleration, ds, s, dt, T;
    vector<Vector2d> p_ref;

    FrenetCoordinateSystemParameters fcs_params;
    fcs_params.Init();
    std::unique_ptr<FrenetCoordinateSystem> fcs(
            new FrenetCoordinateSystem(centerline_x, centerline_y, fcs_params));
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
    // narrow-passage speed limit along s
    vector<double> speed_limit_s, speed_limit_v;
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
            bound_points = vector<Vec2d>{left_lane_boundary_1,
                                         left_lane_boundary_2,
                                         right_lane_boundary_1,
                                         right_lane_boundary_2};
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
            bound_points = vector<Vec2d>{left_lane_boundary_1,
                                         left_lane_boundary_2,
                                         right_lane_boundary_1,
                                         right_lane_boundary_2};
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
        // Minimum distance everywhere (hard); the preferred distance is a soft cost that
        // gives way where the passage is narrow.
        left_safe_dist.emplace_back(half_veh_width + min_safe_buffer);
        right_safe_dist.emplace_back(half_veh_width + min_safe_buffer);
        left_comfort_dist.emplace_back(half_veh_width + max_safe_buffer);
        right_comfort_dist.emplace_back(half_veh_width + max_safe_buffer);
        const double free_width = bound_points[0].y() - bound_points[2].y() -
                                  2.0 * (half_veh_width + min_safe_buffer);
        speed_limit_s.emplace_back(frenet_pt.x());
        speed_limit_v.emplace_back(kNarrowSpeedBase +
                                   kNarrowSpeedGain *
                                           std::max(0.0, free_width));
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
            s_new = s_old + v0 * delta_t + 0.5 * acc * sqr(delta_t) +
                    (1.0 / 6.0) * jerk * cube(delta_t);
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

    // Speed limit along s: never below what the vehicle can reach from its initial speed with a
    // comfortable deceleration, v^2 = v0^2 - 2 * a * (s - s0), so the constraint stays feasible.
    for (int i = 0; i < static_cast<int>(speed_limit_s.size()); ++i) {
        const double v_reach_sq =
                sqr(kStartSpeed) -
                2.0 * kNarrowBrakeDecel *
                        std::max(0.0, speed_limit_s[i] - speed_limit_s.front());
        speed_limit_v[i] = std::max(speed_limit_v[i],
                                    std::sqrt(std::max(0.0, v_reach_sq)));
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
            const double ds_i =
                    std::max(0.0, speed_limit_s[i + 1] - speed_limit_s[i]);
            envelope[i] = std::min(envelope[i],
                                   std::sqrt(sqr(envelope[i + 1]) +
                                             2.0 * kNarrowBrakeDecel * ds_i));
        }
        double v_cur = kStartSpeed;
        double s_cur = speed_limit_s.front();
        for (int t = 0; t < static_cast<int>(target_speed.size()); ++t) {
            target_speed[t] = v_cur;
            if (t >= static_cast<int>(dt.size())) {
                continue;
            }
            const double v_envelope = InterpolatePiecewiseLinear(
                    speed_limit_s, envelope, s_cur + v_cur * dt[t]);
            const double v_next = std::max(
                    v_cur - kNarrowBrakeDecel * dt[t],
                    std::min({velocity[std::min(
                                      t + 1,
                                      static_cast<int>(velocity.size()) - 1)],
                              v_envelope,
                              v_cur + kSpeedProfileAccel * dt[t]}));
            s_cur += 0.5 * (v_cur + v_next) * dt[t];
            v_cur = v_next;
        }
    }

    const auto size = velocity.size();
    vector<BoundType> l_bound(size, BoundType::Others);
    vector<BoundType> r_bound(size, BoundType::Others);
    Scene scene;
    auto &config = scene.config;
    config.velocity = std::move(target_speed);
    config.acceleration = std::move(acceleration);
    config.yaw = std::move(yaw);
    config.final_yaw_rate = 0.0;
    config.final_yaw = config.yaw.back();
    config.ref_x = std::move(centerline_x);
    config.ref_y = std::move(centerline_y);
    config.boundary_types.at(0) = std::move(l_bound);
    config.boundary_types.at(1) = std::move(r_bound);
    config.p_ref = std::move(p_ref);
    config.steps = std::move(dt);
    config.start_s = start_s;
    config.p_bound = std::move(p_bounds);
    config.safe_dist = safe_dist;
    config.buffer_dis = buffer_dis;
    config.lane_width = lane_width;
    config.bound_s = std::move(bound_s);
    config.bound_segments = config.p_bound;
    config.bound_left_safe_dist = left_safe_dist;
    config.bound_right_safe_dist = right_safe_dist;
    config.left_comfort_dist = std::move(left_comfort_dist);
    config.right_comfort_dist = std::move(right_comfort_dist);
    config.left_safe_dist = std::move(left_safe_dist);
    config.right_safe_dist = std::move(right_safe_dist);
    config.half_veh_length = half_veh_width;
    config.max_safe_buffer = max_safe_buffer;
    config.min_safe_buffer = min_safe_buffer;

    scene.fcs = std::move(fcs);
    scene.start = {Vec2d(rx.front(), ry.front()), kappa_, kStartSpeed};
    scene.ref_vel_min = vel_min;
    scene.ref_vel_max = vel_max;
    scene.ref_lat_acc = acc_max;
    scene.lb_x = std::move(lb_x);
    scene.lb_y = std::move(lb_y);
    scene.rb_x = std::move(rb_x);
    scene.rb_y = std::move(rb_y);
    scene.ref_x = std::move(ref_x);
    scene.ref_y = std::move(ref_y);
    scene.ref_velocity = std::move(velocity);
    scene.time = std::move(T);
    scene.speed_limit_s = std::move(speed_limit_s);
    scene.speed_limit_v = std::move(speed_limit_v);
    return scene;
}

using Model = MixModel;
using State = Model::State;
using Controls = Model::Controls;

// Write the data consumed by scripts/plot_results.py.
void ExportResults(const std::string &output_dir,
                   const Scene &scene,
                   const Model::States &x_res_seq,
                   const Controls &u_res_seq,
                   bool projection_incomplete) {
    vector<double> ilqr_x, ilqr_y, ilqr_h, ilqr_k, ilqr_dk, ilqr_v, ilqr_a,
            ilqr_j;
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
    std::for_each(
            ilqr_k.begin(), ilqr_k.end(), [&scene, &smooth_v](const double k) {
                double v =
                        std::min(scene.ref_vel_max,
                                 std::max(std::sqrt(scene.ref_lat_acc / abs(k)),
                                          scene.ref_vel_min));
                smooth_v.emplace_back(v);
            });

    if (projection_incomplete) {
        std::cerr << "ALTRO projection did not reach tolerance. Exporting the "
                     "approximate result.\n";
    }
    WriteCsv(output_dir + "/mp_trajectory.csv",
             {{"t", scene.time},
              {"ref_yaw", scene.config.yaw},
              {"ref_v", scene.ref_velocity},
              {"x", ilqr_x},
              {"y", ilqr_y},
              {"theta", ilqr_h},
              {"kappa", ilqr_k},
              {"v", ilqr_v},
              {"a", ilqr_a},
              {"jerk", ilqr_j},
              {"dkappa", ilqr_dk}});
    WriteCsv(output_dir + "/mp_road.csv",
             {{"ref_x", scene.ref_x},
              {"ref_y", scene.ref_y},
              {"left_x", scene.lb_x},
              {"left_y", scene.lb_y},
              {"right_x", scene.rb_x},
              {"right_y", scene.rb_y}});
}

} // namespace

int main(int argc, char **argv) {
    auto options = ParseSolverCli(argc, argv, ALTRO_DEFAULT_CONFIG);
    if (!options) {
        return 1;
    }
    options->solver.update_config = true;
    // 1. Build scene data and constraint boundaries.
    auto scene = BuildScene();

    // 2. Build the solver and initial trajectory.
    State x0;
    Controls u0;
    auto solver = BuildALTROSolver(options->solver,
                                   scene.config,
                                   scene.start,
                                   *scene.fcs,
                                   scene.speed_limit_s,
                                   scene.speed_limit_v,
                                   x0,
                                   u0);
    if (!solver) {
        return 1;
    }

    // 3. Solve with the configured solver and initial trajectory.
    Model::States x_res_seq;
    Controls u_res_seq;
    const auto projection_status = solver->Solve(x0, u0, x_res_seq, u_res_seq);

    // 4. Export the results for plotting.
    const bool projection_incomplete =
            altro::ProjectionIncomplete(projection_status);
    ExportResults(options->output_dir,
                  scene,
                  x_res_seq,
                  u_res_seq,
                  projection_incomplete);
    return projection_incomplete ? 1 : 0;
}
