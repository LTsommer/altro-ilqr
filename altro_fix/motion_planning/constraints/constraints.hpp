//
// Created by LTsommer on 2024/6/28.
//

#ifndef ALILQR_CONSTRAINTS_HPP
#define ALILQR_CONSTRAINTS_HPP

#include "../../ilqr_system_declaration.hpp"
#include "../../common/math_utils.h"
#include "../../cost_function/cost_function.hpp"
#include "../../constraint_definition/constraint_formulation.h"
#include "../../constraint_definition/constraint.hpp"
#include "../../basic_constraint/basic_constraint.hpp"
#include "../mix_model.h"
#include "../lateral_kinematics.h"
#include "../../frenet_coordinate/frenet_coordinate_system.h"

/* linear acceleration limitation
 * acc_min < a < acc_max
  g(u) = a_ * u + b_[i] > 0
  represents u_min < u < u_max
  for u_min < u, g(u) = -u + u_min < 0
      u < u_max, g(u) = u - u_max < 0
 */
class AccLimitsConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    AccLimitsConstraint(const CostTermConfig &config, const bool is_lb);
    ~AccLimitsConstraint() = default;

    std::string GetName() const override { return std::string("AccLimitsConstraint"); }

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(AccLimitsConstraint)); }

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override;

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override;

private:
    int a_;
    double b_;
};

/* linear jerk limitation
 * acc_min < a < acc_max
  g(u) = a_ * u + b_[i] > 0
  represents u_min < u < u_max
  for u_min < u, g(u) = -u + u_min < 0
      u < u_max, g(u) = u - u_max < 0
 */
class JerkLimitsConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    JerkLimitsConstraint(const CostTermConfig &config, const bool is_lb);
    ~JerkLimitsConstraint() = default;

    std::string GetName() const override { return std::string("JerkLimitsConstraint"); }

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(JerkLimitsConstraint)); }

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override;

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override;

private:
    int a_;
    double b_;
};

/* linear velocity limitation
 * vel_min < vel < acc_max
  g(u) = a_ * u + b_[i] > 0
  represents u_min < u < u_max
  for u_min < u, g(u) = -u + u_min < 0
      u < u_max, g(u) = u - u_max < 0
 */
class VelLimitsConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    VelLimitsConstraint(const CostTermConfig &config, const bool is_lb);
    ~VelLimitsConstraint() = default;

    std::string GetName() const override { return std::string("VelLimitsConstraint"); }

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(VelLimitsConstraint)); }

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override;

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override;

private:
    int a_;
    double b_;
};

/*
  g(u) = a_ * u + b_[i] > 0
  represents u_min < u < u_max
  for u_min < u, g(u) = -u + u_min < 0
      u < u_max, g(u) = u - u_max < 0
 */
class CurveLimitsConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    CurveLimitsConstraint(const CostTermConfig &config, const bool is_lb);
    ~CurveLimitsConstraint() = default;

    std::string GetName() const override { return std::string("CurveLimitsConstraint"); }

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(CurveLimitsConstraint)); }

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override;

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override;

private:
    int a_;
    double b_;
};

/*
  u = v^2 * kappa
  g(u) = a_ * u + b_[i] > 0
  represents u_min < u < u_max
  for u_min < u, g(u) = -u + u_min < 0
      u < u_max, g(u) = u - u_max < 0
 */
class CentripetalAccLimitsConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    CentripetalAccLimitsConstraint(const CostTermConfig &config, const bool is_lb);

    ~CentripetalAccLimitsConstraint() = default;

    std::string GetName() const override { return std::string("CentripetalAccLimitsConstraint"); }

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(CentripetalAccLimitsConstraint)); }

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override;

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override;

private:
    int a_;
    double b_;
};

/*
 *u = 2 * v * a * kappa + v^2 * dot_kappa
  g(u) = a_ * u + b_[i] > 0
  represents u_min < u < u_max
  for u_min < u, g(u) = -u + u_min < 0
      u < u_max, g(u) = u - u_max < 0
 */
class CentripetalJerkLimitsConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    CentripetalJerkLimitsConstraint(const CostTermConfig &config, const bool is_lb);

    ~CentripetalJerkLimitsConstraint() = default;

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(CentripetalJerkLimitsConstraint)); }

    std::string GetName() const override { return std::string("CentripetalJerkLimitsConstraint"); }

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override;

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override;

private:
    int a_;
    double b_;
};

/*
 *u = u - u_ref
  g(u) = a_ * u + b_[i] > 0
  represents u_min < u < u_max
  for u_min < u, g(u) = -u + u_min < 0
      u < u_max, g(u) = u - u_max < 0
 */
class HeadingDiffLimitsConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    HeadingDiffLimitsConstraint(const CostTermConfig &config, const bool is_lb);

    void UpdateConfig(const CostTermConfig &config, const bool is_lb);

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(HeadingDiffLimitsConstraint)); }

    std::string GetName() const override { return std::string("HeadingDiffLimitsConstraint"); }

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override;

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override;

private:
    int a_;
    double offset_;
    vector<double> yaw_;
};

/*----------- Constraint Penalty Functions Definitions-----------*/

class AccConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    AccConstraint(const CostTermConfig &config);

    ~AccConstraint() = default;

    std::string GetName() const override { return std::string("AccConstraint"); }

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(AccConstraint)); }

    // Raw sub-constraints g <= 0 behind the hinge, used by ALTRO polishing.
    bool RawConstraints(const int step,
                        const State &x,
                        const Control &u,
                        RawConstraintRows<double, 6, 2> &out) const override;

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override;

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    /* f(g)'' = f''(g)g'g' + f'(g)g''
     * */
    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override;

private:
    const double w_clb_; // control lower bound weight
    const double w_cub_; // control upper bound weight
    const OneSideCost one_side_cost_;
    const AccLimitsConstraint acc_limit_constraint_lb_;
    const AccLimitsConstraint acc_limit_constraint_ub_;
};

class VelConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    VelConstraint(const CostTermConfig &config);

    ~VelConstraint() = default;

    std::string GetName() const override { return std::string("VelConstraint"); }

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(VelConstraint)); }

    // Raw sub-constraints g <= 0 behind the hinge, used by ALTRO polishing.
    bool RawConstraints(const int step,
                        const State &x,
                        const Control &u,
                        RawConstraintRows<double, 6, 2> &out) const override;

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override;

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    /* f(g)'' = f''(g)g'g' + f'(g)g''
     * */
    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override;

private:
    const double w_clb_; // control lower bound weight
    const double w_cub_; // control upper bound weight
    const OneSideCost one_side_cost_;
    const VelLimitsConstraint vel_limit_constraint_lb_;
    const VelLimitsConstraint vel_limit_constraint_ub_;
};

class JerkConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    JerkConstraint(const CostTermConfig &config);

    ~JerkConstraint() = default;

    std::string GetName() const override { return std::string("JerkConstraint"); }

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(JerkConstraint)); }

    // Raw sub-constraints g <= 0 behind the hinge, used by ALTRO polishing.
    bool RawConstraints(const int step,
                        const State &x,
                        const Control &u,
                        RawConstraintRows<double, 6, 2> &out) const override;

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override;

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    /* f(g)'' = f''(g)g'g' + f'(g)g''
     * */
    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override;

private:
    const double w_clb_; // control lower bound weight
    const double w_cub_; // control upper bound weight
    const OneSideCost one_side_cost_;
    const JerkLimitsConstraint jerk_limit_constraint_lb_;
    const JerkLimitsConstraint jerk_limit_constraint_ub_;
};

/*
  J_ulimit = w_clb * f(g_clb(u)) + w_cub * f(g_cub(u))
  where f() is the penalty function, currently using one-sided quadratic
  function, and g_clb(u), g_cub(u) represents the control limit constraints
 */
class CurveConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    CurveConstraint(const CostTermConfig &config);

    ~CurveConstraint() = default;

    std::string GetName() const override { return std::string("CurveConstraint"); }

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(CurveLimitsConstraint)); }

    // Raw sub-constraints g <= 0 behind the hinge, used by ALTRO polishing.
    bool RawConstraints(const int step,
                        const State &x,
                        const Control &u,
                        RawConstraintRows<double, 6, 2> &out) const override;

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override;

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override;

private:
    const double w_clb_; // control lower bound weight
    const double w_cub_; // control upper bound weight
    const OneSideCost one_side_cost_;
    const CurveLimitsConstraint kappa_limit_constraint_lb_;
    const CurveLimitsConstraint kappa_limit_constraint_ub_;
};

class CentripetalAccConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    CentripetalAccConstraint(const CostTermConfig &config);

    ~CentripetalAccConstraint() = default;

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(CentripetalAccConstraint)); }

    std::string GetName() const override { return std::string("CentripetalAccConstraint"); }

    // Raw sub-constraints g <= 0 behind the hinge, used by ALTRO polishing.
    bool RawConstraints(const int step,
                        const State &x,
                        const Control &u,
                        RawConstraintRows<double, 6, 2> &out) const override;

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override;

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    /* f(g)'' = f''(g)g'g' + f'(g)g''
     * */
    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override;

private:
    const double w_clb_; // control lower bound weight
    const double w_cub_; // control upper bound weight
    const OneSideCost one_side_cost_;
    const CentripetalAccLimitsConstraint centripetal_acc_limit_constraint_lb_;
    const CentripetalAccLimitsConstraint centripetal_acc_limit_constraint_ub_;
};

class CentripetalJerkConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    CentripetalJerkConstraint(const CostTermConfig &config);

    ~CentripetalJerkConstraint() = default;

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(CentripetalJerkConstraint)); }

    std::string GetName() const override { return std::string("CentripetalJerkConstraint"); }

    // Raw sub-constraints g <= 0 behind the hinge, used by ALTRO polishing.
    bool RawConstraints(const int step,
                        const State &x,
                        const Control &u,
                        RawConstraintRows<double, 6, 2> &out) const override;

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override;

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    /* f(g)'' = f''(g)g'g' + f'(g)g''
     * */
    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override;

private:
    const double w_clb_; // control lower bound weight
    const double w_cub_; // control upper bound weight
    const OneSideCost one_side_cost_;
    const CentripetalJerkLimitsConstraint centripetal_jerk_limit_constraint_lb_;
    const CentripetalJerkLimitsConstraint centripetal_jerk_limit_constraint_ub_;
};

class HeadingDiffConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    HeadingDiffConstraint(const CostTermConfig &config);

    void UpdateConfig(const CostTermConfig &config) override;

    ~HeadingDiffConstraint() = default;

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(HeadingDiffConstraint)); }

    std::string GetName() const override { return std::string("HeadingDiffConstraint"); }

    // Raw sub-constraints g <= 0 behind the hinge, used by ALTRO polishing.
    bool RawConstraints(const int step,
                        const State &x,
                        const Control &u,
                        RawConstraintRows<double, 6, 2> &out) const override;

    bool Evaluate(const int step, const State &x, const Control &u, double &cost_val) const override;

    // (f(g))' = f'g'
    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    // (f(g))'' = (f'(g)g')' = f''g'g' + f'g''
    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override;

private:
    const double w_clb_; // control lower bound weight
    const double w_cub_; // control upper bound weight
    const OneSideCost one_side_cost_;
    HeadingDiffLimitsConstraint heading_diff_limit_constraint_lb_;
    HeadingDiffLimitsConstraint heading_diff_limit_constraint_ub_;
};

// TODO change to HuberLoss
class SafeDistanceToBoundaryConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    SafeDistanceToBoundaryConstraint(const CostTermConfig &config);

    ~SafeDistanceToBoundaryConstraint() = default;

    void UpdateConfig(const CostTermConfig &config) override;

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(SafeDistanceToBoundaryConstraint)); }

    virtual std::string GetName() const override { return std::string("SafeDistanceToBoundaryConstraint"); }

    // Raw sub-constraints g <= 0 behind the hinge, used by ALTRO polishing.
    bool RawConstraints(const int step,
                        const State &x,
                        const Control &u,
                        RawConstraintRows<double, 6, 2> &out) const override;

    bool Evaluate(const int step, const State &x, const Control &u, double &val) const override;

    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    // f(g)'' = f'' * g' * g' + f' * g''
    virtual bool Hessian(const int step,
                         const State &x,
                         const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const override;

private:
    const double w_plb_;       // path left bound weight
    const double w_prb_;       // path right bound weight
    const uint32_t steps_num_; // total number of steps
    const OneSideCost one_side_cost_;
    LateralOffsetConstraint<double, 6, 2> line_segment_constraint_lb_;
    LateralOffsetConstraint<double, 6, 2> line_segment_constraint_rb_;
};

/*
  Position-dependent speed limit, e.g. lower speed in narrow passages:
      g(x) = v - v_lim(s(x)) <= 0
  s(x) is the projection of the position onto the reference line and v_lim(s) is linear
  between the given points (constant outside). The limit follows where the vehicle actually is,
  not the time step, so it needs no re-anchoring by the solver.

  The global projection onto the reference line (search for the nearest point) is expensive and
  the solver evaluates every step many times with positions that barely move, so each step keeps
  its last s and refines it with a few local Newton steps on the reference curve r(s):
      s <- s + tangent(s) . (p - r(s)) / (1 - kappa(s) * l)
  This gives the exact projection; the global search is only used for the first evaluation or
  when the local iteration moves too far.
 */
class SpeedLimitBySConstraint final : public Constraint<double, 6, 2, Inequality> {
public:
    static constexpr int kNewtonSteps = 2;
    static constexpr double kMaxLocalShift = 2.0; // m of s; beyond this, project globally

    SpeedLimitBySConstraint(const CostTermConfig &config,
                            const FrenetCoordinateSystem *fcs,
                            vector<double> s_pts,
                            vector<double> v_lim_pts);

    std::string GetName() const override { return std::string("SpeedLimitBySConstraint"); }

    std::type_index GetTypeIndex() const override { return std::type_index(typeid(SpeedLimitBySConstraint)); }

    bool RawConstraints(const int step,
                        const State &x,
                        const Control &u,
                        RawConstraintRows<double, 6, 2> &out) const override;

    bool Evaluate(const int step, const State &x, const Control &u, double &val) const override;

    // dg/dv = 1; dg/d(x, y) = -dv_lim/ds * ds/d(x, y), ds/d(x, y) = tangent / (1 - kappa * l).
    bool Gradient(
            const int step, const State &x, const Control &u, Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override;

    bool Hessian(const int step,
                 const State &x,
                 const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override;

private:
    struct Projection {
        double s = 0.0;
        double ds_dx = 0.0;
        double ds_dy = 0.0;
    };

    // Exact projection of the position onto the reference line, warm-started from the
    // previous s of this step. Returns false if it cannot be projected at all.
    bool Project(const int step, const State &x, Projection &p) const;

    // v_lim(s) and its slope dv_lim/ds.
    const FrenetCoordinateSystem *fcs_;
    vector<double> s_pts_;
    vector<double> v_lim_pts_;
    mutable std::vector<double> last_s_; // warm start of the projection, per step
    mutable std::vector<bool> has_s_;
    mutable std::vector<Vector2d, Eigen::aligned_allocator<Vector2d>> last_p_; // position of last_proj_
    mutable std::vector<Projection> last_proj_;
};

#endif //ALILQR_CONSTRAINTS_HPP
