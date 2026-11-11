#ifndef __FRENET_COORDINATE_SYSTEM_H__
#define __FRENET_COORDINATE_SYSTEM_H__

#include <algorithm>
#include <cassert>
#include <limits>
#include <vector>

#include "cartesian_coordinate_system.h"
#include "types.h"
#include "spline.h"
#include "vec2d.h"

using std::vector;

enum TRANSFORM_STATUS {TRANSFORM_SUCCESS = 0, TRANSFORM_FAILED };

// const double m_fcs_params.zero_speed = 0.1;  // Speed threshold where the
// particle is considered as stopped const double m_fcs_params.precision = 0.01;
// const double m_fcs_params.step_s = 0.3;
// const double m_fcs_params.coarse_step_s = 1.0;
// const double m_fcs_params.optimization_gamma = 0.5;  // Gradient descending
// parameter const int m_fcs_params.max_iter = 15;  // Max iteration for the
// Newton's method

struct FrenetCoordinateSystemParameters {
  double zero_speed_threshold;
  double coord_transform_precision;
  double step_s;
  double coarse_step_s;
  double optimization_gamma;
  int max_iter;
  void Init() {
    zero_speed_threshold = 0.1;
    coord_transform_precision = 1e-2;
    step_s = 0.3;
    coarse_step_s = 1.0;
    optimization_gamma = 0.5;
    max_iter = 15;
  }
};

/*
 * Frenet coordinate:
 * s is the longitudinal offset along the curve,
 * r is the lateral offset with respect to the nearest point on the curve
 */
/**
 * @struct FrenetState
 * @brief Component internal structure:
 * state of an object in Frenet coordinate system with
 * longitudinal motion following a curve and lateral motion perpendicular to a
 * curve
 */
struct FrenetState {
  double s;
  double ds;
  double dds;
  double r;
  double dr;
  double ddr;
  double dr_ds;
  double ddr_dsds;
};

/**
 * @class FrenetCoordinateSystem
 * @brief Define a frenet coordinate system w.r.t certain curve,
 * provide functions for mapping between Cartesian point/state to Frenet
 * point/state
 */
class FrenetCoordinateSystem {
 private:
  FrenetCoordinateSystemParameters m_fcs_params;
  //! Length of the curve
  double m_length;
  //! x coordinate of the curve as a function of the arc length
  spline m_x_s;
  //! y coordinate of the curve as a function of the arc length
  spline m_y_s;
  //! curvature of the curve as a function of the arc length
  spline m_k_s;
  //! heading of the curve as a function of the arc length
//  spline m_h_s;
  vector<double> vec_s_;
  //! Available or not
  bool m_available;
  //! curvature parameters to fit the curve
  vector<double> curvature_fit_parameters;
  //! heading and curvature of the line
  vector<std::pair<double, double>> heading_kappa;
  vector<std::pair<double, Point2D>> coarse_search_points_{};
  void set_coarse_search_points();
  //! Squared distance to a point on the curve (represented by the arclength s
  //! along the path )
  double SquaredDistanceToOffSetOnCurve(double s, const Point2D &cart0) const;
  //! Gradient of the SquaredDistanceToOffSetOnCurve function
  double GradientSquaredDistanceToOffSetOnCurve(double s,
                                                const Point2D &cart0) const;
  //! Hessian of the SquaredDistanceToOffSetOnCurve function
  double HessianSquaredDistanceToPointOnCurve(double s,
                                              const Point2D &cart0) const;
  //! Find the offset that has the minimal distance to the given point
  double FindMinDistanceOffsetOnCurve(const Point2D &cart0, double yaw = 0.0,
                                      bool has_yaw = false,
                                      bool has_heuristics = false,
                                      double s_begin = 0.0,
                                      double s_end = 120.0) const;

  void ConstructFrenetCoordinateSystem(
      const vector<double> &vec_x, const vector<double> &vec_y,
      double s_polyfit_start = 0.0,
      double s_polyfit_end = std::numeric_limits<double>::max());

  vector<std::pair<double, double>> compute_path_profile(const vector<double> &x, const vector<double> &y);

 public:
  //! Default constructor
  FrenetCoordinateSystem() {}
  //! Default destructor
  ~FrenetCoordinateSystem() {}
  //! Constructor with vector of x and y
  explicit FrenetCoordinateSystem(
      const vector<double> &vec_x, const vector<double> &vec_y,
      const FrenetCoordinateSystemParameters &fcs_params);
  //! Constructor with vector of points
  explicit FrenetCoordinateSystem(
      const vector<Point2D> &vec_pts,
      const FrenetCoordinateSystemParameters &fcs_params);
  //! Constructor with vector of points
  explicit FrenetCoordinateSystem(
          const vector<Vec2d>& vec_pts,
          const FrenetCoordinateSystemParameters& fcs_params);
  //! Constructor without default curvature
  explicit FrenetCoordinateSystem(
      double length, spline &&x_s, spline &&y_s,
      const FrenetCoordinateSystemParameters &fcs_params);
  //! Constructor with default curvature
  explicit FrenetCoordinateSystem(
      double length, spline &&x_s, spline &&y_s, spline &&k_s,
      const FrenetCoordinateSystemParameters &fcs_params);

  explicit FrenetCoordinateSystem(
      const vector<double> &vec_x, const vector<double> &vec_y,
      const FrenetCoordinateSystemParameters &fcs_params, double s_start,
      double s_end);

  bool Available() const { return m_available; }
  double GetLength() const { return m_length; }
  vector<double> get_vec_s() {return vec_s_;}
  //! Cartesian coordinate to frenet coordinate
  TRANSFORM_STATUS CartCoord2FrenetCoord(const Point2D &cart, Point2D &frenet,
                                         bool has_heuristics = false,
                                         double s_begin = 0.0,
                                         double s_end = 120.0) const;
  TRANSFORM_STATUS CartCoord2FrenetCoord(const Vec2d &cart, Vec2d &frenet,
                                         bool has_heuristics = false,
                                         double s_begin = 0.0,
                                         double s_end = 120.0) const;
  //! Cartesian coordinate to frenet coordinate
  //! with yaw information to better determine the actual frenet coordinate
  TRANSFORM_STATUS CartCoord2FrenetCoord(const Point2D &cart, Point2D &frenet,
                                         double yaw) const;
  TRANSFORM_STATUS CartCoord2FrenetCoord(const Vec2d &cart, Vec2d &frenet,
                                         double yaw) const;
  // std::vector<double> CartCoord2FrenetCoord1(
  //    const Point2D& cart, Point2D& frenet) const;
  //! Frenet coordinate to cartesian coordinate
  TRANSFORM_STATUS FrenetCoord2CartCoord(const Point2D &frenet,
                                         Point2D &cart) const;
  TRANSFORM_STATUS FrenetCoord2CartCoord(const Vec2d &frenet,
                                         Vec2d &cart) const;
  //! Cartesian state to frenet state
  TRANSFORM_STATUS CartState2FrenetState(const CartesianState &cart_state,
                                         FrenetState &frenet_state) const;
  //! Frenet state to cartesian state
  TRANSFORM_STATUS FrenetState2CartState(const FrenetState &frenet_state,
                                         CartesianState &cart_state) const;
  /*
  std::vector<double>  CartCoord2FrenetCoord1(
      const Point2D& cart, Point2D& frenet) const;
  */
  //! Heading of the ref curve
  double GetRefCurveHeading(double s) const {
    assert(s <= m_length && s >= 0);
//    return m_h_s.extrapolate(s);
    return atan2(m_y_s.deriv<1>(s), m_x_s.deriv<1>(s));
  }
  //! Heading of the ref curve
  double GetRefCurveAvgHeading(double s, double delta_s) const {
    assert(s <= m_length && s >= 0);
    double real_delta = std::min<double>(m_length - s, delta_s);
    if (real_delta > 0.1)
      return atan2(m_y_s.extrapolate(s + real_delta) - m_y_s.extrapolate(s),
                   m_x_s.extrapolate(s + real_delta) - m_x_s.extrapolate(s));
    else
      return GetRefCurveHeading(s);
  }
  //! Curvature of the ref curve
  double GetRefCurveCurvature(double s) const {
    assert(s <= m_length && s >= 0);
    return m_k_s.extrapolate(s);
  }
  //! Derivative of curvature of the ref curve
  double GetRefCurveDCurvature(double s) const {
    assert(s <= m_length && s >= 0);
    return m_k_s.deriv<1>(s);
  }
  //! Get the point on the ref curve using arc length
  Point2D GetRefCurvePoint(double s) const {
    assert(s <= m_length && s >= 0);
    return {m_x_s.extrapolate(s), m_y_s.extrapolate(s)};
  }

  double GetSlength() const { return m_length; }

  std::vector<double> polyfit(vector<std::pair<double, double>> &array, int n);

  const std::vector<double> &GetCurveParameters() const {
    return curvature_fit_parameters;
  }
};

inline double twice(double x) { return x + x; }
inline double sqr(double x) { return x * x; }
inline double cube(double x) { return x * x * x; }

// Squared distance to a point on the curve (represented by the offset s)
__attribute__((always_inline)) inline double
FrenetCoordinateSystem::SquaredDistanceToOffSetOnCurve(
    double s, const Point2D &cart0) const {
  Point2D cur_point = {m_x_s.extrapolate(s), m_y_s.extrapolate(s)};
  Point2D cart = cart0;
  return PointsSquareDistance(cart, cur_point);
}

// Gradient of the previous function
__attribute__((always_inline)) inline double
FrenetCoordinateSystem::GradientSquaredDistanceToOffSetOnCurve(
    double s, const Point2D &cart0) const {
  // return 2.0 * m_x_s.deriv(1, s) * (m_x_s(s) - cart0.x) +
  //        2.0 * m_y_s.deriv(1, s) * (m_y_s(s) - cart0.y);
  return twice(m_x_s.deriv<1>(s) * (m_x_s.extrapolate(s) - cart0.x) +
               m_y_s.deriv<1>(s) * (m_y_s.extrapolate(s) - cart0.y));
}

// Hessian of the distance function
__attribute__((always_inline)) inline double
FrenetCoordinateSystem::HessianSquaredDistanceToPointOnCurve(
    double s, const Point2D &cart0) const {
  // 2.0 * m_x_s.deriv(2, s) * (m_x_s(s) - cart0.x) +
  //        2.0 * m_y_s.deriv(2, s) * (m_y_s(s) - cart0.y) +
  //        2.0 * pow(m_x_s.deriv(1, s), 2) + 2.0 * pow(m_y_s.deriv(1, s), 2);
  return twice(m_x_s.deriv<2>(s) * (m_x_s.extrapolate(s) - cart0.x) +
               m_y_s.deriv<2>(s) * (m_y_s.extrapolate(s) - cart0.y) +
               sqr(m_x_s.deriv<1>(s)) + sqr(m_y_s.deriv<1>(s)));
}


#endif
