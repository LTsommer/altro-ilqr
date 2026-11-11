#include "frenet_coordinate_system.h"

#include <fstream>
#include <iostream>
#include <utility>

#include <Eigen/LU>

/*
ADD MORE VALIDITY CHECK TO MAKE IT AS ROBUST AS POSSIBLE

FUNCTIONAL COMPONENT EVALUATION CART -> FRENET in different cases, circle, etc

WORST CASE SCENARIO, ETC

*/
#define POLYFIT_GAIN 1000.0


FrenetCoordinateSystem::FrenetCoordinateSystem(
    double length, spline &&x_s, spline &&y_s,
    const FrenetCoordinateSystemParameters &fcs_params)
    : m_fcs_params(fcs_params),
      m_length(length),
      m_x_s(std::move(x_s)),
      m_y_s(std::move(y_s)),
      m_available(false) {
  // If the curvature profile is not given, then compute the curvature profile
  double tmp_s = 0;
  vector<double> vec_s;
  vector<double> vec_k;
  vec_s.reserve(length / fcs_params.step_s + 1);
  while (tmp_s <= length) {
    double dx = x_s.deriv<1>(tmp_s);
    double dy = y_s.deriv<1>(tmp_s);
    double ddx = x_s.deriv<2>(tmp_s);
    double ddy = y_s.deriv<2>(tmp_s);
    double tmp_k = (dx * ddy - dy * ddx) / pow(dx * dx + dy * dy, 1.5);
    if (tmp_k > 0.5) {
      tmp_k = 0.5;
    } else if (tmp_k < -0.5) {
      tmp_k = -0.5;
    }
    vec_k.push_back(tmp_k);
    vec_s.push_back(tmp_s);
    tmp_s += m_fcs_params.step_s;
  }
  m_k_s.set_points(vec_s, vec_k);
  m_available = true;
}

FrenetCoordinateSystem::FrenetCoordinateSystem(
    double length, spline &&x_s, spline &&y_s, spline &&k_s,
    const FrenetCoordinateSystemParameters &fcs_params)
    : m_fcs_params(fcs_params),
      m_length(length),
      m_x_s(std::move(x_s)),
      m_y_s(std::move(y_s)),
      m_k_s(std::move(k_s)),
      m_available(true) {}

FrenetCoordinateSystem::FrenetCoordinateSystem(
    const vector<Point2D> &vec_pts,
    const FrenetCoordinateSystemParameters &fcs_params)
    : m_fcs_params(fcs_params), m_available(false) {
  vector<double> vec_x;
  vector<double> vec_y;
  vec_x.reserve(vec_pts.size());
  vec_y.reserve(vec_pts.size());
  for (auto &pt : vec_pts) {
    vec_x.push_back(pt.x);
    vec_y.push_back(pt.y);
  }
  ConstructFrenetCoordinateSystem(vec_x, vec_y);
}

FrenetCoordinateSystem::FrenetCoordinateSystem(
        const vector<Vec2d> &vec_pts,
        const FrenetCoordinateSystemParameters &fcs_params)
        : m_fcs_params(fcs_params), m_available(false) {
    vector<double> vec_x, vec_y;
    vec_x.reserve(vec_pts.size());
    vec_y.reserve(vec_pts.size());
    for (const auto& pt : vec_pts) {
        vec_x.emplace_back(pt.x());
        vec_y.emplace_back(pt.y());
    }
    ConstructFrenetCoordinateSystem(vec_x, vec_y);
}

FrenetCoordinateSystem::FrenetCoordinateSystem(
    const vector<double> &vec_x, const vector<double> &vec_y,
    const FrenetCoordinateSystemParameters &fcs_params)
    : m_fcs_params(fcs_params), m_available(false) {
  ConstructFrenetCoordinateSystem(vec_x, vec_y);
  //    int i=0;
}
FrenetCoordinateSystem::FrenetCoordinateSystem(
    const vector<double> &vec_x, const vector<double> &vec_y,
    const FrenetCoordinateSystemParameters &fcs_params, double s_start,
    double s_end)
    : m_fcs_params(fcs_params), m_available(false) {
  ConstructFrenetCoordinateSystem(vec_x, vec_y, s_start, s_end);
}

void FrenetCoordinateSystem::set_coarse_search_points() {
  double start_pos{0.0};
  double end_pos{m_length};
  double step = m_fcs_params.coarse_step_s;
  for (double pos = start_pos; pos < end_pos; pos += step) {
    coarse_search_points_.emplace_back(std::make_pair(
        pos, Point2D{m_x_s.extrapolate(pos), m_y_s.extrapolate(pos)}));
  }
  coarse_search_points_.emplace_back(std::make_pair(
      end_pos,
      Point2D{m_x_s.extrapolate(end_pos), m_y_s.extrapolate(end_pos)}));
}

void FrenetCoordinateSystem::ConstructFrenetCoordinateSystem(
    const vector<double> &vec_x, const vector<double> &vec_y,
    double s_polyfit_start, double s_polyfit_end) {
  if (vec_x.empty() || vec_y.empty()) {
    m_available = false;
    return;
  }
//  vector<double> vec_s;
  assert(vec_x.size() == vec_y.size());
  vector<std::pair<double, double>> vec_c;
  vec_s_.push_back(0);
  for (int i = 1; i < (int)vec_x.size(); i++) {
    vec_s_.push_back(vec_s_[i - 1] + sqrt(sqr(vec_x[i] - vec_x[i - 1]) +
                                          sqr(vec_y[i] - vec_y[i - 1])));
  }
  m_x_s.set_points(vec_s_, vec_x);
  m_y_s.set_points(vec_s_, vec_y);
  m_length = vec_s_[vec_x.size() - 1];
  vector<double> vec_k;
  assert(s_polyfit_start < s_polyfit_end);
  s_polyfit_start = std::max(0.0, s_polyfit_start);
  double s_end = std::min(m_length, s_polyfit_end);
  for (int i = 0; i < (int)vec_s_.size(); i++) {
    double tmp_s = vec_s_[i];
    double dx = m_x_s.deriv<1>(tmp_s);
    double dy = m_y_s.deriv<1>(tmp_s);
    double ddx = m_x_s.deriv<2>(tmp_s);
    double ddy = m_y_s.deriv<2>(tmp_s);
    double tmp_k = (dx * ddy - dy * ddx) / pow(dx * dx + dy * dy, 1.5);
    if (tmp_k > 0.5) {
      tmp_k = 0.5;
    } else if (tmp_k < -0.5) {
      tmp_k = -0.5;
    }
    vec_k.push_back(tmp_k);
    if (tmp_s >= s_polyfit_start && tmp_s <= s_end) {
      if (tmp_s > vec_s_[0] + 30.0) {
        vec_c.push_back(std::make_pair(0.0, tmp_s));
      } else {
        vec_c.push_back(std::make_pair(tmp_k * POLYFIT_GAIN, tmp_s));
      }
    }
  }
//  heading_kappa = compute_path_profile(vec_x, vec_y);
//  vector<double> vec_h;
//  std::for_each(heading_kappa.begin(), heading_kappa.end(), [&vec_k, &vec_h](const std::pair<double, double>& p){
//      vec_h.emplace_back(p.first);
//      vec_k.emplace_back(p.second);
//  });
  m_k_s.set_points(vec_s_, vec_k);
//  m_h_s.set_points(vec_s, vec_h);
  // polyfit curvature of frenet coordinate
  curvature_fit_parameters =
      polyfit(vec_c, 4);  // Fitting 3rd degree polynomial:4 coefficients
  curvature_fit_parameters.push_back(POLYFIT_GAIN);
  m_available = true;

  set_coarse_search_points();
}

vector<std::pair<double, double>> FrenetCoordinateSystem::compute_path_profile(
        const vector<double> &x, const vector<double> &y) {
    assert(x.size() == y.size() and x.size() > 3);
    vector<double> headings, kappas;
    // first : heading, second: curvature
    vector<std::pair<double, double>> rst;
    vector<double> dxs, dys, ddxs, ddys, dddxs, dddys;
    int num = x.size();
    rst.resize(num);
    for (int i = 0; i < num; ++i) {
        double delta_x{0.0};
        double delta_y{0.0};
        if (i == 0) {
            delta_x = x[i + 1] - x[i];
            delta_y = y[i + 1] - y[i];
        }
        else if( i == num - 1) {
            delta_x = x[i] - x[i - 1];
            delta_y = y[i] - y[i - 1];
        }
        else {
            delta_x = 0.5 * (x[i + 1] - x[i - 1]);
            delta_y = 0.5 * (y[i + 1] - y[i - 1]);
        }
        dxs.emplace_back(delta_x);
        dys.emplace_back(delta_y);
        headings.emplace_back(std::atan2(delta_y, delta_x));
    }

    double distance{0.0};
    vector<double> accumulated_s;
    double fx{x.front()};
    double fy{y.front()};
    double nx{0.0};
    double ny{0.0};
    for (int i = 1; i < num; ++i) {
        nx = x[i];
        ny = y[i];
        double end_segment_s = std::hypot(fx - nx, fy - ny);
        accumulated_s.emplace_back(end_segment_s + distance);
        distance += end_segment_s;
        fx = nx;
        fy = ny;
    }

    for (int i = 0; i < num; ++i) {
        double xds{0.0};
        double yds{0.0};
        if (i == 0) {
            xds = (x[i + 1] - x[i]) / (accumulated_s[i + 1] - accumulated_s[i]);
            yds = (y[i + 1] - y[i]) / (accumulated_s[i + 1] - accumulated_s[i]);
        }
        else if (i == num - 1) {
            xds = (x[i] - x[i - 1]) / (accumulated_s[i] - accumulated_s[i - 1]);
            yds = (y[i] - y[i - 1]) / (accumulated_s[i] - accumulated_s[i - 1]);
        }
        else {
            xds = (x[i + 1] - x[i - 1]) / (accumulated_s[i + 1] - accumulated_s[i - 1]);
            yds = (y[i + 1] - y[i - 1]) / (accumulated_s[i + 1] - accumulated_s[i - 1]);
        }
        ddxs.emplace_back(xds);
        ddys.emplace_back(yds);
    }

    for (int i = 0; i < num; ++i) {
        double xdds{0.0};
        double ydds{0.0};
        if (i == 0) {
            xdds = (ddxs[i + 1] - ddxs[i]) / (accumulated_s[i + 1] - accumulated_s[i]);
            ydds = (ddys[i + 1] - ddys[i]) / (accumulated_s[i + 1] - accumulated_s[i]);
        }
        else if (i == num - 1) {
            xdds = (ddxs[i] - ddxs[i - 1]) / (accumulated_s[i] - accumulated_s[i - 1]);
            ydds = (ddys[i] - ddys[i - 1]) / (accumulated_s[i] - accumulated_s[i - 1]);
        }
        else {
            xdds = (ddxs[i + 1] - ddxs[i - 1]) / (accumulated_s[i + 1] - accumulated_s[i - 1]);
            ydds = (ddys[i + 1] - ddys[i - 1]) / (accumulated_s[i + 1] - accumulated_s[i - 1]);
        }

        dddxs.emplace_back(xdds);
        dddys.emplace_back(ydds);
    }

    for (int i = 0; i < num; ++i) {
        double xds = ddxs[i];
        double yds = ddys[i];
        double xdds = dddxs[i];
        double ydds = dddys[i];
        double k = (xds * ydds - yds * xdds) / (hypot(xds, yds) * (xds * xds + yds * yds) + 1e-6);
        kappas.emplace_back(k);
    }

    int iter = 0;
    std::for_each(rst.begin(), rst.end(), [&iter, headings, kappas](std::pair<double, double> &p){
        p.first = headings[iter];
        p.second = kappas[iter];
        ++iter;
    });

    return rst;
}


// Find the offset that has the minimal distance to the given point
double FrenetCoordinateSystem::FindMinDistanceOffsetOnCurve(
    const Point2D &cart0, double yaw, bool has_yaw, bool has_heuristics,
    double s_begin, double s_end) const {
  double squared_distance = std::numeric_limits<double>::infinity();
  double shortest_s = 0.0;
  double check_s_begin{0.0}, check_s_end{m_length};
  if (has_heuristics) {
    assert(s_begin < m_length && s_end > 0.0 && s_end > s_begin);
    check_s_begin = std::max(0.0, s_begin);
    check_s_end = std::min(s_end, m_length);
  }

  int start_idx = static_cast<int>(check_s_begin / m_fcs_params.coarse_step_s);
  int end_idx = static_cast<int>(check_s_end / m_fcs_params.coarse_step_s) + 2;
  end_idx = std::min(end_idx, static_cast<int>(coarse_search_points_.size()));
  // Coarsed linear search to counter possible local minimum
  for (int i = start_idx; i < end_idx; i++) {
    double cur_s = coarse_search_points_[i].first;
    const auto &cur_point = coarse_search_points_[i].second;
    double temp = PointsSquareDistance(cart0, cur_point);
    if (squared_distance > temp) {
      // Check if the yaw difference between the road heading and the vehicle
      // heading is too large
      // If larger than pi/2, means that the mapping is not correct
      if (has_yaw && cos(yaw - GetRefCurveHeading(cur_s)) <= 0.0) {
        continue;
      }
      squared_distance = temp;
      shortest_s = cur_s;
    }
  }

  // Gradient descend to find the minimum
  double begin_s = std::max(shortest_s - m_fcs_params.coarse_step_s, 0.0);
  double end_s = std::min(shortest_s + m_fcs_params.coarse_step_s, m_length);
  double s_old = std::numeric_limits<double>::max();
  double s_new = (begin_s + end_s) / 2.0;
  int count = 0;
  const double coarse_step = m_fcs_params.coarse_step_s / 10;
  while (!(std::abs(s_new - s_old) <= m_fcs_params.coord_transform_precision ||
           s_new - begin_s <= 0 || s_new - end_s >= 0 ||
           count > m_fcs_params.max_iter)) {
    s_old = s_new;
    double step_val = -(m_fcs_params.optimization_gamma /
                        HessianSquaredDistanceToPointOnCurve(s_old, cart0)) *
                      GradientSquaredDistanceToOffSetOnCurve(s_old, cart0);
    if (step_val < -coarse_step) {
      step_val = -coarse_step;
    } else if (step_val > coarse_step) {
      step_val = coarse_step;
    }

    s_new += step_val;
    count++;
  }
  return s_old;
}

// Cartesian coordinate to frenet coordinate
TRANSFORM_STATUS FrenetCoordinateSystem::CartCoord2FrenetCoord(
    const Point2D &cart, Point2D &frenet, bool has_heuristics, double s_begin,
    double s_end) const {
  if (!m_available) {
    return TRANSFORM_FAILED;
  }
  double min_point_s;
  if (has_heuristics) {
    min_point_s = FindMinDistanceOffsetOnCurve(cart, 0.0, false, has_heuristics,
                                               s_begin, s_end);
  } else {
    min_point_s = FindMinDistanceOffsetOnCurve(cart);
  }
  double distance =
      std::sqrt(SquaredDistanceToOffSetOnCurve(min_point_s, cart));
  Point2D nearest_point = {m_x_s.extrapolate(min_point_s),
                           m_y_s.extrapolate(min_point_s)};
  Segment2D roadHeading = {
      nearest_point,
      {nearest_point.x + m_fcs_params.step_s * m_x_s.deriv<1>(min_point_s),
       nearest_point.y + m_fcs_params.step_s * m_y_s.deriv<1>(min_point_s)}};

  Segment2D tangent = {nearest_point, cart};

  double cross_p = CrossProduct(roadHeading, tangent);
  frenet.x = min_point_s;
  double cos_yaw_diff = std::abs<double>(DotProduct(roadHeading, tangent) /
                                         (Norm(roadHeading) * Norm(tangent)));
  if (distance > 1.0 /*far away from the reference line*/ &&
      std::abs(cos_yaw_diff) > 0.1 /*tolerance for not completely ortho*/)
    return TRANSFORM_FAILED;
  if (cross_p > 0)
    frenet.y = distance;
  else
    frenet.y = -distance;
  return TRANSFORM_SUCCESS;
}

TRANSFORM_STATUS FrenetCoordinateSystem::CartCoord2FrenetCoord(
        const Vec2d &cart, Vec2d &frenet, bool has_heuristics, double s_begin,
        double s_end) const {
    Point2D cart_(cart.x(), cart.y());
    Point2D frenet_;
    TRANSFORM_STATUS status = CartCoord2FrenetCoord(cart_, frenet_, has_heuristics, s_begin, s_end);
    if (status == TRANSFORM_SUCCESS) {
        frenet.set_x(frenet_.x);
        frenet.set_y(frenet_.y);
        return status;
    }
    return status;
}
/*
std::vector<double> FrenetCoordinateSystem::
CartCoord2FrenetCoord1(const Point2D& cart, Point2D& frenet) const {
    std::vector<double> res;
    double min_point_s = FindMinDistanceOffsetOnCurve(cart);
    double distance = std::sqrt(SquaredDistanceToOffSetOnCurve(min_point_s,
cart)); Point2D nearest_point = { m_x_s(min_point_s),m_y_s(min_point_s) };
    Segment2D roadHeading = { nearest_point,
                                { nearest_point.x + m_fcs_params.step_s *
m_x_s.deriv(1, min_point_s), nearest_point.y + m_fcs_params.step_s *
m_y_s.deriv(1, min_point_s)
                                }
                            };

    Segment2D tangent = { nearest_point, cart };

    double cross_p = CrossProduct(roadHeading, tangent);
    frenet.x = min_point_s;
    res.push_back(frenet.x);
    res.push_back(cross_p);
    res.push_back(m_fcs_params.step_s * m_x_s.deriv(1, min_point_s));
    res.push_back(m_fcs_params.step_s * m_y_s.deriv(1, min_point_s));
    res.push_back(cart.x - nearest_point.x);
    res.push_back(cart.y -nearest_point.y);
    res.push_back(std::abs<double>(
                DotProduct(roadHeading,tangent)
                /(Norm(roadHeading)*Norm(tangent))));

    if (cross_p > 0)
        frenet.y = distance;
    else
        frenet.y = -distance;
    return res;
}*/

// Cartesian coordinate to frenet coordinate, with yaw information
TRANSFORM_STATUS FrenetCoordinateSystem::CartCoord2FrenetCoord(
    const Point2D &cart, Point2D &frenet, double yaw) const {
  if (!m_available) {
    return TRANSFORM_FAILED;
  }
  // TODO(fmy)
  double min_point_s = FindMinDistanceOffsetOnCurve(cart, yaw, false);
  double distance = sqrt(SquaredDistanceToOffSetOnCurve(min_point_s, cart));
  Point2D nearest_point{m_x_s.extrapolate(min_point_s),
                        m_y_s.extrapolate(min_point_s)};
  Segment2D roadHeading = {
      nearest_point,
      {nearest_point.x + m_fcs_params.step_s * m_x_s.deriv<1>(min_point_s),
       nearest_point.y + m_fcs_params.step_s * m_y_s.deriv<1>(min_point_s)}};

  Segment2D tangent = {nearest_point, cart};

  double cross_p = CrossProduct(roadHeading, tangent);
  frenet.x = min_point_s;
  // Frenet coordinate ill-defined if closing to the end of the curve
  if (frenet.x > m_length - m_fcs_params.step_s) {
    return TRANSFORM_FAILED;
  }
  double cos_yaw_diff = std::abs<double>(DotProduct(roadHeading, tangent) /
                                         (Norm(roadHeading) * Norm(tangent)));
  if (distance > 1.0 /*far away from the reference line*/ &&
      std::abs(cos_yaw_diff) > 0.1 /*tolerance for not completely ortho*/) {
    return TRANSFORM_FAILED;
  }
  if (cross_p > 0)
    frenet.y = distance;
  else
    frenet.y = -distance;
  return TRANSFORM_SUCCESS;
}

TRANSFORM_STATUS FrenetCoordinateSystem::CartCoord2FrenetCoord(
        const Vec2d &cart, Vec2d &frenet, double yaw) const {
    Point2D cart_(cart.x(), cart.y());
    Point2D frenet_;
    TRANSFORM_STATUS status = CartCoord2FrenetCoord(cart_, frenet_, yaw);
    if (status == TRANSFORM_SUCCESS) {
        frenet.set_x(frenet_.x);
        frenet.set_y(frenet_.y);
        return status;
    }
    return status;
}
// Frenet coordinate to cartesian coordinate
TRANSFORM_STATUS
FrenetCoordinateSystem::FrenetCoord2CartCoord(const Point2D &frenet,
                                              Point2D &cart) const {
  if (!m_available || frenet.x < 0 || frenet.x > m_length) {
    return TRANSFORM_FAILED;
  }
  Point2D ref_point{m_x_s.extrapolate(frenet.x), m_y_s.extrapolate(frenet.x)};
  // int index = FindClosestIndex(frenet0.x, 0, m_vec_s.size(), m_vec_s);
  // ref_point = { m_vec_x[index],m_vec_y[index] };

  double angle = atan2(m_y_s.deriv<1>(frenet.x), m_x_s.deriv<1>(frenet.x));
  double x1 = ref_point.x + cos(angle + PI / 2) * frenet.y;
  double y1 = ref_point.y + sin(angle + PI / 2) * frenet.y;
  cart.x = x1;
  cart.y = y1;
  return TRANSFORM_SUCCESS;
}

TRANSFORM_STATUS FrenetCoordinateSystem::FrenetCoord2CartCoord(
        const Vec2d &frenet, Vec2d &cart) const {
    Point2D frenet_(frenet.x(), frenet.y());
    Point2D cart_;
    TRANSFORM_STATUS status = FrenetCoord2CartCoord(frenet_, cart_);
    if (status == TRANSFORM_SUCCESS) {
        cart.set_x(cart_.x);
        cart.set_y(cart_.y);
        return status;
    }
    return status;
}

// Cartesian state to coupled frenet state
TRANSFORM_STATUS
FrenetCoordinateSystem::CartState2FrenetState(const CartesianState &cart_state,
                                              FrenetState &frenet_state) const {
  if (!m_available) {
    return TRANSFORM_FAILED;
  }
  Point2D cart_coord;
  cart_coord.x = cart_state.x;
  cart_coord.y = cart_state.y;
  Point2D frenet_coord;
  TRANSFORM_STATUS stat =
      CartCoord2FrenetCoord(cart_coord, frenet_coord, cart_state.yaw);
  if (stat == TRANSFORM_FAILED) return TRANSFORM_FAILED;
  frenet_state.s = frenet_coord.x;
  frenet_state.r = frenet_coord.y;
  double ref_yaw = GetRefCurveHeading(frenet_state.s);
  double ref_curvature = GetRefCurveCurvature(frenet_state.s);
  double yaw_diff = cart_state.yaw - ref_yaw;
  // To get dds and ddr, we should first compute dr_ds and ddr_ds_ds
  double dr_ds = (1 - ref_curvature * frenet_state.r) * tan(yaw_diff);
  frenet_state.dr_ds = dr_ds;
  // dr
  frenet_state.dr = cart_state.speed * sin(yaw_diff);
  double dcurvature = GetRefCurveDCurvature(frenet_state.s);
  double term_a = (dcurvature * frenet_state.r + ref_curvature * dr_ds);
  double term_b = (cart_state.curvature * (1 - ref_curvature * frenet_state.r) /
                       cos(yaw_diff) -
                   ref_curvature);
  double term_c = (1 - ref_curvature * frenet_state.r);
  double ddr_dsds = -term_a * tan(yaw_diff) +
                    term_b * term_c / (cos(yaw_diff) * cos(yaw_diff));
  frenet_state.ddr_dsds = ddr_dsds;
  frenet_state.ds =
      cart_state.speed * cos(yaw_diff) / (1 - ref_curvature * frenet_state.r);
  frenet_state.dds =
      (cart_state.acceleration -
       sqr(frenet_state.ds) * (term_b * term_c * tan(yaw_diff) - term_a) /
           cos(yaw_diff)) /
      (term_c / cos(yaw_diff));

  // Get ddr
  frenet_state.ddr = frenet_state.dds * dr_ds + sqr(frenet_state.ds) * ddr_dsds;
  return TRANSFORM_SUCCESS;
}

// Coupled frenet state to cartesian state
TRANSFORM_STATUS FrenetCoordinateSystem::FrenetState2CartState(
    const FrenetState &frenet_state, CartesianState &cart_state) const {
  if (!m_available) {
    return TRANSFORM_FAILED;
  }
  Point2D frenet_coord;
  frenet_coord.x = frenet_state.s;
  frenet_coord.y = frenet_state.r;
  Point2D cart_coord;
  TRANSFORM_STATUS stat = FrenetCoord2CartCoord(frenet_coord, cart_coord);
  if (stat == TRANSFORM_FAILED) return TRANSFORM_FAILED;
  // Cartesian position
  cart_state.x = cart_coord.x;
  cart_state.y = cart_coord.y;
  double ref_yaw = GetRefCurveHeading(frenet_state.s);
  double ref_curvature = GetRefCurveCurvature(frenet_state.s);
  double dcurvature = GetRefCurveDCurvature(frenet_state.s);

  cart_state.speed =
      sqrt(sqr((1 - ref_curvature * frenet_state.r) * frenet_state.ds) +
           sqr(frenet_state.dr));
  double yaw_diff =
      atan(frenet_state.dr_ds / (1 - ref_curvature * frenet_state.r));
  cart_state.yaw = ref_yaw + yaw_diff;
  double term_a =
      dcurvature * frenet_state.r + ref_curvature * frenet_state.dr_ds;
  double term_b = 1 - ref_curvature * frenet_state.r;
  cart_state.curvature = ((frenet_state.ddr_dsds + term_a * tan(yaw_diff)) *
                              (cos(yaw_diff) * cos(yaw_diff)) / term_b +
                          ref_curvature) *
                         cos(yaw_diff) / term_b;
  cart_state.acceleration =
      frenet_state.dds * term_b / cos(yaw_diff) +
      (sqr(frenet_state.ds) / cos(yaw_diff)) *
          (term_b * tan(yaw_diff) *
               (cart_state.curvature * term_b / cos(yaw_diff) - ref_curvature) -
           term_a);
  return TRANSFORM_SUCCESS;
}
//-----------------------------------------------------------------//
//--- Return fit coefficient of polynomial, at least squares fit   ---//
//--- array: value, pair.first to pair.second                    ---//
//--- n: the number of coefficients                              ---//
//-----------------------------------------------------------------//
vector<double> FrenetCoordinateSystem::polyfit(
    vector<std::pair<double, double>> &array, int n) {
  int num = array.size();
  double a[num * n];
  double b[num];
  double c[n];
  std::memset(c, 0, sizeof(c));
  for (int i = 0; i < num; i++) {
    for (int j = 0; j < n; j++) {
      a[n * i + j] = pow(array[i].second, n - 1 - j);
    }
    b[i] = array[i].first;
  }
  // CvMat A = cvMat(num, n, CV_64FC1, a);
  // CvMat B = cvMat(num, 1, CV_64FC1, b);
  // CvMat X = cvMat(n, 1, CV_64FC1, c);

  // cvSolve(&A,   &B,   &X,   CV_LU);

  // vector<double> res;
  // for(int i = 0; i < n; i++){
  //     res.push_back(X.data.db[i]);
  // }

  // Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > A;
  // A = Eigen::MatrixXd::Zero( num, n);
  // Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > B;
  // B = Eigen::MatrixXd::Zero( num, 1);
  // Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > X;
  // X = Eigen::MatrixXd::Zero( n, 1);
  Eigen::MatrixXd A, B, X;
  A.resize(num, n);
  B.resize(num, 1);
  X.resize(n, 1);
  for (int i = 0; i < num; i++) {
    for (int j = 0; j < n; j++) {
      A(i, j) = a[i * n + j];
    }
  }
  for (int i = 0; i < num; i++) {
    B(i, 0) = b[i];
  }
  X = A.fullPivLu().solve(B);
  vector<double> res;
  for (int i = 0; i < n; i++) {
    res.push_back(X(i, 0));
  }
  return res;
}


