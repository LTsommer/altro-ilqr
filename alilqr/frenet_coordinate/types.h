// NPP - The Navigation Pilot Planning Module
// Copyright (c) 2022 MOMENTA.AI. All Rights Reserved

#ifndef NPP_CORE_COMMON_TYPES_H
#define NPP_CORE_COMMON_TYPES_H

#include <string>

struct Point2D {
  double x = 0.0;
  double y = 0.0;

  Point2D() = default;
  Point2D(double xx, double yy) : x(xx), y(yy) {}
};

struct Point3D {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;

  Point3D() = default;
  Point3D(double xx, double yy, double zz) : x(xx), y(yy), z(zz) {};
};

struct Quaternion {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double w = 0.0;

  Quaternion() = default;
  Quaternion(double xx, double yy, double zz, double ww)
      : x(xx), y(yy), z(zz), w(ww) {}
};

struct Pose2D {
  double x = 0.0;
  double y = 0.0;
  double theta = 0.0;

  Pose2D() = default;
  Pose2D(double xx, double yy, double _theta) : x(xx), y(yy), theta(_theta) {}
};

struct PathPoint {
  // coordinates
  double x;
  double y;
  double z;

  // direction on the x-y plane
  double theta;
  // curvature on the x-y plane
  double kappa;
  // accumulated distance from beginning of the path
  double s;

  // derivative of kappa w.r.t s.
  double dkappa;
  // derivative of derivative of kappa w.r.t s.
  double ddkappa;
  // The lane ID where the path point is on
  std::string lane_id;

  // derivative of x and y w.r.t parametric parameter t in CosThetareferenceline
  double x_derivative{};
  double y_derivative{};

  // zyl add:
  void set_x(double x_) { x = x_; }
  void set_y(double y_) { y = y_; }
  void set_z(double z_) { z = z_; }
  void set_s(double s_) { s = s_; }
  void set_theta(double theta_) { theta = theta_; }
  void set_kappa(double kappa_) { kappa = kappa_; }
  void set_dkappa(double dkappa_) { dkappa = dkappa_; }
  void set_ddkappa(double ddkappa_) { ddkappa = ddkappa_; }
};


#endif  // NPP_CORE_COMMON_TYPES_H
