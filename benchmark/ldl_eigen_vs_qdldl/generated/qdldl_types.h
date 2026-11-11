/* Hand-written equivalent of QDLDL's CMake-generated qdldl_types.h (no cmake on this machine).
 * Options: double precision, 32-bit int indices (QDLDL_LONG off) so index arrays match Eigen's. */
#ifndef QDLDL_TYPES_H
#define QDLDL_TYPES_H
#include <limits.h>
typedef int QDLDL_int;
typedef double QDLDL_float;
typedef unsigned char QDLDL_bool;
#define QDLDL_INT_MAX INT_MAX
#endif
