//
// Created by LTsommer on 2024/5/30.
//

#ifndef CILQR_ILQR_SYSTEM_DECLARATION_HPP
#define CILQR_ILQR_SYSTEM_DECLARATION_HPP

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/StdVector>
#include <vector>

using Eigen::Matrix;
using Eigen::Matrix2d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::VectorXd;
using std::vector;
typedef std::vector<VectorXd, Eigen::aligned_allocator<VectorXd>> VecOfVecXd;
typedef std::vector<MatrixXd, Eigen::aligned_allocator<MatrixXd>> VecOfMatXd;

//    constexpr static T Ts = 0.01;
// M is the num of state variables
//    constexpr static int M = Eigen::Dynamic;
// N is the num of control variables
//    constexpr static int N = Eigen::Dynamic;
//enum XIndex : unsigned int {
//    kXPos,
//    kYPos,
//    kTheta,
//    kKappa,
//    kDkappa,
//};
//
//enum UIndex : unsigned int {
//    kDDkappa,
//    kJerk,
//};

enum StateKIndex : unsigned int { kXPos = 0, kYPos, kTheta, kKappa, kDkappa };

enum UKIndex : unsigned int { kDDkappa = 0 };

enum StateIndex { X = 0, Y, V, Theta };
// index of the acc, w in a Control vec
enum UIndex { Acc = 0, W };

#define OCP_VARIABLES(T, M, N)                                                      \
    typedef Matrix<T, M, 1> State;                                                  \
    typedef Matrix<T, N, 1> Control;                                                \
    typedef Matrix<T, M, 1> VecX;                                                   \
    typedef Matrix<T, N, 1> VecU;                                                   \
    typedef Matrix<T, M, M> A;                                                      \
    typedef Matrix<T, M, N> B;                                                      \
    typedef Matrix<T, M, M> MatrixLXX;                                              \
    typedef Matrix<T, N, N> MatrixLUU;                                              \
    typedef Matrix<T, M, N> MatrixLXU;                                              \
    typedef Matrix<T, N, M> MatrixLUX;                                              \
    typedef Matrix<T, M + N, N + N> MatrixCF;                                       \
    typedef Matrix<T, M + N, 1> MatrixZ;                                            \
    typedef Matrix<T, M + N, M + N> MatrixHZ;                                       \
    typedef std::vector<MatrixZ, Eigen::aligned_allocator<MatrixZ>> MatrixZs;       \
    typedef std::vector<State, Eigen::aligned_allocator<State>> States;             \
    typedef std::vector<Control, Eigen::aligned_allocator<Control>> Controls;       \
    typedef std::vector<VecX, Eigen::aligned_allocator<State>> VecXs;               \
    typedef std::vector<VecU, Eigen::aligned_allocator<Control>> VecUs;             \
    typedef std::vector<MatrixLXX, Eigen::aligned_allocator<MatrixLXX>> MatrixLXXs; \
    typedef std::vector<MatrixLUU, Eigen::aligned_allocator<MatrixLUU>> MatrixLUUs; \
    typedef std::vector<MatrixLXU, Eigen::aligned_allocator<MatrixLXU>> MatrixLXUs; \
    typedef std::vector<MatrixLUX, Eigen::aligned_allocator<MatrixLUX>> MatrixLUXs; \
    typedef std::vector<MatrixCF, Eigen::aligned_allocator<MatrixCF>> MatrixCFs;

template<typename T>
__attribute__((always_inline)) inline T sqr(const T &value) {
    return value * value;
}

template<typename T>
__attribute((always_inline)) inline T cube(const T &value) {
    return sqr<T>(value) * value;
}

template<typename T>
__attribute__((always_inline)) inline T doppel(const T &value) {
    return value + value;
}

template<typename T>
__attribute__((always_inline)) inline int sgn(T &value) {
    return (T(0) < value) - (value < T(0));
}

template<typename T = Vector2d>
__attribute__((always_inline)) inline double cross_prod(const T &w, const T &v) {
    return (w(0) * v(1) - w(1) * v(0));
}

#endif //CILQR_ILQR_SYSTEM_DECLARATION_HPP
