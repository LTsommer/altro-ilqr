// Eigen SimplicialLDLT vs QDLDL on the KKT systems this project used for ALTRO polishing
// before the Riccati rewrite (PR #14):
//
//     K = [ H   D^T    ]      H : block-diagonal cost Hessian over z = (x_0,u_0, ..., x_N)
//         [ D  -delta*I ]      D : dynamics rows  x_{k+1} - A_k x_k - B_k u_k = 0, the initial
//                                  condition, and a few active path-constraint rows per stage
//
// Each solver is timed on: symbolic analysis (once per sparsity pattern), numeric factorization
// (every iteration), and one solve. QDLDL has no fill-reducing ordering of its own, so it is run
// with three orderings; Eigen is run with its default AMD and with the same explicit orderings.
#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

extern "C" {
#include "qdldl.h"
}

using SpMat = Eigen::SparseMatrix<double>; // column-major, int indices
using Perm = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>;
using Eigen::VectorXd;

struct Problem {
    SpMat K; // full symmetric matrix
    VectorXd rhs;
    int n_primal = 0, n_dual = 0;
    std::vector<int> stage_order; // interleaved ordering: (x_k, u_k, duals attached to stage k)
};

// nx states, nu controls, N steps, nc active path constraints per stage.
Problem BuildKkt(int nx, int nu, int N, int nc, double delta, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    const int nz = nx + nu;
    const int n_primal = N * nz + nx;
    std::vector<Eigen::Triplet<double>> trip;
    std::vector<std::vector<int>> stage_vars(N + 1);

    // Cost Hessian: dense SPD block per stage.
    for (int k = 0; k <= N; ++k) {
        const int size = k < N ? nz : nx;
        Eigen::MatrixXd R = Eigen::MatrixXd::NullaryExpr(size, size, [&]() { return uni(gen); });
        const Eigen::MatrixXd H = R * R.transpose() + Eigen::MatrixXd::Identity(size, size);
        for (int i = 0; i < size; ++i) {
            stage_vars[k].push_back(k * nz + i);
            for (int j = 0; j < size; ++j)
                trip.emplace_back(k * nz + i, k * nz + j, H(i, j));
        }
    }
    int row = n_primal;
    auto add = [&](int r, int c, double v) {
        trip.emplace_back(r, c, v);
        trip.emplace_back(c, r, v);
    };
    // Initial condition x_0 = given.
    for (int i = 0; i < nx; ++i, ++row) {
        add(row, i, 1.0);
        stage_vars[0].push_back(row);
    }
    for (int k = 0; k < N; ++k) {
        // Dynamics: x_{k+1} - A x_k - B u_k = 0 (A, B dense, as for a linearized vehicle model).
        for (int i = 0; i < nx; ++i, ++row) {
            for (int j = 0; j < nz; ++j)
                add(row, k * nz + j, (j == i ? -1.0 : 0.0) - 0.1 * uni(gen));
            add(row, (k + 1) * nz + i, 1.0);
            stage_vars[k + 1].push_back(row);
        }
        // Active path constraints on (x_k, u_k).
        for (int c = 0; c < nc; ++c, ++row) {
            for (int j = 0; j < nz; ++j)
                add(row, k * nz + j, uni(gen));
            stage_vars[k].push_back(row);
        }
    }
    const int n = row;
    for (int i = n_primal; i < n; ++i)
        trip.emplace_back(i, i, -delta);

    Problem p;
    p.n_primal = n_primal;
    p.n_dual = n - n_primal;
    p.K.resize(n, n);
    p.K.setFromTriplets(trip.begin(), trip.end());
    p.K.makeCompressed();
    p.rhs = VectorXd::NullaryExpr(n, [&]() { return uni(gen); });
    for (const auto &vars : stage_vars)
        p.stage_order.insert(p.stage_order.end(), vars.begin(), vars.end());
    return p;
}

double MedianMicros(int reps, const std::function<void()> &fn) {
    std::vector<double> t(reps);
    for (auto &ti : t) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        ti = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    }
    std::nth_element(t.begin(), t.begin() + reps / 2, t.end());
    return t[reps / 2];
}

struct Result {
    std::string name;
    double symbolic_us, numeric_us, solve_us, residual;
    long nnz_l;
};

// perm[new] = old. Empty perm = let Eigen pick (AMD).
Result RunEigen(const Problem &p, const std::string &name, const std::vector<int> *perm, int reps) {
    const SpMat &K = p.K;
    SpMat Kp = K;
    Perm P(K.rows());
    if (perm) {
        for (int i = 0; i < K.rows(); ++i)
            P.indices()((*perm)[i]) = i; // old -> new
        Kp = K.selfadjointView<Eigen::Upper>().twistedBy(P);
    }
    using Amd = Eigen::SimplicialLDLT<SpMat, Eigen::Lower>;
    using Nat = Eigen::SimplicialLDLT<SpMat, Eigen::Lower, Eigen::NaturalOrdering<int>>;
    Result r{name, 0, 0, 0, 0, 0};
    VectorXd x;
    auto run = [&](auto &solver) {
        r.symbolic_us = MedianMicros(reps, [&]() { solver.analyzePattern(Kp); });
        r.numeric_us = MedianMicros(reps, [&]() { solver.factorize(Kp); });
        VectorXd b = perm ? VectorXd(P * p.rhs) : p.rhs;
        r.solve_us = MedianMicros(reps, [&]() { x = solver.solve(b); });
        r.nnz_l = solver.matrixL().nestedExpression().nonZeros();
        if (perm) x = P.inverse() * x;
    };
    if (perm) {
        Nat solver;
        run(solver);
    } else {
        Amd solver;
        run(solver);
    }
    r.residual = (K * x - p.rhs).lpNorm<Eigen::Infinity>();
    return r;
}

Result RunQdldl(const Problem &p, const std::string &name, const std::vector<int> &perm, int reps) {
    const SpMat &K = p.K;
    const int n = K.rows();
    Perm P(n);
    for (int i = 0; i < n; ++i)
        P.indices()(perm[i]) = i; // old -> new

    // One-time setup (like OSQP): permuted upper-triangular pattern, plus a map from each of its
    // entries to the entry of K it comes from, so later refactorizations only copy values.
    SpMat index_mat = K;
    for (int i = 0; i < index_mat.nonZeros(); ++i)
        index_mat.valuePtr()[i] = i + 1;
    SpMat permuted(K.rows(), K.cols());
    permuted = index_mat.selfadjointView<Eigen::Upper>().twistedBy(P);
    SpMat U = permuted.triangularView<Eigen::Upper>();
    U.makeCompressed();
    std::vector<int> value_map(U.nonZeros());
    for (int i = 0; i < U.nonZeros(); ++i)
        value_map[i] = static_cast<int>(U.valuePtr()[i]) - 1;
    const QDLDL_int *Ap = U.outerIndexPtr(), *Ai = U.innerIndexPtr();
    std::vector<QDLDL_float> Ax(U.nonZeros());

    std::vector<QDLDL_int> etree(n), Lnz(n), iwork(3 * n), Lp(n + 1);
    std::vector<QDLDL_bool> bwork(n);
    std::vector<QDLDL_float> fwork(n), D(n), Dinv(n);
    Result r{name, 0, 0, 0, 0, 0};
    QDLDL_int sum_lnz = 0;
    r.symbolic_us =
            MedianMicros(reps, [&]() { sum_lnz = QDLDL_etree(n, Ap, Ai, iwork.data(), Lnz.data(), etree.data()); });
    if (sum_lnz < 0) {
        std::printf("QDLDL_etree failed for %s\n", name.c_str());
        return r;
    }
    std::vector<QDLDL_int> Li(sum_lnz);
    std::vector<QDLDL_float> Lx(sum_lnz);
    QDLDL_int n_pos = 0;
    r.numeric_us = MedianMicros(reps, [&]() {
        for (size_t i = 0; i < Ax.size(); ++i)
            Ax[i] = K.valuePtr()[value_map[i]]; // permute values
        n_pos = QDLDL_factor(n,
                             Ap,
                             Ai,
                             Ax.data(),
                             Lp.data(),
                             Li.data(),
                             Lx.data(),
                             D.data(),
                             Dinv.data(),
                             Lnz.data(),
                             etree.data(),
                             bwork.data(),
                             iwork.data(),
                             fwork.data());
    });
    if (n_pos < 0) {
        std::printf("QDLDL_factor failed for %s\n", name.c_str());
        return r;
    }
    VectorXd x;
    r.solve_us = MedianMicros(reps, [&]() {
        x = P * p.rhs;
        QDLDL_solve(n, Lp.data(), Li.data(), Lx.data(), Dinv.data(), x.data());
        x = P.inverse() * x;
    });
    r.nnz_l = sum_lnz; // strictly-lower entries, same convention as Eigen above
    r.residual = (K * x - p.rhs).lpNorm<Eigen::Infinity>();
    return r;
}

void Bench(const std::string &label, int nx, int nu, int N, int nc, int reps) {
    const Problem p = BuildKkt(nx, nu, N, nc, 1e-6, 42);
    const int n = p.K.rows();
    std::vector<int> natural(n);
    for (int i = 0; i < n; ++i)
        natural[i] = i;
    // AMD ordering from Eigen (pattern only, computed once) so QDLDL can use the same one.
    Perm amd_inv;
    Eigen::AMDOrdering<int> amd;
    const SpMat lower = p.K.triangularView<Eigen::Lower>();
    amd(lower.selfadjointView<Eigen::Lower>(), amd_inv); // amd_inv: new -> old
    std::vector<int> amd_perm(amd_inv.indices().data(), amd_inv.indices().data() + n);

    std::vector<Result> results;
    results.push_back(RunEigen(p, "Eigen  AMD (default, old polishing)", nullptr, reps));
    results.push_back(RunQdldl(p, "QDLDL  AMD (ordering from Eigen)", amd_perm, reps));
    results.push_back(RunEigen(p, "Eigen  stage-interleaved", &p.stage_order, reps));
    results.push_back(RunQdldl(p, "QDLDL  stage-interleaved", p.stage_order, reps));
    results.push_back(RunEigen(p, "Eigen  natural [primal; dual]", &natural, reps));
    results.push_back(RunQdldl(p, "QDLDL  natural [primal; dual]", natural, reps));

    std::printf("\n%s: nx=%d nu=%d N=%d, %d active constraints/stage -> KKT %d x %d, nnz(K)=%ld\n",
                label.c_str(),
                nx,
                nu,
                N,
                nc,
                n,
                n,
                static_cast<long>(p.K.nonZeros()));
    std::printf("  %-38s %10s %10s %10s %10s %10s\n",
                "solver / ordering",
                "symbolic",
                "numeric",
                "solve",
                "nnz(L)",
                "residual");
    for (const auto &r : results)
        std::printf("  %-38s %8.1f us %8.1f us %8.1f us %10ld %10.1e\n",
                    r.name.c_str(),
                    r.symbolic_us,
                    r.numeric_us,
                    r.solve_us,
                    r.nnz_l,
                    r.residual);
}

int main() {
    Bench("curvature smoothing size", 5, 1, 60, 2, 201);
    Bench("lat-lon planning size", 6, 2, 50, 3, 201);
    Bench("longer horizon", 6, 2, 200, 3, 101);
    Bench("long horizon", 6, 2, 1000, 3, 31);
    return 0;
}
