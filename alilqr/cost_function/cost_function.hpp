//
// Created by LTsommer on 2024/6/20.
//

#ifndef ALILQR_COST_FUNCTION_HPP
#define ALILQR_COST_FUNCTION_HPP

#include "../ilqr_system_declaration.hpp"
#include <typeindex>

template<typename T, unsigned M, unsigned N>
class FunctionBase {
public:
    OCP_VARIABLES(T, M, N)
    FunctionBase() = default;
    ~FunctionBase() = default;

    virtual bool Evaluate(const int step, const State &x, const Control &u, double &val) const = 0;
    virtual bool Gradient(const int step, const State &x, const Control &u,
                          Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const = 0;
    virtual bool Hessian(const int step, const State &x, const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const = 0;
    int GetStateDim() const {return M;}
    int GetControlDim() const {return N;}

};


// start from t(0), end at t(N) if there u(0) ... u(N-1)
template<typename T, int M, int N>
class CostFunc : public FunctionBase<T, M, N> {
public:
    OCP_VARIABLES(T, M, N)
    CostFunc() = default;
    virtual ~CostFunc() = default;

    void set_horizon(const int &step) {
        horizon = step;
    }

    virtual std::type_index get_type_index() {
        return std::type_index(typeid(CostFunc));
    }

    virtual bool Evaluate(const int step, const State &x, const Control &u, double &val) const = 0;

    /* gradient matrix
     * [df_1/dx_1, df_1/dx_2, ... df_1/dx_j, ...]
     * [df_2/dx_1, df_2/dx_2, ... df_2/dx_j, ...]
     * [   :        ...       ...             : ]
     * [df_i/dx_1, df_i/dx_2, ... df_i/dx_j, ...]
     * g_ij = df_i/dx_j
     * */
    virtual bool Gradient(const int step, const State &x, const Control &u,
                          Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const = 0;
    /* H_ij = d(df_i/dx_i)/d_x_j
     * */
    virtual bool Hessian(const int step, const State &x, const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const = 0;
protected:
    int horizon;
};

template<typename T, unsigned M, unsigned N>
using CostFuncPtr = std::unique_ptr<CostFunc<T, M, N>>;


#endif //ALILQR_COST_FUNCTION_HPP
