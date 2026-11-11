//
// Created by LTsommer on 2024/5/30.
//

#include "cost_calc.h"

template<typename T, unsigned M, unsigned int N>
double CostUnion<T, M, N>::Evaluate(const int step, const State &x, const Control &u) const {
    // AL-iLQR-tutorial 33
    double total_val = 0.0;
    for(const auto &ct : cost_terms_) {
        double cost_val = 0.0;
        bool success = ct->Evaluate(step, x, u, cost_val);
        if (success)
            total_val += cost_val;
    }

    for (const auto &cons_ptr : eqs_) {
        bool success = cons_ptr->Evaluate(step, x, u);
        if (success) {
            double cost_val = cons_ptr->AugLagVal();
            total_val += cost_val;
        }
    }

    for (const auto &cons_ptr : ineqs_) {
        bool success = cons_ptr->Evaluate(step, x, u);
        double cost_val = cons_ptr->AugLagVal();
        if (success)
            total_val += cost_val;
    }

    return total_val;
}

template<typename T, unsigned M, unsigned int N>
void CostUnion<T, M, N>::CalcCostGradient(const int step,
                                          const State &x,
                                          const Control &u,
                                          Eigen::Ref<VecX> lx,
                                          Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    // AL-iLQR-tutorial 41-43
    for (const auto &ct : cost_terms_) {
        VecX cur_lx;
        VecU cur_lu;
        ct->Gradient(step, x, u, cur_lx, cur_lu);
        lx += cur_lx;
        lu += cur_lu;
    }
}

template<typename T, unsigned M, unsigned N>
void CostUnion<T, M, N>::CalcCostHessian(const int step, const State &x, const Control &u,
                                         Eigen::Ref<MatrixLXX> lxx,
                                         Eigen::Ref<MatrixLUU> luu,
                                         Eigen::Ref<MatrixLXU> lxu) const {
    lxx.setZero();
    luu.setZero();
    lxu.setZero();
    // AL-iLQR-tutorial 41-43
    for (const auto &ct : cost_terms_) {
        MatrixLXX cur_lxx;
        MatrixLUU cur_luu;
        MatrixLXU cur_lxu;
        ct->Hessian(step, x, u, cur_lxx, cur_luu, cur_lxu);
        lxx += cur_lxx;
        luu += cur_luu;
        lxu += cur_lxu;
    }
}

template<typename T, unsigned M, unsigned N>
void CostUnion<T, M, N>::CalcAugLagConstraintGradient(const int step, const State &x, const Control &u,
                                                Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const {
    lx.setZero();
    lu.setZero();
    // AL-iLQR-tutorial 44-45
    for (const auto &cons_ptr : eqs_) {
        cons_ptr->Gradient(step, x, u);
        VecX cur_cx = cons_ptr->GradientX();
        VecU cur_cu = cons_ptr->GradientU();
        // lambda + penalty * c(x)
        double lag_cons_val = cons_ptr->GetLambda() +
                cons_ptr->GetPenalty() * cons_ptr->ConsVal();
        lx += cur_cx * lag_cons_val;
        lu += cur_cu * lag_cons_val;
    }

    for (const auto &cons_ptr : ineqs_) {
        cons_ptr->Gradient(step, x, u);
        VecX cur_cx = cons_ptr->GradientX();
        VecU cur_cu = cons_ptr->GradientU();
        // lambda + penalty * c(x)
        double lag_cons_val = cons_ptr->GetLambda() +
                              cons_ptr->GetPenalty() * cons_ptr->ConsVal();
        lx += cur_cx * lag_cons_val;
        lu += cur_cu * lag_cons_val;
    }
}

template<typename T, unsigned M, unsigned N>
void CostUnion<T, M, N>::CalcAugLagConstraintHessian(const int step, const State &x, const Control &u,
                                                     Eigen::Ref<MatrixLXX> cxx,
                                                     Eigen::Ref<MatrixLUU> cuu,
                                                     Eigen::Ref<MatrixLXU> cxu) const {
    cxx.setZero();
    cuu.setZero();
    cxu.setZero();
    for (const auto &cons_ptr : eqs_) {
        cons_ptr->Gradient(step, x, u);
        VecX cur_cx = cons_ptr->GradientX();
        MatrixLXX cur_cxx = cur_cx * cons_ptr->GetPenalty() * cur_cx.transpose();
        cxx += cur_cxx;
        VecU cur_cu = cons_ptr->GradientU();
        MatrixLUU cur_cuu = cur_cu * cons_ptr->GetPenalty() * cur_cu.transpose();
        cuu += cur_cuu;
        MatrixLXU cur_lxu = cur_cx * cons_ptr->GetPenalty() * cur_cu.transpose();
        cxu += cur_lxu;
    }
}

template<typename T, unsigned M, unsigned N>
void CostUnion<T, M , N>::CalcTerminalCostToGo(const State &x) {
    pN.setZero();
    PN.setZero();
    for (const auto &ct : cost_terms_) {
        VecX cur_lx;
        VecU cur_lu;
        bool success = ct->Gradient(horizon_, x, Control::Zero(), cur_lx, cur_lu);
        // AL-iLQR-tutorial 38
        if (success)
            pN += cur_lx;

        MatrixLXX lxx;
        MatrixLUU luu;
        MatrixLXU lxu;
        success = ct->Hessian(horizon_, x, Control::Zero(), lxx, luu, lxu);
        // AL-iLQR-tutorial 39
        if (success)
            PN += lxx;
    }

    for (const auto &cons_ptr : eqs_) {
        cons_ptr->Evaluate(horizon_, x, Control::Zero());
        cons_ptr->Gradient(horizon_, x, Control::Zero());
        // AL-iLQR-tutorial 38-39
        double lag_cons_val = cons_ptr->GetLambda() +
                          cons_ptr->GetPenalty() * cons_ptr->ConsVal();
        pN += cons_ptr->GradientX() * lag_cons_val;
        PN += cons_ptr->GradientX() * cons_ptr->GetPenalty() *
                cons_ptr->GradientX().transpose();
    }

    for (const auto &cons_ptr : ineqs_) {
        cons_ptr->Evaluate(horizon_, x, Control::Zero());
        cons_ptr->Gradient(horizon_, x, Control::Zero());
        // AL-iLQR-tutorial 38-39
        double lag_cons_val = cons_ptr->GetLambda() +
                              cons_ptr->GetPenalty() * cons_ptr->ConsVal();
        pN += cons_ptr->GradientX() * lag_cons_val;
        PN += cons_ptr->GradientX() * cons_ptr->GetPenalty() *
                cons_ptr->GradientX().transpose();
    }
}

template<typename T, unsigned M, unsigned N>
void CostUnion<T, M, N>::EvaluateConstraints(const States &x, const Controls &u) {
    max_violation_ = std::numeric_limits<double>::infinity();
    for (int t = 0; t < horizon_; ++t) {
        for (const auto &cons_ptr : eqs_) {
            bool success = cons_ptr->Evaluate(t, x[t], u[t]);
            if (success) {
                double cons_val = cons_ptr->ConsVal();
                max_violation_ = std::min(max_violation_, cons_val);
            }
        }

        for (const auto&cons_ptr : ineqs_) {
            bool success = cons_ptr->Evaluate(t, x[t], u[t]);
            if (success) {
                double cons_val = cons_ptr->ConsVal();
                max_violation_ = std::min(max_violation_, cons_val);
            }
        }
    }
}

//template<typename T, unsigned M, unsigned N>
//double CostUnion<T, M, N>::GetMaxViolation() {
//    for (const auto &cons_ptr : eqs_) {
//        double cur_max_violation = cons_ptr->MaxViolation();
//        max_violation_ = std::max(max_violation_, cur_max_violation);
//    }
//    for (const auto &cons_ptr : ineqs_) {
//        double cur_max_violation = cons_ptr->MaxViolation();
//        max_violation_ = std::max(max_violation_, cur_max_violation);
//    }
//}

//template<typename T, unsigned M, unsigned N>
//void CostUnion<T, M, N>::ResetViolations() {
//    eqc_violations_.setConstant(Eigen::Infinity);
//    ineqc_violations_.setConstant(Eigen::Infinity);
//}

template class CostUnion<double, 5, 1>;

template class CostUnion<double, 4, 2>;

template class CostUnion<double, 6, 2>;