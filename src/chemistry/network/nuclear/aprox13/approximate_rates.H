#ifndef APPROXIMATE_RATES_H
#define APPROXIMATE_RATES_H

#include <amrex_bridge.H>

#include <tfactors.H>
#include <actual_network.H>
#include <rate_type.H>

namespace approximate_rates {

    using namespace Rates;
    using namespace Species;

    template <typename T>
    inline
    void rate_C12_C12_to_Mg24_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pY = rate_eval.screened_rates(k_Na23_p_to_C12_C12_reaclib);
        Real r_pa = rate_eval.screened_rates(k_Na23_p_to_He4_Ne20_reaclib);
        Real r_pg = rate_eval.screened_rates(k_Na23_p_to_Mg24_reaclib);
        Real dd = 1.0_rt / (r_pY + r_pa + r_pg);
        Real r_Yg = 0.0_rt;
        Real r_Yp = rate_eval.screened_rates(k_C12_C12_to_p_Na23_reaclib);
        rate = r_Yg + r_Yp * r_pg * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_pY = rate_eval.dscreened_rates_dT(k_Na23_p_to_C12_C12_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_Na23_p_to_He4_Ne20_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_Na23_p_to_Mg24_reaclib);
            Real drdT_Yg = 0.0_rt;
            Real drdT_Yp = rate_eval.dscreened_rates_dT(k_C12_C12_to_p_Na23_reaclib);
            drate_dT = drdT_Yg + drdT_Yp * r_pg * dd + r_Yp * drdT_pg * dd - r_Yp * r_pg * dd * dd * (drdT_pY + drdT_pa + drdT_pg);
        }
    }

    template <typename T>
    inline
    void rate_Mg24_to_C12_C12_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pY = rate_eval.screened_rates(k_Na23_p_to_C12_C12_reaclib);
        Real r_pa = rate_eval.screened_rates(k_Na23_p_to_He4_Ne20_reaclib);
        Real r_pg = rate_eval.screened_rates(k_Na23_p_to_Mg24_reaclib);
        Real dd = 1.0_rt / (r_pY + r_pa + r_pg);
        Real r_gY = 0.0_rt;
        Real r_gp = rate_eval.screened_rates(k_Mg24_to_p_Na23_reaclib);
        rate = r_gY + r_pY * r_gp * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_pY = rate_eval.dscreened_rates_dT(k_Na23_p_to_C12_C12_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_Na23_p_to_He4_Ne20_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_Na23_p_to_Mg24_reaclib);
            Real drdT_gY = 0.0_rt;
            Real drdT_gp = rate_eval.dscreened_rates_dT(k_Mg24_to_p_Na23_reaclib);
            drate_dT = drdT_gY + drdT_pY * r_gp * dd + r_pY * drdT_gp * dd - r_pY * r_gp * dd * dd * (drdT_pY + drdT_pa + drdT_pg);
        }
    }

    template <typename T>
    inline
    void rate_C12_C12_to_Ne20_He4_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pY = rate_eval.screened_rates(k_Na23_p_to_C12_C12_reaclib);
        Real r_pa = rate_eval.screened_rates(k_Na23_p_to_He4_Ne20_reaclib);
        Real r_pg = rate_eval.screened_rates(k_Na23_p_to_Mg24_reaclib);
        Real dd = 1.0_rt / (r_pY + r_pa + r_pg);
        Real r_Ya = rate_eval.screened_rates(k_C12_C12_to_He4_Ne20_reaclib);
        Real r_Yp = rate_eval.screened_rates(k_C12_C12_to_p_Na23_reaclib);
        rate = r_Ya + r_Yp * r_pa * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_pY = rate_eval.dscreened_rates_dT(k_Na23_p_to_C12_C12_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_Na23_p_to_He4_Ne20_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_Na23_p_to_Mg24_reaclib);
            Real drdT_Ya = rate_eval.dscreened_rates_dT(k_C12_C12_to_He4_Ne20_reaclib);
            Real drdT_Yp = rate_eval.dscreened_rates_dT(k_C12_C12_to_p_Na23_reaclib);
            drate_dT = drdT_Ya + drdT_Yp * r_pa * dd + r_Ya * drdT_pa * dd - r_Yp * r_pa * dd * dd * (drdT_pY + drdT_pa + drdT_pg);
        }
    }

    template <typename T>
    inline
    void rate_Ne20_He4_to_C12_C12_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pY = rate_eval.screened_rates(k_Na23_p_to_C12_C12_reaclib);
        Real r_pa = rate_eval.screened_rates(k_Na23_p_to_He4_Ne20_reaclib);
        Real r_pg = rate_eval.screened_rates(k_Na23_p_to_Mg24_reaclib);
        Real dd = 1.0_rt / (r_pY + r_pa + r_pg);
        Real r_aY = rate_eval.screened_rates(k_Ne20_He4_to_C12_C12_reaclib);
        Real r_ap = rate_eval.screened_rates(k_Ne20_He4_to_p_Na23_reaclib);
        rate = r_aY + r_pY * r_ap * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_pY = rate_eval.dscreened_rates_dT(k_Na23_p_to_C12_C12_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_Na23_p_to_He4_Ne20_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_Na23_p_to_Mg24_reaclib);
            Real drdT_aY = rate_eval.dscreened_rates_dT(k_Ne20_He4_to_C12_C12_reaclib);
            Real drdT_ap = rate_eval.dscreened_rates_dT(k_Ne20_He4_to_p_Na23_reaclib);
            drate_dT = drdT_aY + drdT_pY * r_ap * dd + r_pY * drdT_ap * dd - r_pY * r_ap * dd * dd * (drdT_pY + drdT_pa + drdT_pg);
        }
    }

    template <typename T>
    inline
    void rate_Ne20_He4_to_Mg24_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pg = rate_eval.screened_rates(k_Na23_p_to_Mg24_reaclib);
        Real r_pa = rate_eval.screened_rates(k_Na23_p_to_He4_Ne20_reaclib);
        Real r_pY = rate_eval.screened_rates(k_Na23_p_to_C12_C12_reaclib);
        Real dd = 1.0_rt / (r_pg + r_pa + r_pY);
        Real r_ag = rate_eval.screened_rates(k_Ne20_He4_to_Mg24_reaclib);
        Real r_ap = rate_eval.screened_rates(k_Ne20_He4_to_p_Na23_reaclib);
        rate = r_ag + r_ap * r_pg * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_ag = rate_eval.dscreened_rates_dT(k_Ne20_He4_to_Mg24_reaclib);
            Real drdT_ap = rate_eval.dscreened_rates_dT(k_Ne20_He4_to_p_Na23_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_Na23_p_to_Mg24_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_Na23_p_to_He4_Ne20_reaclib);
            Real drdT_pY = rate_eval.dscreened_rates_dT(k_Na23_p_to_C12_C12_reaclib);
            drate_dT = drdT_ag + drdT_ap * r_pg * dd + r_ap * drdT_pg * dd - r_ap * r_pg * dd * dd * (drdT_pg + drdT_pa + drdT_pY);
        }
    }

    template <typename T>
    inline
    void rate_Mg24_to_Ne20_He4_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pg = rate_eval.screened_rates(k_Na23_p_to_Mg24_reaclib);
        Real r_pa = rate_eval.screened_rates(k_Na23_p_to_He4_Ne20_reaclib);
        Real r_pY = rate_eval.screened_rates(k_Na23_p_to_C12_C12_reaclib);
        Real dd = 1.0_rt / (r_pg + r_pa + r_pY);
        Real r_ga = rate_eval.screened_rates(k_Mg24_to_He4_Ne20_reaclib);
        Real r_gp = rate_eval.screened_rates(k_Mg24_to_p_Na23_reaclib);
        rate = r_ga + r_gp * r_pa * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_ga = rate_eval.dscreened_rates_dT(k_Mg24_to_He4_Ne20_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_Na23_p_to_He4_Ne20_reaclib);
            Real drdT_gp = rate_eval.dscreened_rates_dT(k_Mg24_to_p_Na23_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_Na23_p_to_Mg24_reaclib);
            Real drdT_pY = rate_eval.dscreened_rates_dT(k_Na23_p_to_C12_C12_reaclib);
            drate_dT = drdT_ga + drdT_gp * r_pa * dd + r_gp * drdT_pa * dd - r_gp * r_pa * dd * dd * (drdT_pg + drdT_pa + drdT_pY);
        }
    }

    template <typename T>
    inline
    void rate_C12_O16_to_Si28_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pY = rate_eval.screened_rates(k_Al27_p_to_C12_O16_reaclib);
        Real r_pa = rate_eval.screened_rates(k_Al27_p_to_He4_Mg24_reaclib);
        Real r_pg = rate_eval.screened_rates(k_Al27_p_to_Si28_reaclib);
        Real dd = 1.0_rt / (r_pY + r_pa + r_pg);
        Real r_Yg = 0.0_rt;
        Real r_Yp = rate_eval.screened_rates(k_O16_C12_to_p_Al27_reaclib);
        rate = r_Yg + r_Yp * r_pg * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_pY = rate_eval.dscreened_rates_dT(k_Al27_p_to_C12_O16_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_Al27_p_to_He4_Mg24_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_Al27_p_to_Si28_reaclib);
            Real drdT_Yg = 0.0_rt;
            Real drdT_Yp = rate_eval.dscreened_rates_dT(k_O16_C12_to_p_Al27_reaclib);
            drate_dT = drdT_Yg + drdT_Yp * r_pg * dd + r_Yp * drdT_pg * dd - r_Yp * r_pg * dd * dd * (drdT_pY + drdT_pa + drdT_pg);
        }
    }

    template <typename T>
    inline
    void rate_Si28_to_O16_C12_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pY = rate_eval.screened_rates(k_Al27_p_to_C12_O16_reaclib);
        Real r_pa = rate_eval.screened_rates(k_Al27_p_to_He4_Mg24_reaclib);
        Real r_pg = rate_eval.screened_rates(k_Al27_p_to_Si28_reaclib);
        Real dd = 1.0_rt / (r_pY + r_pa + r_pg);
        Real r_gY = 0.0_rt;
        Real r_gp = rate_eval.screened_rates(k_Si28_to_p_Al27_reaclib);
        rate = r_gY + r_pY * r_gp * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_pY = rate_eval.dscreened_rates_dT(k_Al27_p_to_C12_O16_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_Al27_p_to_He4_Mg24_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_Al27_p_to_Si28_reaclib);
            Real drdT_gY = 0.0_rt;
            Real drdT_gp = rate_eval.dscreened_rates_dT(k_Si28_to_p_Al27_reaclib);
            drate_dT = drdT_gY + drdT_pY * r_gp * dd + r_pY * drdT_gp * dd - r_pY * r_gp * dd * dd * (drdT_pY + drdT_pa + drdT_pg);
        }
    }

    template <typename T>
    inline
    void rate_C12_O16_to_Mg24_He4_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pY = rate_eval.screened_rates(k_Al27_p_to_C12_O16_reaclib);
        Real r_pa = rate_eval.screened_rates(k_Al27_p_to_He4_Mg24_reaclib);
        Real r_pg = rate_eval.screened_rates(k_Al27_p_to_Si28_reaclib);
        Real dd = 1.0_rt / (r_pY + r_pa + r_pg);
        Real r_Ya = rate_eval.screened_rates(k_O16_C12_to_He4_Mg24_reaclib);
        Real r_Yp = rate_eval.screened_rates(k_O16_C12_to_p_Al27_reaclib);
        rate = r_Ya + r_Yp * r_pa * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_pY = rate_eval.dscreened_rates_dT(k_Al27_p_to_C12_O16_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_Al27_p_to_He4_Mg24_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_Al27_p_to_Si28_reaclib);
            Real drdT_Ya = rate_eval.dscreened_rates_dT(k_O16_C12_to_He4_Mg24_reaclib);
            Real drdT_Yp = rate_eval.dscreened_rates_dT(k_O16_C12_to_p_Al27_reaclib);
            drate_dT = drdT_Ya + drdT_Yp * r_pa * dd + r_Ya * drdT_pa * dd - r_Yp * r_pa * dd * dd * (drdT_pY + drdT_pa + drdT_pg);
        }
    }

    template <typename T>
    inline
    void rate_Mg24_He4_to_O16_C12_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pY = rate_eval.screened_rates(k_Al27_p_to_C12_O16_reaclib);
        Real r_pa = rate_eval.screened_rates(k_Al27_p_to_He4_Mg24_reaclib);
        Real r_pg = rate_eval.screened_rates(k_Al27_p_to_Si28_reaclib);
        Real dd = 1.0_rt / (r_pY + r_pa + r_pg);
        Real r_aY = rate_eval.screened_rates(k_Mg24_He4_to_C12_O16_reaclib);
        Real r_ap = rate_eval.screened_rates(k_Mg24_He4_to_p_Al27_reaclib);
        rate = r_aY + r_pY * r_ap * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_pY = rate_eval.dscreened_rates_dT(k_Al27_p_to_C12_O16_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_Al27_p_to_He4_Mg24_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_Al27_p_to_Si28_reaclib);
            Real drdT_aY = rate_eval.dscreened_rates_dT(k_Mg24_He4_to_C12_O16_reaclib);
            Real drdT_ap = rate_eval.dscreened_rates_dT(k_Mg24_He4_to_p_Al27_reaclib);
            drate_dT = drdT_aY + drdT_pY * r_ap * dd + r_pY * drdT_ap * dd - r_pY * r_ap * dd * dd * (drdT_pY + drdT_pa + drdT_pg);
        }
    }

    template <typename T>
    inline
    void rate_Mg24_He4_to_Si28_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pg = rate_eval.screened_rates(k_Al27_p_to_Si28_reaclib);
        Real r_pa = rate_eval.screened_rates(k_Al27_p_to_He4_Mg24_reaclib);
        Real r_pY = rate_eval.screened_rates(k_Al27_p_to_C12_O16_reaclib);
        Real dd = 1.0_rt / (r_pg + r_pa + r_pY);
        Real r_ag = rate_eval.screened_rates(k_Mg24_He4_to_Si28_reaclib);
        Real r_ap = rate_eval.screened_rates(k_Mg24_He4_to_p_Al27_reaclib);
        rate = r_ag + r_ap * r_pg * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_ag = rate_eval.dscreened_rates_dT(k_Mg24_He4_to_Si28_reaclib);
            Real drdT_ap = rate_eval.dscreened_rates_dT(k_Mg24_He4_to_p_Al27_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_Al27_p_to_Si28_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_Al27_p_to_He4_Mg24_reaclib);
            Real drdT_pY = rate_eval.dscreened_rates_dT(k_Al27_p_to_C12_O16_reaclib);
            drate_dT = drdT_ag + drdT_ap * r_pg * dd + r_ap * drdT_pg * dd - r_ap * r_pg * dd * dd * (drdT_pg + drdT_pa + drdT_pY);
        }
    }

    template <typename T>
    inline
    void rate_Si28_to_Mg24_He4_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pg = rate_eval.screened_rates(k_Al27_p_to_Si28_reaclib);
        Real r_pa = rate_eval.screened_rates(k_Al27_p_to_He4_Mg24_reaclib);
        Real r_pY = rate_eval.screened_rates(k_Al27_p_to_C12_O16_reaclib);
        Real dd = 1.0_rt / (r_pg + r_pa + r_pY);
        Real r_ga = rate_eval.screened_rates(k_Si28_to_He4_Mg24_reaclib);
        Real r_gp = rate_eval.screened_rates(k_Si28_to_p_Al27_reaclib);
        rate = r_ga + r_gp * r_pa * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_ga = rate_eval.dscreened_rates_dT(k_Si28_to_He4_Mg24_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_Al27_p_to_He4_Mg24_reaclib);
            Real drdT_gp = rate_eval.dscreened_rates_dT(k_Si28_to_p_Al27_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_Al27_p_to_Si28_reaclib);
            Real drdT_pY = rate_eval.dscreened_rates_dT(k_Al27_p_to_C12_O16_reaclib);
            drate_dT = drdT_ga + drdT_gp * r_pa * dd + r_gp * drdT_pa * dd - r_gp * r_pa * dd * dd * (drdT_pg + drdT_pa + drdT_pY);
        }
    }

    template <typename T>
    inline
    void rate_O16_O16_to_S32_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pY = rate_eval.screened_rates(k_P31_p_to_O16_O16_reaclib);
        Real r_pa = rate_eval.screened_rates(k_P31_p_to_He4_Si28_reaclib);
        Real r_pg = rate_eval.screened_rates(k_P31_p_to_S32_reaclib);
        Real dd = 1.0_rt / (r_pY + r_pa + r_pg);
        Real r_Yg = 0.0_rt;
        Real r_Yp = rate_eval.screened_rates(k_O16_O16_to_p_P31_reaclib);
        rate = r_Yg + r_Yp * r_pg * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_pY = rate_eval.dscreened_rates_dT(k_P31_p_to_O16_O16_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_P31_p_to_He4_Si28_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_P31_p_to_S32_reaclib);
            Real drdT_Yg = 0.0_rt;
            Real drdT_Yp = rate_eval.dscreened_rates_dT(k_O16_O16_to_p_P31_reaclib);
            drate_dT = drdT_Yg + drdT_Yp * r_pg * dd + r_Yp * drdT_pg * dd - r_Yp * r_pg * dd * dd * (drdT_pY + drdT_pa + drdT_pg);
        }
    }

    template <typename T>
    inline
    void rate_S32_to_O16_O16_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pY = rate_eval.screened_rates(k_P31_p_to_O16_O16_reaclib);
        Real r_pa = rate_eval.screened_rates(k_P31_p_to_He4_Si28_reaclib);
        Real r_pg = rate_eval.screened_rates(k_P31_p_to_S32_reaclib);
        Real dd = 1.0_rt / (r_pY + r_pa + r_pg);
        Real r_gY = 0.0_rt;
        Real r_gp = rate_eval.screened_rates(k_S32_to_p_P31_reaclib);
        rate = r_gY + r_pY * r_gp * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_pY = rate_eval.dscreened_rates_dT(k_P31_p_to_O16_O16_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_P31_p_to_He4_Si28_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_P31_p_to_S32_reaclib);
            Real drdT_gY = 0.0_rt;
            Real drdT_gp = rate_eval.dscreened_rates_dT(k_S32_to_p_P31_reaclib);
            drate_dT = drdT_gY + drdT_pY * r_gp * dd + r_pY * drdT_gp * dd - r_pY * r_gp * dd * dd * (drdT_pY + drdT_pa + drdT_pg);
        }
    }

    template <typename T>
    inline
    void rate_O16_O16_to_Si28_He4_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pY = rate_eval.screened_rates(k_P31_p_to_O16_O16_reaclib);
        Real r_pa = rate_eval.screened_rates(k_P31_p_to_He4_Si28_reaclib);
        Real r_pg = rate_eval.screened_rates(k_P31_p_to_S32_reaclib);
        Real dd = 1.0_rt / (r_pY + r_pa + r_pg);
        Real r_Ya = rate_eval.screened_rates(k_O16_O16_to_He4_Si28_reaclib);
        Real r_Yp = rate_eval.screened_rates(k_O16_O16_to_p_P31_reaclib);
        rate = r_Ya + r_Yp * r_pa * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_pY = rate_eval.dscreened_rates_dT(k_P31_p_to_O16_O16_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_P31_p_to_He4_Si28_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_P31_p_to_S32_reaclib);
            Real drdT_Ya = rate_eval.dscreened_rates_dT(k_O16_O16_to_He4_Si28_reaclib);
            Real drdT_Yp = rate_eval.dscreened_rates_dT(k_O16_O16_to_p_P31_reaclib);
            drate_dT = drdT_Ya + drdT_Yp * r_pa * dd + r_Ya * drdT_pa * dd - r_Yp * r_pa * dd * dd * (drdT_pY + drdT_pa + drdT_pg);
        }
    }

    template <typename T>
    inline
    void rate_Si28_He4_to_O16_O16_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pY = rate_eval.screened_rates(k_P31_p_to_O16_O16_reaclib);
        Real r_pa = rate_eval.screened_rates(k_P31_p_to_He4_Si28_reaclib);
        Real r_pg = rate_eval.screened_rates(k_P31_p_to_S32_reaclib);
        Real dd = 1.0_rt / (r_pY + r_pa + r_pg);
        Real r_aY = rate_eval.screened_rates(k_Si28_He4_to_O16_O16_reaclib);
        Real r_ap = rate_eval.screened_rates(k_Si28_He4_to_p_P31_reaclib);
        rate = r_aY + r_pY * r_ap * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_pY = rate_eval.dscreened_rates_dT(k_P31_p_to_O16_O16_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_P31_p_to_He4_Si28_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_P31_p_to_S32_reaclib);
            Real drdT_aY = rate_eval.dscreened_rates_dT(k_Si28_He4_to_O16_O16_reaclib);
            Real drdT_ap = rate_eval.dscreened_rates_dT(k_Si28_He4_to_p_P31_reaclib);
            drate_dT = drdT_aY + drdT_pY * r_ap * dd + r_pY * drdT_ap * dd - r_pY * r_ap * dd * dd * (drdT_pY + drdT_pa + drdT_pg);
        }
    }

    template <typename T>
    inline
    void rate_Si28_He4_to_S32_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pg = rate_eval.screened_rates(k_P31_p_to_S32_reaclib);
        Real r_pa = rate_eval.screened_rates(k_P31_p_to_He4_Si28_reaclib);
        Real r_pY = rate_eval.screened_rates(k_P31_p_to_O16_O16_reaclib);
        Real dd = 1.0_rt / (r_pg + r_pa + r_pY);
        Real r_ag = rate_eval.screened_rates(k_Si28_He4_to_S32_reaclib);
        Real r_ap = rate_eval.screened_rates(k_Si28_He4_to_p_P31_reaclib);
        rate = r_ag + r_ap * r_pg * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_ag = rate_eval.dscreened_rates_dT(k_Si28_He4_to_S32_reaclib);
            Real drdT_ap = rate_eval.dscreened_rates_dT(k_Si28_He4_to_p_P31_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_P31_p_to_S32_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_P31_p_to_He4_Si28_reaclib);
            Real drdT_pY = rate_eval.dscreened_rates_dT(k_P31_p_to_O16_O16_reaclib);
            drate_dT = drdT_ag + drdT_ap * r_pg * dd + r_ap * drdT_pg * dd - r_ap * r_pg * dd * dd * (drdT_pg + drdT_pa + drdT_pY);
        }
    }

    template <typename T>
    inline
    void rate_S32_to_Si28_He4_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pg = rate_eval.screened_rates(k_P31_p_to_S32_reaclib);
        Real r_pa = rate_eval.screened_rates(k_P31_p_to_He4_Si28_reaclib);
        Real r_pY = rate_eval.screened_rates(k_P31_p_to_O16_O16_reaclib);
        Real dd = 1.0_rt / (r_pg + r_pa + r_pY);
        Real r_ga = rate_eval.screened_rates(k_S32_to_He4_Si28_reaclib);
        Real r_gp = rate_eval.screened_rates(k_S32_to_p_P31_reaclib);
        rate = r_ga + r_gp * r_pa * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_ga = rate_eval.dscreened_rates_dT(k_S32_to_He4_Si28_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_P31_p_to_He4_Si28_reaclib);
            Real drdT_gp = rate_eval.dscreened_rates_dT(k_S32_to_p_P31_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_P31_p_to_S32_reaclib);
            Real drdT_pY = rate_eval.dscreened_rates_dT(k_P31_p_to_O16_O16_reaclib);
            drate_dT = drdT_ga + drdT_gp * r_pa * dd + r_gp * drdT_pa * dd - r_gp * r_pa * dd * dd * (drdT_pg + drdT_pa + drdT_pY);
        }
    }

    template <typename T>
    inline
    void rate_S32_He4_to_Ar36_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pg = rate_eval.screened_rates(k_Cl35_p_to_Ar36_reaclib);
        Real r_pa = rate_eval.screened_rates(k_Cl35_p_to_He4_S32_reaclib);
        Real r_pY = 0.0_rt;
        Real dd = 1.0_rt / (r_pg + r_pa + r_pY);
        Real r_ag = rate_eval.screened_rates(k_S32_He4_to_Ar36_reaclib);
        Real r_ap = rate_eval.screened_rates(k_S32_He4_to_p_Cl35_reaclib);
        rate = r_ag + r_ap * r_pg * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_ag = rate_eval.dscreened_rates_dT(k_S32_He4_to_Ar36_reaclib);
            Real drdT_ap = rate_eval.dscreened_rates_dT(k_S32_He4_to_p_Cl35_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_Cl35_p_to_Ar36_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_Cl35_p_to_He4_S32_reaclib);
            Real drdT_pY = 0.0_rt;
            drate_dT = drdT_ag + drdT_ap * r_pg * dd + r_ap * drdT_pg * dd - r_ap * r_pg * dd * dd * (drdT_pg + drdT_pa + drdT_pY);
        }
    }

    template <typename T>
    inline
    void rate_Ar36_to_S32_He4_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pg = rate_eval.screened_rates(k_Cl35_p_to_Ar36_reaclib);
        Real r_pa = rate_eval.screened_rates(k_Cl35_p_to_He4_S32_reaclib);
        Real r_pY = 0.0_rt;
        Real dd = 1.0_rt / (r_pg + r_pa + r_pY);
        Real r_ga = rate_eval.screened_rates(k_Ar36_to_He4_S32_reaclib);
        Real r_gp = rate_eval.screened_rates(k_Ar36_to_p_Cl35_reaclib);
        rate = r_ga + r_gp * r_pa * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_ga = rate_eval.dscreened_rates_dT(k_Ar36_to_He4_S32_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_Cl35_p_to_He4_S32_reaclib);
            Real drdT_gp = rate_eval.dscreened_rates_dT(k_Ar36_to_p_Cl35_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_Cl35_p_to_Ar36_reaclib);
            Real drdT_pY = 0.0_rt;
            drate_dT = drdT_ga + drdT_gp * r_pa * dd + r_gp * drdT_pa * dd - r_gp * r_pa * dd * dd * (drdT_pg + drdT_pa + drdT_pY);
        }
    }

    template <typename T>
    inline
    void rate_Ar36_He4_to_Ca40_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pg = rate_eval.screened_rates(k_K39_p_to_Ca40_reaclib);
        Real r_pa = rate_eval.screened_rates(k_K39_p_to_He4_Ar36_reaclib);
        Real r_pY = 0.0_rt;
        Real dd = 1.0_rt / (r_pg + r_pa + r_pY);
        Real r_ag = rate_eval.screened_rates(k_Ar36_He4_to_Ca40_reaclib);
        Real r_ap = rate_eval.screened_rates(k_Ar36_He4_to_p_K39_reaclib);
        rate = r_ag + r_ap * r_pg * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_ag = rate_eval.dscreened_rates_dT(k_Ar36_He4_to_Ca40_reaclib);
            Real drdT_ap = rate_eval.dscreened_rates_dT(k_Ar36_He4_to_p_K39_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_K39_p_to_Ca40_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_K39_p_to_He4_Ar36_reaclib);
            Real drdT_pY = 0.0_rt;
            drate_dT = drdT_ag + drdT_ap * r_pg * dd + r_ap * drdT_pg * dd - r_ap * r_pg * dd * dd * (drdT_pg + drdT_pa + drdT_pY);
        }
    }

    template <typename T>
    inline
    void rate_Ca40_to_Ar36_He4_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pg = rate_eval.screened_rates(k_K39_p_to_Ca40_reaclib);
        Real r_pa = rate_eval.screened_rates(k_K39_p_to_He4_Ar36_reaclib);
        Real r_pY = 0.0_rt;
        Real dd = 1.0_rt / (r_pg + r_pa + r_pY);
        Real r_ga = rate_eval.screened_rates(k_Ca40_to_He4_Ar36_reaclib);
        Real r_gp = rate_eval.screened_rates(k_Ca40_to_p_K39_reaclib);
        rate = r_ga + r_gp * r_pa * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_ga = rate_eval.dscreened_rates_dT(k_Ca40_to_He4_Ar36_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_K39_p_to_He4_Ar36_reaclib);
            Real drdT_gp = rate_eval.dscreened_rates_dT(k_Ca40_to_p_K39_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_K39_p_to_Ca40_reaclib);
            Real drdT_pY = 0.0_rt;
            drate_dT = drdT_ga + drdT_gp * r_pa * dd + r_gp * drdT_pa * dd - r_gp * r_pa * dd * dd * (drdT_pg + drdT_pa + drdT_pY);
        }
    }

    template <typename T>
    inline
    void rate_Ca40_He4_to_Ti44_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pg = rate_eval.screened_rates(k_Sc43_p_to_Ti44_reaclib);
        Real r_pa = rate_eval.screened_rates(k_Sc43_p_to_He4_Ca40_reaclib);
        Real r_pY = 0.0_rt;
        Real dd = 1.0_rt / (r_pg + r_pa + r_pY);
        Real r_ag = rate_eval.screened_rates(k_Ca40_He4_to_Ti44_reaclib);
        Real r_ap = rate_eval.screened_rates(k_Ca40_He4_to_p_Sc43_reaclib);
        rate = r_ag + r_ap * r_pg * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_ag = rate_eval.dscreened_rates_dT(k_Ca40_He4_to_Ti44_reaclib);
            Real drdT_ap = rate_eval.dscreened_rates_dT(k_Ca40_He4_to_p_Sc43_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_Sc43_p_to_Ti44_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_Sc43_p_to_He4_Ca40_reaclib);
            Real drdT_pY = 0.0_rt;
            drate_dT = drdT_ag + drdT_ap * r_pg * dd + r_ap * drdT_pg * dd - r_ap * r_pg * dd * dd * (drdT_pg + drdT_pa + drdT_pY);
        }
    }

    template <typename T>
    inline
    void rate_Ti44_to_Ca40_He4_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pg = rate_eval.screened_rates(k_Sc43_p_to_Ti44_reaclib);
        Real r_pa = rate_eval.screened_rates(k_Sc43_p_to_He4_Ca40_reaclib);
        Real r_pY = 0.0_rt;
        Real dd = 1.0_rt / (r_pg + r_pa + r_pY);
        Real r_ga = rate_eval.screened_rates(k_Ti44_to_He4_Ca40_reaclib);
        Real r_gp = rate_eval.screened_rates(k_Ti44_to_p_Sc43_reaclib);
        rate = r_ga + r_gp * r_pa * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_ga = rate_eval.dscreened_rates_dT(k_Ti44_to_He4_Ca40_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_Sc43_p_to_He4_Ca40_reaclib);
            Real drdT_gp = rate_eval.dscreened_rates_dT(k_Ti44_to_p_Sc43_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_Sc43_p_to_Ti44_reaclib);
            Real drdT_pY = 0.0_rt;
            drate_dT = drdT_ga + drdT_gp * r_pa * dd + r_gp * drdT_pa * dd - r_gp * r_pa * dd * dd * (drdT_pg + drdT_pa + drdT_pY);
        }
    }

    template <typename T>
    inline
    void rate_Ti44_He4_to_Cr48_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pg = rate_eval.screened_rates(k_V47_p_to_Cr48_reaclib);
        Real r_pa = rate_eval.screened_rates(k_V47_p_to_He4_Ti44_reaclib);
        Real r_pY = 0.0_rt;
        Real dd = 1.0_rt / (r_pg + r_pa + r_pY);
        Real r_ag = rate_eval.screened_rates(k_Ti44_He4_to_Cr48_reaclib);
        Real r_ap = rate_eval.screened_rates(k_Ti44_He4_to_p_V47_reaclib);
        rate = r_ag + r_ap * r_pg * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_ag = rate_eval.dscreened_rates_dT(k_Ti44_He4_to_Cr48_reaclib);
            Real drdT_ap = rate_eval.dscreened_rates_dT(k_Ti44_He4_to_p_V47_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_V47_p_to_Cr48_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_V47_p_to_He4_Ti44_reaclib);
            Real drdT_pY = 0.0_rt;
            drate_dT = drdT_ag + drdT_ap * r_pg * dd + r_ap * drdT_pg * dd - r_ap * r_pg * dd * dd * (drdT_pg + drdT_pa + drdT_pY);
        }
    }

    template <typename T>
    inline
    void rate_Cr48_to_Ti44_He4_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pg = rate_eval.screened_rates(k_V47_p_to_Cr48_reaclib);
        Real r_pa = rate_eval.screened_rates(k_V47_p_to_He4_Ti44_reaclib);
        Real r_pY = 0.0_rt;
        Real dd = 1.0_rt / (r_pg + r_pa + r_pY);
        Real r_ga = rate_eval.screened_rates(k_Cr48_to_He4_Ti44_reaclib);
        Real r_gp = rate_eval.screened_rates(k_Cr48_to_p_V47_reaclib);
        rate = r_ga + r_gp * r_pa * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_ga = rate_eval.dscreened_rates_dT(k_Cr48_to_He4_Ti44_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_V47_p_to_He4_Ti44_reaclib);
            Real drdT_gp = rate_eval.dscreened_rates_dT(k_Cr48_to_p_V47_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_V47_p_to_Cr48_reaclib);
            Real drdT_pY = 0.0_rt;
            drate_dT = drdT_ga + drdT_gp * r_pa * dd + r_gp * drdT_pa * dd - r_gp * r_pa * dd * dd * (drdT_pg + drdT_pa + drdT_pY);
        }
    }

    template <typename T>
    inline
    void rate_Cr48_He4_to_Fe52_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pg = rate_eval.screened_rates(k_Mn51_p_to_Fe52_reaclib);
        Real r_pa = rate_eval.screened_rates(k_Mn51_p_to_He4_Cr48_reaclib);
        Real r_pY = 0.0_rt;
        Real dd = 1.0_rt / (r_pg + r_pa + r_pY);
        Real r_ag = rate_eval.screened_rates(k_Cr48_He4_to_Fe52_reaclib);
        Real r_ap = rate_eval.screened_rates(k_Cr48_He4_to_p_Mn51_reaclib);
        rate = r_ag + r_ap * r_pg * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_ag = rate_eval.dscreened_rates_dT(k_Cr48_He4_to_Fe52_reaclib);
            Real drdT_ap = rate_eval.dscreened_rates_dT(k_Cr48_He4_to_p_Mn51_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_Mn51_p_to_Fe52_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_Mn51_p_to_He4_Cr48_reaclib);
            Real drdT_pY = 0.0_rt;
            drate_dT = drdT_ag + drdT_ap * r_pg * dd + r_ap * drdT_pg * dd - r_ap * r_pg * dd * dd * (drdT_pg + drdT_pa + drdT_pY);
        }
    }

    template <typename T>
    inline
    void rate_Fe52_to_Cr48_He4_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pg = rate_eval.screened_rates(k_Mn51_p_to_Fe52_reaclib);
        Real r_pa = rate_eval.screened_rates(k_Mn51_p_to_He4_Cr48_reaclib);
        Real r_pY = 0.0_rt;
        Real dd = 1.0_rt / (r_pg + r_pa + r_pY);
        Real r_ga = rate_eval.screened_rates(k_Fe52_to_He4_Cr48_reaclib);
        Real r_gp = rate_eval.screened_rates(k_Fe52_to_p_Mn51_reaclib);
        rate = r_ga + r_gp * r_pa * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_ga = rate_eval.dscreened_rates_dT(k_Fe52_to_He4_Cr48_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_Mn51_p_to_He4_Cr48_reaclib);
            Real drdT_gp = rate_eval.dscreened_rates_dT(k_Fe52_to_p_Mn51_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_Mn51_p_to_Fe52_reaclib);
            Real drdT_pY = 0.0_rt;
            drate_dT = drdT_ga + drdT_gp * r_pa * dd + r_gp * drdT_pa * dd - r_gp * r_pa * dd * dd * (drdT_pg + drdT_pa + drdT_pY);
        }
    }

    template <typename T>
    inline
    void rate_Fe52_He4_to_Ni56_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pg = rate_eval.screened_rates(k_Co55_p_to_Ni56_reaclib);
        Real r_pa = rate_eval.screened_rates(k_Co55_p_to_He4_Fe52_reaclib);
        Real r_pY = 0.0_rt;
        Real dd = 1.0_rt / (r_pg + r_pa + r_pY);
        Real r_ag = rate_eval.screened_rates(k_Fe52_He4_to_Ni56_reaclib);
        Real r_ap = rate_eval.screened_rates(k_Fe52_He4_to_p_Co55_reaclib);
        rate = r_ag + r_ap * r_pg * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_ag = rate_eval.dscreened_rates_dT(k_Fe52_He4_to_Ni56_reaclib);
            Real drdT_ap = rate_eval.dscreened_rates_dT(k_Fe52_He4_to_p_Co55_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_Co55_p_to_Ni56_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_Co55_p_to_He4_Fe52_reaclib);
            Real drdT_pY = 0.0_rt;
            drate_dT = drdT_ag + drdT_ap * r_pg * dd + r_ap * drdT_pg * dd - r_ap * r_pg * dd * dd * (drdT_pg + drdT_pa + drdT_pY);
        }
    }

    template <typename T>
    inline
    void rate_Ni56_to_Fe52_He4_approx(const T& rate_eval, Real& rate, Real& drate_dT) {

        Real r_pg = rate_eval.screened_rates(k_Co55_p_to_Ni56_reaclib);
        Real r_pa = rate_eval.screened_rates(k_Co55_p_to_He4_Fe52_reaclib);
        Real r_pY = 0.0_rt;
        Real dd = 1.0_rt / (r_pg + r_pa + r_pY);
        Real r_ga = rate_eval.screened_rates(k_Ni56_to_He4_Fe52_reaclib);
        Real r_gp = rate_eval.screened_rates(k_Ni56_to_p_Co55_reaclib);
        rate = r_ga + r_gp * r_pa * dd;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            Real drdT_ga = rate_eval.dscreened_rates_dT(k_Ni56_to_He4_Fe52_reaclib);
            Real drdT_pa = rate_eval.dscreened_rates_dT(k_Co55_p_to_He4_Fe52_reaclib);
            Real drdT_gp = rate_eval.dscreened_rates_dT(k_Ni56_to_p_Co55_reaclib);
            Real drdT_pg = rate_eval.dscreened_rates_dT(k_Co55_p_to_Ni56_reaclib);
            Real drdT_pY = 0.0_rt;
            drate_dT = drdT_ga + drdT_gp * r_pa * dd + r_gp * drdT_pa * dd - r_gp * r_pa * dd * dd * (drdT_pg + drdT_pa + drdT_pY);
        }
    }



    template <int do_T_derivatives, typename T>
    inline
    void
    fill_rates([[maybe_unused]] const tf_t& tfactors,
               [[maybe_unused]] const Real rho,
               [[maybe_unused]] const Array1D<Real, 1, NumSpec>& Y,
               [[maybe_unused]] T& rate_eval)
    {

        [[maybe_unused]] Real rate{};
        [[maybe_unused]] Real drate_dT{};

        rate_C12_C12_to_Mg24_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_C12_C12_to_Mg24_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_C12_C12_to_Mg24_approx) = drate_dT;
        }

        rate_Mg24_to_C12_C12_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_Mg24_to_C12_C12_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_Mg24_to_C12_C12_approx) = drate_dT;
        }

        rate_C12_C12_to_Ne20_He4_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_C12_C12_to_Ne20_He4_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_C12_C12_to_Ne20_He4_approx) = drate_dT;
        }

        rate_Ne20_He4_to_C12_C12_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_Ne20_He4_to_C12_C12_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_Ne20_He4_to_C12_C12_approx) = drate_dT;
        }

        rate_Ne20_He4_to_Mg24_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_Ne20_He4_to_Mg24_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_Ne20_He4_to_Mg24_approx) = drate_dT;
        }

        rate_Mg24_to_Ne20_He4_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_Mg24_to_Ne20_He4_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_Mg24_to_Ne20_He4_approx) = drate_dT;
        }

        rate_C12_O16_to_Si28_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_C12_O16_to_Si28_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_C12_O16_to_Si28_approx) = drate_dT;
        }

        rate_Si28_to_O16_C12_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_Si28_to_O16_C12_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_Si28_to_O16_C12_approx) = drate_dT;
        }

        rate_C12_O16_to_Mg24_He4_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_C12_O16_to_Mg24_He4_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_C12_O16_to_Mg24_He4_approx) = drate_dT;
        }

        rate_Mg24_He4_to_O16_C12_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_Mg24_He4_to_O16_C12_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_Mg24_He4_to_O16_C12_approx) = drate_dT;
        }

        rate_Mg24_He4_to_Si28_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_Mg24_He4_to_Si28_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_Mg24_He4_to_Si28_approx) = drate_dT;
        }

        rate_Si28_to_Mg24_He4_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_Si28_to_Mg24_He4_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_Si28_to_Mg24_He4_approx) = drate_dT;
        }

        rate_O16_O16_to_S32_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_O16_O16_to_S32_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_O16_O16_to_S32_approx) = drate_dT;
        }

        rate_S32_to_O16_O16_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_S32_to_O16_O16_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_S32_to_O16_O16_approx) = drate_dT;
        }

        rate_O16_O16_to_Si28_He4_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_O16_O16_to_Si28_He4_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_O16_O16_to_Si28_He4_approx) = drate_dT;
        }

        rate_Si28_He4_to_O16_O16_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_Si28_He4_to_O16_O16_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_Si28_He4_to_O16_O16_approx) = drate_dT;
        }

        rate_Si28_He4_to_S32_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_Si28_He4_to_S32_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_Si28_He4_to_S32_approx) = drate_dT;
        }

        rate_S32_to_Si28_He4_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_S32_to_Si28_He4_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_S32_to_Si28_He4_approx) = drate_dT;
        }

        rate_S32_He4_to_Ar36_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_S32_He4_to_Ar36_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_S32_He4_to_Ar36_approx) = drate_dT;
        }

        rate_Ar36_to_S32_He4_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_Ar36_to_S32_He4_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_Ar36_to_S32_He4_approx) = drate_dT;
        }

        rate_Ar36_He4_to_Ca40_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_Ar36_He4_to_Ca40_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_Ar36_He4_to_Ca40_approx) = drate_dT;
        }

        rate_Ca40_to_Ar36_He4_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_Ca40_to_Ar36_He4_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_Ca40_to_Ar36_He4_approx) = drate_dT;
        }

        rate_Ca40_He4_to_Ti44_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_Ca40_He4_to_Ti44_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_Ca40_He4_to_Ti44_approx) = drate_dT;
        }

        rate_Ti44_to_Ca40_He4_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_Ti44_to_Ca40_He4_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_Ti44_to_Ca40_He4_approx) = drate_dT;
        }

        rate_Ti44_He4_to_Cr48_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_Ti44_He4_to_Cr48_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_Ti44_He4_to_Cr48_approx) = drate_dT;
        }

        rate_Cr48_to_Ti44_He4_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_Cr48_to_Ti44_He4_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_Cr48_to_Ti44_He4_approx) = drate_dT;
        }

        rate_Cr48_He4_to_Fe52_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_Cr48_He4_to_Fe52_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_Cr48_He4_to_Fe52_approx) = drate_dT;
        }

        rate_Fe52_to_Cr48_He4_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_Fe52_to_Cr48_He4_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_Fe52_to_Cr48_He4_approx) = drate_dT;
        }

        rate_Fe52_He4_to_Ni56_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_Fe52_He4_to_Ni56_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_Fe52_He4_to_Ni56_approx) = drate_dT;
        }

        rate_Ni56_to_Fe52_He4_approx<T>(rate_eval, rate, drate_dT);
        rate_eval.screened_rates(k_Ni56_to_Fe52_He4_approx) = rate;
        if constexpr (std::is_same_v<T, rate_derivs_t>) {
            rate_eval.dscreened_rates_dT(k_Ni56_to_Fe52_He4_approx) = drate_dT;
        }


    }

}
#endif
