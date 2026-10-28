#ifndef REACLIB_RATES_H
#define REACLIB_RATES_H

#include <amrex_bridge.H>

#include <tfactors.H>
#include <actual_network.H>
#include <rate_type.H>

namespace reaclib_rates {

    using namespace Rates;
    using namespace Species;

    template <int do_T_derivatives>
    inline
    void rate_O16_to_He4_C12_reaclib(const tf_t& tfactors, const Real log_scor, const Real dlog_scor_dT, Real& rate, Real& drate_dT) {

        // O16 --> He4 + C12

        rate = 0.0;
        drate_dT = 0.0;

        Real ln_set_rate{0.0};
        Real dln_set_rate_dT9{0.0};
        Real set_rate{0.0};

        // nac2 
        ln_set_rate =  94.3131 + -84.503 * tfactors.T9i + 58.9128 * tfactors.T913i + -148.273 * tfactors.T913
                             + 9.08324 * tfactors.T9 + -0.541041 * tfactors.T953 + 71.8554 * tfactors.lnT9;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  84.503 * tfactors.T9i * tfactors.T9i + -(1.0/3.0) * 58.9128 * tfactors.T943i + (1.0/3.0) * -148.273 * tfactors.T923i
                                      + 9.08324 + (5.0/3.0) * -0.541041 * tfactors.T923 + 71.8554 * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

        // nac2 
        ln_set_rate =  279.295 + -84.9515 * tfactors.T9i + 103.411 * tfactors.T913i + -420.567 * tfactors.T913
                             + 64.0874 * tfactors.T9 + -12.4624 * tfactors.T953 + 138.803 * tfactors.lnT9;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  84.9515 * tfactors.T9i * tfactors.T9i + -(1.0/3.0) * 103.411 * tfactors.T943i + (1.0/3.0) * -420.567 * tfactors.T923i
                                      + 64.0874 + (5.0/3.0) * -12.4624 * tfactors.T923 + 138.803 * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

    }

    template <int do_T_derivatives>
    inline
    void rate_Ne20_to_He4_O16_reaclib(const tf_t& tfactors, const Real log_scor, const Real dlog_scor_dT, Real& rate, Real& drate_dT) {

        // Ne20 --> He4 + O16

        rate = 0.0;
        drate_dT = 0.0;

        Real ln_set_rate{0.0};
        Real dln_set_rate_dT9{0.0};
        Real set_rate{0.0};

        // co10r
        ln_set_rate =  28.6431 + -65.246 * tfactors.T9i;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  65.246 * tfactors.T9i * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

        // co10n
        ln_set_rate =  48.6604 + -54.8875 * tfactors.T9i + -39.7262 * tfactors.T913i + -0.210799 * tfactors.T913
                             + 0.442879 * tfactors.T9 + -0.0797753 * tfactors.T953 + 0.833333 * tfactors.lnT9;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  54.8875 * tfactors.T9i * tfactors.T9i + -(1.0/3.0) * -39.7262 * tfactors.T943i + (1.0/3.0) * -0.210799 * tfactors.T923i
                                      + 0.442879 + (5.0/3.0) * -0.0797753 * tfactors.T923 + 0.833333 * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

        // co10r
        ln_set_rate =  34.2658 + -67.6518 * tfactors.T9i + -3.65925 * tfactors.T913
                             + 0.714224 * tfactors.T9 + -0.00107508 * tfactors.T953;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  67.6518 * tfactors.T9i * tfactors.T9i + (1.0/3.0) * -3.65925 * tfactors.T923i
                                      + 0.714224 + (5.0/3.0) * -0.00107508 * tfactors.T923;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

    }

    template <int do_T_derivatives>
    inline
    void rate_Mg24_to_He4_Ne20_reaclib(const tf_t& tfactors, const Real log_scor, const Real dlog_scor_dT, Real& rate, Real& drate_dT) {

        // Mg24 --> He4 + Ne20

        rate = 0.0;
        drate_dT = 0.0;

        Real ln_set_rate{0.0};
        Real dln_set_rate_dT9{0.0};
        Real set_rate{0.0};

        // il10r
        ln_set_rate =  16.0203 + -120.895 * tfactors.T9i + 16.9229 * tfactors.T913
                             + -2.57325 * tfactors.T9 + 0.208997 * tfactors.T953;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  120.895 * tfactors.T9i * tfactors.T9i + (1.0/3.0) * 16.9229 * tfactors.T923i
                                      + -2.57325 + (5.0/3.0) * 0.208997 * tfactors.T923;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

        // il10r
        ln_set_rate =  26.8017 + -117.334 * tfactors.T9i;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  117.334 * tfactors.T9i * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

        // il10r
        ln_set_rate =  -13.8869 + -110.62 * tfactors.T9i;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  110.62 * tfactors.T9i * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

        // il10n
        ln_set_rate =  49.3244 + -108.114 * tfactors.T9i + -46.2525 * tfactors.T913i + 5.58901 * tfactors.T913
                             + 7.61843 * tfactors.T9 + -3.683 * tfactors.T953 + 0.833333 * tfactors.lnT9;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  108.114 * tfactors.T9i * tfactors.T9i + -(1.0/3.0) * -46.2525 * tfactors.T943i + (1.0/3.0) * 5.58901 * tfactors.T923i
                                      + 7.61843 + (5.0/3.0) * -3.683 * tfactors.T923 + 0.833333 * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

    }

    template <int do_T_derivatives>
    inline
    void rate_C12_to_3He4_reaclib(const tf_t& tfactors, const Real log_scor, const Real dlog_scor_dT, Real& rate, Real& drate_dT) {

        // C12 --> 3 He4

        rate = 0.0;
        drate_dT = 0.0;

        Real ln_set_rate{0.0};
        Real dln_set_rate_dT9{0.0};
        Real set_rate{0.0};

        // fy05r
        ln_set_rate =  22.394 + -88.5493 * tfactors.T9i + -13.49 * tfactors.T913i + 21.4259 * tfactors.T913
                             + -1.34769 * tfactors.T9 + 0.0879816 * tfactors.T953 + -10.1653 * tfactors.lnT9;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  88.5493 * tfactors.T9i * tfactors.T9i + -(1.0/3.0) * -13.49 * tfactors.T943i + (1.0/3.0) * 21.4259 * tfactors.T923i
                                      + -1.34769 + (5.0/3.0) * 0.0879816 * tfactors.T923 + -10.1653 * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

        // fy05r
        ln_set_rate =  34.9561 + -85.4472 * tfactors.T9i + -23.57 * tfactors.T913i + 20.4886 * tfactors.T913
                             + -12.9882 * tfactors.T9 + -20.0 * tfactors.T953 + 0.83333 * tfactors.lnT9;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  85.4472 * tfactors.T9i * tfactors.T9i + -(1.0/3.0) * -23.57 * tfactors.T943i + (1.0/3.0) * 20.4886 * tfactors.T923i
                                      + -12.9882 + (5.0/3.0) * -20.0 * tfactors.T923 + 0.83333 * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

        // fy05n
        ln_set_rate =  45.7734 + -84.4227 * tfactors.T9i + -37.06 * tfactors.T913i + 29.3493 * tfactors.T913
                             + -115.507 * tfactors.T9 + -10.0 * tfactors.T953 + 1.66667 * tfactors.lnT9;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  84.4227 * tfactors.T9i * tfactors.T9i + -(1.0/3.0) * -37.06 * tfactors.T943i + (1.0/3.0) * 29.3493 * tfactors.T923i
                                      + -115.507 + (5.0/3.0) * -10.0 * tfactors.T923 + 1.66667 * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

    }

    template <int do_T_derivatives>
    inline
    void rate_C12_He4_to_O16_reaclib(const tf_t& tfactors, const Real log_scor, const Real dlog_scor_dT, Real& rate, Real& drate_dT) {

        // C12 + He4 --> O16

        rate = 0.0;
        drate_dT = 0.0;

        Real ln_set_rate{0.0};
        Real dln_set_rate_dT9{0.0};
        Real set_rate{0.0};

        // nac2 
        ln_set_rate =  254.634 + -1.84097 * tfactors.T9i + 103.411 * tfactors.T913i + -420.567 * tfactors.T913
                             + 64.0874 * tfactors.T9 + -12.4624 * tfactors.T953 + 137.303 * tfactors.lnT9;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  1.84097 * tfactors.T9i * tfactors.T9i + -(1.0/3.0) * 103.411 * tfactors.T943i + (1.0/3.0) * -420.567 * tfactors.T923i
                                      + 64.0874 + (5.0/3.0) * -12.4624 * tfactors.T923 + 137.303 * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

        // nac2 
        ln_set_rate =  69.6526 + -1.39254 * tfactors.T9i + 58.9128 * tfactors.T913i + -148.273 * tfactors.T913
                             + 9.08324 * tfactors.T9 + -0.541041 * tfactors.T953 + 70.3554 * tfactors.lnT9;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  1.39254 * tfactors.T9i * tfactors.T9i + -(1.0/3.0) * 58.9128 * tfactors.T943i + (1.0/3.0) * -148.273 * tfactors.T923i
                                      + 9.08324 + (5.0/3.0) * -0.541041 * tfactors.T923 + 70.3554 * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

    }

    template <int do_T_derivatives>
    inline
    void rate_O16_He4_to_Ne20_reaclib(const tf_t& tfactors, const Real log_scor, const Real dlog_scor_dT, Real& rate, Real& drate_dT) {

        // O16 + He4 --> Ne20

        rate = 0.0;
        drate_dT = 0.0;

        Real ln_set_rate{0.0};
        Real dln_set_rate_dT9{0.0};
        Real set_rate{0.0};

        // co10r
        ln_set_rate =  3.88571 + -10.3585 * tfactors.T9i
                             + -1.5 * tfactors.lnT9;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  10.3585 * tfactors.T9i * tfactors.T9i
                                      + -1.5 * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

        // co10n
        ln_set_rate =  23.903 + -39.7262 * tfactors.T913i + -0.210799 * tfactors.T913
                             + 0.442879 * tfactors.T9 + -0.0797753 * tfactors.T953 + -0.666667 * tfactors.lnT9;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  + -(1.0/3.0) * -39.7262 * tfactors.T943i + (1.0/3.0) * -0.210799 * tfactors.T923i
                                      + 0.442879 + (5.0/3.0) * -0.0797753 * tfactors.T923 + -0.666667 * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

        // co10r
        ln_set_rate =  9.50848 + -12.7643 * tfactors.T9i + -3.65925 * tfactors.T913
                             + 0.714224 * tfactors.T9 + -0.00107508 * tfactors.T953 + -1.5 * tfactors.lnT9;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  12.7643 * tfactors.T9i * tfactors.T9i + (1.0/3.0) * -3.65925 * tfactors.T923i
                                      + 0.714224 + (5.0/3.0) * -0.00107508 * tfactors.T923 + -1.5 * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

    }

    template <int do_T_derivatives>
    inline
    void rate_Ne20_He4_to_Mg24_reaclib(const tf_t& tfactors, const Real log_scor, const Real dlog_scor_dT, Real& rate, Real& drate_dT) {

        // Ne20 + He4 --> Mg24

        rate = 0.0;
        drate_dT = 0.0;

        Real ln_set_rate{0.0};
        Real dln_set_rate_dT9{0.0};
        Real set_rate{0.0};

        // il10r
        ln_set_rate =  -8.79827 + -12.7809 * tfactors.T9i + 16.9229 * tfactors.T913
                             + -2.57325 * tfactors.T9 + 0.208997 * tfactors.T953 + -1.5 * tfactors.lnT9;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  12.7809 * tfactors.T9i * tfactors.T9i + (1.0/3.0) * 16.9229 * tfactors.T923i
                                      + -2.57325 + (5.0/3.0) * 0.208997 * tfactors.T923 + -1.5 * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

        // il10r
        ln_set_rate =  1.98307 + -9.22026 * tfactors.T9i
                             + -1.5 * tfactors.lnT9;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  9.22026 * tfactors.T9i * tfactors.T9i
                                      + -1.5 * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

        // il10r
        ln_set_rate =  -38.7055 + -2.50605 * tfactors.T9i
                             + -1.5 * tfactors.lnT9;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  2.50605 * tfactors.T9i * tfactors.T9i
                                      + -1.5 * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

        // il10n
        ln_set_rate =  24.5058 + -46.2525 * tfactors.T913i + 5.58901 * tfactors.T913
                             + 7.61843 * tfactors.T9 + -3.683 * tfactors.T953 + -0.666667 * tfactors.lnT9;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  + -(1.0/3.0) * -46.2525 * tfactors.T943i + (1.0/3.0) * 5.58901 * tfactors.T923i
                                      + 7.61843 + (5.0/3.0) * -3.683 * tfactors.T923 + -0.666667 * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

    }

    template <int do_T_derivatives>
    inline
    void rate_C12_C12_to_He4_Ne20_reaclib(const tf_t& tfactors, const Real log_scor, const Real dlog_scor_dT, Real& rate, Real& drate_dT) {

        // C12 + C12 --> He4 + Ne20

        rate = 0.0;
        drate_dT = 0.0;

        Real ln_set_rate{0.0};
        Real dln_set_rate_dT9{0.0};
        Real set_rate{0.0};

        // cf88r
        ln_set_rate =  61.2863 + -84.165 * tfactors.T913i + -1.56627 * tfactors.T913
                             + -0.0736084 * tfactors.T9 + -0.072797 * tfactors.T953 + -0.666667 * tfactors.lnT9;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  + -(1.0/3.0) * -84.165 * tfactors.T943i + (1.0/3.0) * -1.56627 * tfactors.T923i
                                      + -0.0736084 + (5.0/3.0) * -0.072797 * tfactors.T923 + -0.666667 * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

    }

    template <int do_T_derivatives>
    inline
    void rate_O16_C12_to_He4_Mg24_reaclib(const tf_t& tfactors, const Real log_scor, const Real dlog_scor_dT, Real& rate, Real& drate_dT) {

        // O16 + C12 --> He4 + Mg24

        rate = 0.0;
        drate_dT = 0.0;

        Real ln_set_rate{0.0};
        Real dln_set_rate_dT9{0.0};
        Real set_rate{0.0};

        // cf88r
        ln_set_rate =  48.5341 + 0.37204 * tfactors.T9i + -133.413 * tfactors.T913i + 50.1572 * tfactors.T913
                             + -3.15987 * tfactors.T9 + 0.0178251 * tfactors.T953 + -23.7027 * tfactors.lnT9;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  -0.37204 * tfactors.T9i * tfactors.T9i + -(1.0/3.0) * -133.413 * tfactors.T943i + (1.0/3.0) * 50.1572 * tfactors.T923i
                                      + -3.15987 + (5.0/3.0) * 0.0178251 * tfactors.T923 + -23.7027 * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

    }

    template <int do_T_derivatives>
    inline
    void rate_Ne20_He4_to_C12_C12_reaclib(const tf_t& tfactors, const Real log_scor, const Real dlog_scor_dT, Real& rate, Real& drate_dT) {

        // Ne20 + He4 --> C12 + C12

        rate = 0.0;
        drate_dT = 0.0;

        Real ln_set_rate{0.0};
        Real dln_set_rate_dT9{0.0};
        Real set_rate{0.0};

        // cf88r
        ln_set_rate =  61.4748 + -53.6267 * tfactors.T9i + -84.165 * tfactors.T913i + -1.56627 * tfactors.T913
                             + -0.0736084 * tfactors.T9 + -0.072797 * tfactors.T953 + -0.666667 * tfactors.lnT9;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  53.6267 * tfactors.T9i * tfactors.T9i + -(1.0/3.0) * -84.165 * tfactors.T943i + (1.0/3.0) * -1.56627 * tfactors.T923i
                                      + -0.0736084 + (5.0/3.0) * -0.072797 * tfactors.T923 + -0.666667 * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

    }

    template <int do_T_derivatives>
    inline
    void rate_Mg24_He4_to_C12_O16_reaclib(const tf_t& tfactors, const Real log_scor, const Real dlog_scor_dT, Real& rate, Real& drate_dT) {

        // Mg24 + He4 --> C12 + O16

        rate = 0.0;
        drate_dT = 0.0;

        Real ln_set_rate{0.0};
        Real dln_set_rate_dT9{0.0};
        Real set_rate{0.0};

        // cf88r
        ln_set_rate =  49.5738 + -78.202 * tfactors.T9i + -133.413 * tfactors.T913i + 50.1572 * tfactors.T913
                             + -3.15987 * tfactors.T9 + 0.0178251 * tfactors.T953 + -23.7027 * tfactors.lnT9;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  78.202 * tfactors.T9i * tfactors.T9i + -(1.0/3.0) * -133.413 * tfactors.T943i + (1.0/3.0) * 50.1572 * tfactors.T923i
                                      + -3.15987 + (5.0/3.0) * 0.0178251 * tfactors.T923 + -23.7027 * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

    }

    template <int do_T_derivatives>
    inline
    void rate_He4_He4_He4_to_C12_reaclib(const tf_t& tfactors, const Real log_scor, const Real dlog_scor_dT, Real& rate, Real& drate_dT) {

        // 3 He4 --> C12

        rate = 0.0;
        drate_dT = 0.0;

        Real ln_set_rate{0.0};
        Real dln_set_rate_dT9{0.0};
        Real set_rate{0.0};

        // fy05r
        ln_set_rate =  -11.7884 + -1.02446 * tfactors.T9i + -23.57 * tfactors.T913i + 20.4886 * tfactors.T913
                             + -12.9882 * tfactors.T9 + -20.0 * tfactors.T953 + -2.16667 * tfactors.lnT9;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  1.02446 * tfactors.T9i * tfactors.T9i + -(1.0/3.0) * -23.57 * tfactors.T943i + (1.0/3.0) * 20.4886 * tfactors.T923i
                                      + -12.9882 + (5.0/3.0) * -20.0 * tfactors.T923 + -2.16667 * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

        // fy05n
        ln_set_rate =  -0.971052 + -37.06 * tfactors.T913i + 29.3493 * tfactors.T913
                             + -115.507 * tfactors.T9 + -10.0 * tfactors.T953 + -1.33333 * tfactors.lnT9;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  + -(1.0/3.0) * -37.06 * tfactors.T943i + (1.0/3.0) * 29.3493 * tfactors.T923i
                                      + -115.507 + (5.0/3.0) * -10.0 * tfactors.T923 + -1.33333 * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

        // fy05r
        ln_set_rate =  -24.3505 + -4.12656 * tfactors.T9i + -13.49 * tfactors.T913i + 21.4259 * tfactors.T913
                             + -1.34769 * tfactors.T9 + 0.0879816 * tfactors.T953 + -13.1653 * tfactors.lnT9;

        ln_set_rate += log_scor;

        if constexpr (do_T_derivatives) {
            dln_set_rate_dT9 =  4.12656 * tfactors.T9i * tfactors.T9i + -(1.0/3.0) * -13.49 * tfactors.T943i + (1.0/3.0) * 21.4259 * tfactors.T923i
                                      + -1.34769 + (5.0/3.0) * 0.0879816 * tfactors.T923 + -13.1653 * tfactors.T9i;

            dln_set_rate_dT9 += dlog_scor_dT * 1.0e9_rt;
        }

        // avoid underflows by zeroing rates in [0.0, 1.e-100]
        ln_set_rate = std::max(ln_set_rate, -230.0);
        set_rate = std::exp(ln_set_rate);
        rate += set_rate;
        if constexpr (do_T_derivatives) {
            drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
        }

    }


    template <int do_T_derivatives, typename T>
    inline
    void
    fill_rates([[maybe_unused]] const tf_t& tfactors,
               [[maybe_unused]] T& rate_eval)
    {

        [[maybe_unused]] Real rate;
        [[maybe_unused]] Real drate_dT;

        {
            // O16_to_He4_C12_reaclib

            Real log_scor {0.0_rt};
            Real dlog_scor_dT {0.0_rt};
            rate_O16_to_He4_C12_reaclib<do_T_derivatives>(tfactors, log_scor, dlog_scor_dT, rate, drate_dT);
            rate_eval.screened_rates(k_O16_to_He4_C12_reaclib) = rate;
            if constexpr (std::is_same_v<T, rate_derivs_t>) {
                rate_eval.dscreened_rates_dT(k_O16_to_He4_C12_reaclib) = drate_dT;
            }
        }

        {
            // Ne20_to_He4_O16_reaclib

            Real log_scor {0.0_rt};
            Real dlog_scor_dT {0.0_rt};
            rate_Ne20_to_He4_O16_reaclib<do_T_derivatives>(tfactors, log_scor, dlog_scor_dT, rate, drate_dT);
            rate_eval.screened_rates(k_Ne20_to_He4_O16_reaclib) = rate;
            if constexpr (std::is_same_v<T, rate_derivs_t>) {
                rate_eval.dscreened_rates_dT(k_Ne20_to_He4_O16_reaclib) = drate_dT;
            }
        }

        {
            // Mg24_to_He4_Ne20_reaclib

            Real log_scor {0.0_rt};
            Real dlog_scor_dT {0.0_rt};
            rate_Mg24_to_He4_Ne20_reaclib<do_T_derivatives>(tfactors, log_scor, dlog_scor_dT, rate, drate_dT);
            rate_eval.screened_rates(k_Mg24_to_He4_Ne20_reaclib) = rate;
            if constexpr (std::is_same_v<T, rate_derivs_t>) {
                rate_eval.dscreened_rates_dT(k_Mg24_to_He4_Ne20_reaclib) = drate_dT;
            }
        }

        {
            // C12_to_3He4_reaclib

            Real log_scor {0.0_rt};
            Real dlog_scor_dT {0.0_rt};
            rate_C12_to_3He4_reaclib<do_T_derivatives>(tfactors, log_scor, dlog_scor_dT, rate, drate_dT);
            rate_eval.screened_rates(k_C12_to_3He4_reaclib) = rate;
            if constexpr (std::is_same_v<T, rate_derivs_t>) {
                rate_eval.dscreened_rates_dT(k_C12_to_3He4_reaclib) = drate_dT;
            }
        }

        {
            // C12_He4_to_O16_reaclib

            Real log_scor {0.0_rt};
            Real dlog_scor_dT {0.0_rt};
#ifdef SCREENING
            log_scor = rate_eval.log_screen(k_He4_C12);
            if constexpr (std::is_same_v<T, rate_derivs_t>) {
                dlog_scor_dT = rate_eval.dlog_screen_dT(k_He4_C12);
            }
#endif
            rate_C12_He4_to_O16_reaclib<do_T_derivatives>(tfactors, log_scor, dlog_scor_dT, rate, drate_dT);
            rate_eval.screened_rates(k_C12_He4_to_O16_reaclib) = rate;
            if constexpr (std::is_same_v<T, rate_derivs_t>) {
                rate_eval.dscreened_rates_dT(k_C12_He4_to_O16_reaclib) = drate_dT;
            }
        }

        {
            // O16_He4_to_Ne20_reaclib

            Real log_scor {0.0_rt};
            Real dlog_scor_dT {0.0_rt};
#ifdef SCREENING
            log_scor = rate_eval.log_screen(k_He4_O16);
            if constexpr (std::is_same_v<T, rate_derivs_t>) {
                dlog_scor_dT = rate_eval.dlog_screen_dT(k_He4_O16);
            }
#endif
            rate_O16_He4_to_Ne20_reaclib<do_T_derivatives>(tfactors, log_scor, dlog_scor_dT, rate, drate_dT);
            rate_eval.screened_rates(k_O16_He4_to_Ne20_reaclib) = rate;
            if constexpr (std::is_same_v<T, rate_derivs_t>) {
                rate_eval.dscreened_rates_dT(k_O16_He4_to_Ne20_reaclib) = drate_dT;
            }
        }

        {
            // Ne20_He4_to_Mg24_reaclib

            Real log_scor {0.0_rt};
            Real dlog_scor_dT {0.0_rt};
#ifdef SCREENING
            log_scor = rate_eval.log_screen(k_He4_Ne20);
            if constexpr (std::is_same_v<T, rate_derivs_t>) {
                dlog_scor_dT = rate_eval.dlog_screen_dT(k_He4_Ne20);
            }
#endif
            rate_Ne20_He4_to_Mg24_reaclib<do_T_derivatives>(tfactors, log_scor, dlog_scor_dT, rate, drate_dT);
            rate_eval.screened_rates(k_Ne20_He4_to_Mg24_reaclib) = rate;
            if constexpr (std::is_same_v<T, rate_derivs_t>) {
                rate_eval.dscreened_rates_dT(k_Ne20_He4_to_Mg24_reaclib) = drate_dT;
            }
        }

        {
            // C12_C12_to_He4_Ne20_reaclib

            Real log_scor {0.0_rt};
            Real dlog_scor_dT {0.0_rt};
#ifdef SCREENING
            log_scor = rate_eval.log_screen(k_C12_C12);
            if constexpr (std::is_same_v<T, rate_derivs_t>) {
                dlog_scor_dT = rate_eval.dlog_screen_dT(k_C12_C12);
            }
#endif
            rate_C12_C12_to_He4_Ne20_reaclib<do_T_derivatives>(tfactors, log_scor, dlog_scor_dT, rate, drate_dT);
            rate_eval.screened_rates(k_C12_C12_to_He4_Ne20_reaclib) = rate;
            if constexpr (std::is_same_v<T, rate_derivs_t>) {
                rate_eval.dscreened_rates_dT(k_C12_C12_to_He4_Ne20_reaclib) = drate_dT;
            }
        }

        {
            // O16_C12_to_He4_Mg24_reaclib

            Real log_scor {0.0_rt};
            Real dlog_scor_dT {0.0_rt};
#ifdef SCREENING
            log_scor = rate_eval.log_screen(k_C12_O16);
            if constexpr (std::is_same_v<T, rate_derivs_t>) {
                dlog_scor_dT = rate_eval.dlog_screen_dT(k_C12_O16);
            }
#endif
            rate_O16_C12_to_He4_Mg24_reaclib<do_T_derivatives>(tfactors, log_scor, dlog_scor_dT, rate, drate_dT);
            rate_eval.screened_rates(k_O16_C12_to_He4_Mg24_reaclib) = rate;
            if constexpr (std::is_same_v<T, rate_derivs_t>) {
                rate_eval.dscreened_rates_dT(k_O16_C12_to_He4_Mg24_reaclib) = drate_dT;
            }
        }

        {
            // Ne20_He4_to_C12_C12_reaclib

            Real log_scor {0.0_rt};
            Real dlog_scor_dT {0.0_rt};
#ifdef SCREENING
            log_scor = rate_eval.log_screen(k_He4_Ne20);
            if constexpr (std::is_same_v<T, rate_derivs_t>) {
                dlog_scor_dT = rate_eval.dlog_screen_dT(k_He4_Ne20);
            }
#endif
            rate_Ne20_He4_to_C12_C12_reaclib<do_T_derivatives>(tfactors, log_scor, dlog_scor_dT, rate, drate_dT);
            rate_eval.screened_rates(k_Ne20_He4_to_C12_C12_reaclib) = rate;
            if constexpr (std::is_same_v<T, rate_derivs_t>) {
                rate_eval.dscreened_rates_dT(k_Ne20_He4_to_C12_C12_reaclib) = drate_dT;
            }
        }

        {
            // Mg24_He4_to_C12_O16_reaclib

            Real log_scor {0.0_rt};
            Real dlog_scor_dT {0.0_rt};
#ifdef SCREENING
            log_scor = rate_eval.log_screen(k_He4_Mg24);
            if constexpr (std::is_same_v<T, rate_derivs_t>) {
                dlog_scor_dT = rate_eval.dlog_screen_dT(k_He4_Mg24);
            }
#endif
            rate_Mg24_He4_to_C12_O16_reaclib<do_T_derivatives>(tfactors, log_scor, dlog_scor_dT, rate, drate_dT);
            rate_eval.screened_rates(k_Mg24_He4_to_C12_O16_reaclib) = rate;
            if constexpr (std::is_same_v<T, rate_derivs_t>) {
                rate_eval.dscreened_rates_dT(k_Mg24_He4_to_C12_O16_reaclib) = drate_dT;
            }
        }

        {
            // He4_He4_He4_to_C12_reaclib

            Real log_scor {0.0_rt};
            Real dlog_scor_dT {0.0_rt};
#ifdef SCREENING
            log_scor = rate_eval.log_screen(k_He4_He4) + rate_eval.log_screen(k_He4_Be8);
            if constexpr (std::is_same_v<T, rate_derivs_t>) {
                dlog_scor_dT = rate_eval.dlog_screen_dT(k_He4_He4) + rate_eval.dlog_screen_dT(k_He4_Be8);
            }
#endif
            rate_He4_He4_He4_to_C12_reaclib<do_T_derivatives>(tfactors, log_scor, dlog_scor_dT, rate, drate_dT);
            rate_eval.screened_rates(k_He4_He4_He4_to_C12_reaclib) = rate;
            if constexpr (std::is_same_v<T, rate_derivs_t>) {
                rate_eval.dscreened_rates_dT(k_He4_He4_He4_to_C12_reaclib) = drate_dT;
            }
        }


    }

}
#endif
