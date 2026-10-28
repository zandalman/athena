#ifndef ACTUAL_NETWORK_H
#define ACTUAL_NETWORK_H

#include <amrex_bridge.H>

#include <fundamental_constants.H>
#include <network_properties.H>

inline void actual_network_init() {}

const std::string network_name = "pynucastro-cxx";

namespace network
{

    // nuclei masses (in grams)

    inline Array1D<Real, 1, NumSpec> mion{
        6.646479071584587e-24_rt,  // He4
        1.99264687992e-23_rt,  // C12
        2.6560180592333686e-23_rt,  // O16
        3.3198227947612416e-23_rt,  // Ne20
        3.9828098739467446e-23_rt,  // Mg24
        4.6456779473820677e-23_rt,  // Si28
        5.309087322384128e-23_rt,  // S32
        5.972551377884467e-23_rt,  // Ar36
        6.635944331004904e-23_rt,  // Ca40
        7.299678247096977e-23_rt,  // Ti44
        7.962953983065421e-23_rt,  // Cr48
        8.626187166893794e-23_rt,  // Fe52
        9.289408870379396e-23_rt,  // Ni56
    };
}

namespace Rates
{

    enum NetworkRates
    {
        k_O16_to_He4_C12_reaclib = 1,
        k_Ne20_to_He4_O16_reaclib = 2,
        k_C12_to_3He4_reaclib = 3,
        k_C12_He4_to_O16_reaclib = 4,
        k_O16_He4_to_Ne20_reaclib = 5,
        k_Ne20_C12_to_He4_Si28_reaclib = 6,
        k_Si28_He4_to_C12_Ne20_reaclib = 7,
        k_He4_He4_He4_to_C12_reaclib = 8,
        k_C12_C12_to_p_Na23_reaclib = 9,
        k_Na23_p_to_C12_C12_reaclib = 10,
        k_Na23_p_to_He4_Ne20_reaclib = 11,
        k_Na23_p_to_Mg24_reaclib = 12,
        k_Mg24_to_p_Na23_reaclib = 13,
        k_C12_C12_to_He4_Ne20_reaclib = 14,
        k_Ne20_He4_to_C12_C12_reaclib = 15,
        k_Ne20_He4_to_p_Na23_reaclib = 16,
        k_Ne20_He4_to_Mg24_reaclib = 17,
        k_Mg24_to_He4_Ne20_reaclib = 18,
        k_O16_C12_to_p_Al27_reaclib = 19,
        k_Al27_p_to_C12_O16_reaclib = 20,
        k_Al27_p_to_He4_Mg24_reaclib = 21,
        k_Al27_p_to_Si28_reaclib = 22,
        k_Si28_to_p_Al27_reaclib = 23,
        k_O16_C12_to_He4_Mg24_reaclib = 24,
        k_Mg24_He4_to_C12_O16_reaclib = 25,
        k_Mg24_He4_to_p_Al27_reaclib = 26,
        k_Mg24_He4_to_Si28_reaclib = 27,
        k_Si28_to_He4_Mg24_reaclib = 28,
        k_O16_O16_to_p_P31_reaclib = 29,
        k_P31_p_to_O16_O16_reaclib = 30,
        k_P31_p_to_He4_Si28_reaclib = 31,
        k_P31_p_to_S32_reaclib = 32,
        k_S32_to_p_P31_reaclib = 33,
        k_O16_O16_to_He4_Si28_reaclib = 34,
        k_Si28_He4_to_O16_O16_reaclib = 35,
        k_Si28_He4_to_p_P31_reaclib = 36,
        k_Si28_He4_to_S32_reaclib = 37,
        k_S32_to_He4_Si28_reaclib = 38,
        k_S32_He4_to_p_Cl35_reaclib = 39,
        k_S32_He4_to_Ar36_reaclib = 40,
        k_Cl35_p_to_He4_S32_reaclib = 41,
        k_Cl35_p_to_Ar36_reaclib = 42,
        k_Ar36_to_He4_S32_reaclib = 43,
        k_Ar36_to_p_Cl35_reaclib = 44,
        k_Ar36_He4_to_p_K39_reaclib = 45,
        k_Ar36_He4_to_Ca40_reaclib = 46,
        k_K39_p_to_He4_Ar36_reaclib = 47,
        k_K39_p_to_Ca40_reaclib = 48,
        k_Ca40_to_He4_Ar36_reaclib = 49,
        k_Ca40_to_p_K39_reaclib = 50,
        k_Ca40_He4_to_p_Sc43_reaclib = 51,
        k_Ca40_He4_to_Ti44_reaclib = 52,
        k_Sc43_p_to_He4_Ca40_reaclib = 53,
        k_Sc43_p_to_Ti44_reaclib = 54,
        k_Ti44_to_He4_Ca40_reaclib = 55,
        k_Ti44_to_p_Sc43_reaclib = 56,
        k_Ti44_He4_to_p_V47_reaclib = 57,
        k_Ti44_He4_to_Cr48_reaclib = 58,
        k_V47_p_to_He4_Ti44_reaclib = 59,
        k_V47_p_to_Cr48_reaclib = 60,
        k_Cr48_to_He4_Ti44_reaclib = 61,
        k_Cr48_to_p_V47_reaclib = 62,
        k_Cr48_He4_to_p_Mn51_reaclib = 63,
        k_Cr48_He4_to_Fe52_reaclib = 64,
        k_Mn51_p_to_He4_Cr48_reaclib = 65,
        k_Mn51_p_to_Fe52_reaclib = 66,
        k_Fe52_to_He4_Cr48_reaclib = 67,
        k_Fe52_to_p_Mn51_reaclib = 68,
        k_Fe52_He4_to_p_Co55_reaclib = 69,
        k_Fe52_He4_to_Ni56_reaclib = 70,
        k_Co55_p_to_He4_Fe52_reaclib = 71,
        k_Co55_p_to_Ni56_reaclib = 72,
        k_Ni56_to_He4_Fe52_reaclib = 73,
        k_Ni56_to_p_Co55_reaclib = 74,
        k_C12_C12_to_Mg24_approx = 75,
        k_Mg24_to_C12_C12_approx = 76,
        k_C12_C12_to_Ne20_He4_approx = 77,
        k_Ne20_He4_to_C12_C12_approx = 78,
        k_Ne20_He4_to_Mg24_approx = 79,
        k_Mg24_to_Ne20_He4_approx = 80,
        k_C12_O16_to_Si28_approx = 81,
        k_Si28_to_O16_C12_approx = 82,
        k_C12_O16_to_Mg24_He4_approx = 83,
        k_Mg24_He4_to_O16_C12_approx = 84,
        k_Mg24_He4_to_Si28_approx = 85,
        k_Si28_to_Mg24_He4_approx = 86,
        k_O16_O16_to_S32_approx = 87,
        k_S32_to_O16_O16_approx = 88,
        k_O16_O16_to_Si28_He4_approx = 89,
        k_Si28_He4_to_O16_O16_approx = 90,
        k_Si28_He4_to_S32_approx = 91,
        k_S32_to_Si28_He4_approx = 92,
        k_S32_He4_to_Ar36_approx = 93,
        k_Ar36_to_S32_He4_approx = 94,
        k_Ar36_He4_to_Ca40_approx = 95,
        k_Ca40_to_Ar36_He4_approx = 96,
        k_Ca40_He4_to_Ti44_approx = 97,
        k_Ti44_to_Ca40_He4_approx = 98,
        k_Ti44_He4_to_Cr48_approx = 99,
        k_Cr48_to_Ti44_He4_approx = 100,
        k_Cr48_He4_to_Fe52_approx = 101,
        k_Fe52_to_Cr48_He4_approx = 102,
        k_Fe52_He4_to_Ni56_approx = 103,
        k_Ni56_to_Fe52_He4_approx = 104,
        NumRates = k_Ni56_to_Fe52_He4_approx
    };

    enum ScreenPairs
    {
        k_He4_O16 = 1,
        k_p_K39 = 2,
        k_He4_Ca40 = 3,
        k_He4_Ar36 = 4,
        k_p_Sc43 = 5,
        k_He4_C12 = 6,
        k_He4_Ti44 = 7,
        k_He4_Mg24 = 8,
        k_C12_O16 = 9,
        k_p_V47 = 10,
        k_He4_Cr48 = 11,
        k_C12_C12 = 12,
        k_p_Co55 = 13,
        k_He4_Ne20 = 14,
        k_p_Al27 = 15,
        k_p_Mn51 = 16,
        k_p_Na23 = 17,
        k_He4_Fe52 = 18,
        k_He4_S32 = 19,
        k_p_P31 = 20,
        k_He4_He4 = 21,
        k_He4_Si28 = 22,
        k_C12_Ne20 = 23,
        k_p_Cl35 = 24,
        k_O16_O16 = 25,
        k_He4_Be8 = 26,
        NumScreenPairs = k_He4_Be8
    };

    // rate names -- note: the rates are 1-based, not zero-based, so we pad
    // this vector with rate_names[0] = "" so the indices line up with the
    // NetworkRates enum

    static const std::vector<std::string> rate_names = {
        "",   // 0
        "O16_to_He4_C12_reaclib",  // 1,
        "Ne20_to_He4_O16_reaclib",  // 2,
        "C12_to_3He4_reaclib",  // 3,
        "C12_He4_to_O16_reaclib",  // 4,
        "O16_He4_to_Ne20_reaclib",  // 5,
        "Ne20_C12_to_He4_Si28_reaclib",  // 6,
        "Si28_He4_to_C12_Ne20_reaclib",  // 7,
        "He4_He4_He4_to_C12_reaclib",  // 8,
        "C12_C12_to_p_Na23_reaclib",  // 9,
        "Na23_p_to_C12_C12_reaclib",  // 10,
        "Na23_p_to_He4_Ne20_reaclib",  // 11,
        "Na23_p_to_Mg24_reaclib",  // 12,
        "Mg24_to_p_Na23_reaclib",  // 13,
        "C12_C12_to_He4_Ne20_reaclib",  // 14,
        "Ne20_He4_to_C12_C12_reaclib",  // 15,
        "Ne20_He4_to_p_Na23_reaclib",  // 16,
        "Ne20_He4_to_Mg24_reaclib",  // 17,
        "Mg24_to_He4_Ne20_reaclib",  // 18,
        "O16_C12_to_p_Al27_reaclib",  // 19,
        "Al27_p_to_C12_O16_reaclib",  // 20,
        "Al27_p_to_He4_Mg24_reaclib",  // 21,
        "Al27_p_to_Si28_reaclib",  // 22,
        "Si28_to_p_Al27_reaclib",  // 23,
        "O16_C12_to_He4_Mg24_reaclib",  // 24,
        "Mg24_He4_to_C12_O16_reaclib",  // 25,
        "Mg24_He4_to_p_Al27_reaclib",  // 26,
        "Mg24_He4_to_Si28_reaclib",  // 27,
        "Si28_to_He4_Mg24_reaclib",  // 28,
        "O16_O16_to_p_P31_reaclib",  // 29,
        "P31_p_to_O16_O16_reaclib",  // 30,
        "P31_p_to_He4_Si28_reaclib",  // 31,
        "P31_p_to_S32_reaclib",  // 32,
        "S32_to_p_P31_reaclib",  // 33,
        "O16_O16_to_He4_Si28_reaclib",  // 34,
        "Si28_He4_to_O16_O16_reaclib",  // 35,
        "Si28_He4_to_p_P31_reaclib",  // 36,
        "Si28_He4_to_S32_reaclib",  // 37,
        "S32_to_He4_Si28_reaclib",  // 38,
        "S32_He4_to_p_Cl35_reaclib",  // 39,
        "S32_He4_to_Ar36_reaclib",  // 40,
        "Cl35_p_to_He4_S32_reaclib",  // 41,
        "Cl35_p_to_Ar36_reaclib",  // 42,
        "Ar36_to_He4_S32_reaclib",  // 43,
        "Ar36_to_p_Cl35_reaclib",  // 44,
        "Ar36_He4_to_p_K39_reaclib",  // 45,
        "Ar36_He4_to_Ca40_reaclib",  // 46,
        "K39_p_to_He4_Ar36_reaclib",  // 47,
        "K39_p_to_Ca40_reaclib",  // 48,
        "Ca40_to_He4_Ar36_reaclib",  // 49,
        "Ca40_to_p_K39_reaclib",  // 50,
        "Ca40_He4_to_p_Sc43_reaclib",  // 51,
        "Ca40_He4_to_Ti44_reaclib",  // 52,
        "Sc43_p_to_He4_Ca40_reaclib",  // 53,
        "Sc43_p_to_Ti44_reaclib",  // 54,
        "Ti44_to_He4_Ca40_reaclib",  // 55,
        "Ti44_to_p_Sc43_reaclib",  // 56,
        "Ti44_He4_to_p_V47_reaclib",  // 57,
        "Ti44_He4_to_Cr48_reaclib",  // 58,
        "V47_p_to_He4_Ti44_reaclib",  // 59,
        "V47_p_to_Cr48_reaclib",  // 60,
        "Cr48_to_He4_Ti44_reaclib",  // 61,
        "Cr48_to_p_V47_reaclib",  // 62,
        "Cr48_He4_to_p_Mn51_reaclib",  // 63,
        "Cr48_He4_to_Fe52_reaclib",  // 64,
        "Mn51_p_to_He4_Cr48_reaclib",  // 65,
        "Mn51_p_to_Fe52_reaclib",  // 66,
        "Fe52_to_He4_Cr48_reaclib",  // 67,
        "Fe52_to_p_Mn51_reaclib",  // 68,
        "Fe52_He4_to_p_Co55_reaclib",  // 69,
        "Fe52_He4_to_Ni56_reaclib",  // 70,
        "Co55_p_to_He4_Fe52_reaclib",  // 71,
        "Co55_p_to_Ni56_reaclib",  // 72,
        "Ni56_to_He4_Fe52_reaclib",  // 73,
        "Ni56_to_p_Co55_reaclib",  // 74,
        "C12_C12_to_Mg24_approx",  // 75,
        "Mg24_to_C12_C12_approx",  // 76,
        "C12_C12_to_Ne20_He4_approx",  // 77,
        "Ne20_He4_to_C12_C12_approx",  // 78,
        "Ne20_He4_to_Mg24_approx",  // 79,
        "Mg24_to_Ne20_He4_approx",  // 80,
        "C12_O16_to_Si28_approx",  // 81,
        "Si28_to_O16_C12_approx",  // 82,
        "C12_O16_to_Mg24_He4_approx",  // 83,
        "Mg24_He4_to_O16_C12_approx",  // 84,
        "Mg24_He4_to_Si28_approx",  // 85,
        "Si28_to_Mg24_He4_approx",  // 86,
        "O16_O16_to_S32_approx",  // 87,
        "S32_to_O16_O16_approx",  // 88,
        "O16_O16_to_Si28_He4_approx",  // 89,
        "Si28_He4_to_O16_O16_approx",  // 90,
        "Si28_He4_to_S32_approx",  // 91,
        "S32_to_Si28_He4_approx",  // 92,
        "S32_He4_to_Ar36_approx",  // 93,
        "Ar36_to_S32_He4_approx",  // 94,
        "Ar36_He4_to_Ca40_approx",  // 95,
        "Ca40_to_Ar36_He4_approx",  // 96,
        "Ca40_He4_to_Ti44_approx",  // 97,
        "Ti44_to_Ca40_He4_approx",  // 98,
        "Ti44_He4_to_Cr48_approx",  // 99,
        "Cr48_to_Ti44_He4_approx",  // 100,
        "Cr48_He4_to_Fe52_approx",  // 101,
        "Fe52_to_Cr48_He4_approx",  // 102,
        "Fe52_He4_to_Ni56_approx",  // 103,
        "Ni56_to_Fe52_He4_approx"  // 104,
    };

}

#endif
