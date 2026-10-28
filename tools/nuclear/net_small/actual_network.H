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
    };
}

namespace Rates
{

    enum NetworkRates
    {
        k_O16_to_He4_C12_reaclib = 1,
        k_Ne20_to_He4_O16_reaclib = 2,
        k_Mg24_to_He4_Ne20_reaclib = 3,
        k_C12_to_3He4_reaclib = 4,
        k_C12_He4_to_O16_reaclib = 5,
        k_O16_He4_to_Ne20_reaclib = 6,
        k_Ne20_He4_to_Mg24_reaclib = 7,
        k_C12_C12_to_He4_Ne20_reaclib = 8,
        k_O16_C12_to_He4_Mg24_reaclib = 9,
        k_Ne20_He4_to_C12_C12_reaclib = 10,
        k_Mg24_He4_to_C12_O16_reaclib = 11,
        k_He4_He4_He4_to_C12_reaclib = 12,
        NumRates = k_He4_He4_He4_to_C12_reaclib
    };

    enum ScreenPairs
    {
        k_He4_C12 = 1,
        k_C12_C12 = 2,
        k_He4_He4 = 3,
        k_He4_O16 = 4,
        k_He4_Ne20 = 5,
        k_He4_Mg24 = 6,
        k_C12_O16 = 7,
        k_He4_Be8 = 8,
        NumScreenPairs = k_He4_Be8
    };

    // rate names -- note: the rates are 1-based, not zero-based, so we pad
    // this vector with rate_names[0] = "" so the indices line up with the
    // NetworkRates enum

    static const std::vector<std::string> rate_names = {
        "",   // 0
        "O16_to_He4_C12_reaclib",  // 1,
        "Ne20_to_He4_O16_reaclib",  // 2,
        "Mg24_to_He4_Ne20_reaclib",  // 3,
        "C12_to_3He4_reaclib",  // 4,
        "C12_He4_to_O16_reaclib",  // 5,
        "O16_He4_to_Ne20_reaclib",  // 6,
        "Ne20_He4_to_Mg24_reaclib",  // 7,
        "C12_C12_to_He4_Ne20_reaclib",  // 8,
        "O16_C12_to_He4_Mg24_reaclib",  // 9,
        "Ne20_He4_to_C12_C12_reaclib",  // 10,
        "Mg24_He4_to_C12_O16_reaclib",  // 11,
        "He4_He4_He4_to_C12_reaclib"  // 12,
    };

}

#endif
