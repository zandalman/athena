//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file eos_table.cpp
//! \brief implements functions in class EquationOfState for an EOS lookup table
//======================================================================================

// C headers

// C++ headers
#include <cmath>   // sqrt()
#include <fstream>
#include <iostream> // ifstream
#include <sstream>
#include <stdexcept> // std::invalid_argument
#include <string>

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../field/field.hpp"
#include "../../parameter_input.hpp"
#include "../../utils/interp_table.hpp"
#include "../eos.hpp"

namespace {
//----------------------------------------------------------------------------------------
//! \fn Real GetEosData(EosTable *ptable, int kOut, Real var, Real rho)
//! \brief Gets interpolated data from EOS table assuming 'var' has dimensions
//!        of energy per mass.
inline Real GetEosData(EosTable *ptable, int nTab, int kOut, Real var, Real rho) {
  Real x1 = std::log10(rho * ptable->rhoUnit);
  Real x2 = std::log10(var * ptable->egasUnit / ptable->rhoUnit);
  Real res;
  if ( nTab == 0 ) { 
    res = ptable->table1.interpolate(kOut, x2, x1);
  } else { 
    res = ptable->table2.interpolate(kOut, x2, x1);
  }
  return res;
}
} // namespace

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::PresFromRhoEg(Real rho, Real egas)
//! \brief Return interpolated gas pressure
Real EquationOfState::PresFromRhoEg(Real rho, Real egas) {
  if ( std::log10(egas / rho * ptable->egasUnit / ptable->rhoUnit) < ptable->logEpsMin ) {
    return 2.0/3.0 * egas; // (gam - 1) e
  } else {
    return std::pow((Real)10, GetEosData(ptable, 0, 0, egas / rho, rho)) / egas_unit_;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::EgasFromRhoP(Real rho, Real pres)
//! \brief Return interpolated internal energy density
Real EquationOfState::EgasFromRhoP(Real rho, Real pres) {
  if ( std::log10(pres / rho * ptable->egasUnit / ptable->rhoUnit) < ptable->logPorMin ) {
    return 3.0/2.0 * pres; // p / (gam - 1)
  } else {
    return std::pow((Real)10, GetEosData(ptable, 1, 0, pres / rho, rho)) / egas_unit_;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::AsqFromRhoP(Real rho, Real pres)
//! \brief Return interpolated adiabatic sound speed squared
Real EquationOfState::AsqFromRhoP(Real rho, Real pres) {
  if ( std::log10(pres / rho * ptable->egasUnit / ptable->rhoUnit) < ptable->logPorMin ) {
    return 5.0/3.0 * pres / rho;
  } else {
    return GetEosData(ptable, 1, 1, pres / rho, rho) * pres / rho;
  }
}

//----------------------------------------------------------------------------------------
//! void EquationOfState::InitEosConstants(ParameterInput* pin)
//! \brief Initialize constants for EOS
void EquationOfState::InitEosConstants(ParameterInput* pin) {
  return;
}
