//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file ideal_rad.cpp
//! \brief implements ideal gas EOS with radiation pressure
//======================================================================================

// C headers

// C++ headers

// Athena++ headers
#include "../eos.hpp"

//----------------------------------------------------------------------------------------
//! \fnReal findRootCubic(Real A, Real B)
//! \brief Solve the repressed cubic equation x^4 + 4 B x - A^2 = 0
//! This repressed cubic is equivalent to the quartic equation y^4 + A y - B = 0
//! where y = sqrt(x) * (sqrt(2.0 * A / sqrt(x^3) - 1.0) - 1.0) / 2.0
Real findRootCubic(Real A, Real B) {
  Real q = 4.0/3.0 * B;
  Real r = A*A / 2.0;
  Real a = std::pow(fabs(r) + std::sqrt(r*r + q*q*q), 1.0/3.0);
  return r >= 0.0 ? a - q / a : q / a - a;
}

Real EquationOfState::TempFromRhoEg(Real rho, Real egas) {
  Real A = 1.0 / (gamma_ - 1.0) * rho * kB_ / (arad_ * mu_ * mp_);
  Real B = egas / arad_;
  Real y = findRootCubic(A, B);
  return std::sqrt(y) * (std::sqrt(2.0 * A / std::sqrt(y*y*y) - 1.0) - 1.0) / 2.0;
}

Real EquationOfState::TempFromRhoP(Real rho, Real pres) {
  Real A = 3.0 * rho * kB_ / (arad_ * mu_ * mp_);
  Real B = 3.0 * pres / arad_;
  Real y = findRootCubic(A, B);
  return std::sqrt(y) * (std::sqrt(2.0 * A / std::sqrt(y*y*y) - 1.0) - 1.0) / 2.0;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::PresFromRhoEg(Real rho, Real egas)
//! \brief Return gas pressure
Real EquationOfState::PresFromRhoEg(Real rho, Real egas) {
  Real temp = TempFromRhoEg(rho, egas);
  Real pres_rad = 1.0/3.0 * arad_ * temp*temp*temp*temp;
  Real pres_gas = rho * kB_ * temp / (mu_ * mp_);
  return pres_rad + pres_gas;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::EgasFromRhoP(Real rho, Real pres)
//! \brief Return internal energy density
Real EquationOfState::EgasFromRhoP(Real rho, Real pres) {
  Real temp = TempFromRhoP(rho, pres);
  Real ener_rad = arad_ * temp*temp*temp*temp;
  Real ener_gas = 1.0 / (gamma_ - 1.0) * rho * kB_ * temp / (mu_ * mp_);
  return ener_rad + ener_gas;
}
 
//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::AsqFromRhoP(Real rho, Real pres)
//! \brief Return adiabatic sound speed squared
Real EquationOfState::AsqFromRhoP(Real rho, Real pres) {
  Real A = 3.0 * rho * kB_ / (arad_ * mu_ * mp_);
  Real B = 3.0 * pres / arad_;
  Real y = findRootCubic(A, B);
  Real temp = std::sqrt(y) * (std::sqrt(2.0 * A / std::sqrt(y*y*y) - 1.0) - 1.0) / 2.0;
  Real pres_gas = rho * kB_ * temp / (mu_ * mp_);
  Real beta = pres_gas / pres;
  Real gm1 = gamma_ - 1.0;
  Real Gam1 = (beta*beta + gm1 * (16.0 - 12.0 * beta - 3.0 * beta*beta)) / (beta - 12.0 * gm1 * (beta - 1.0));
  return Gam1 * pres / rho;
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::InitEosConstants(ParameterInput* pin)
//! \brief Initialize constants for EOS
void EquationOfState::InitEosConstants(ParameterInput *pin) {
  return;
}