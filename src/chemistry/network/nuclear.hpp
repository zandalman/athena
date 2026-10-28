#ifndef CHEMISTRY_NETWORK_NUCLEAR_HPP_
#define CHEMISTRY_NETWORK_NUCLEAR_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file nuclear.hpp
//! \brief nuclear reaction network at fixed temperature.
//!
//! Rates (JINA REACLIB), screening and the analytic Jacobian come from pynucastro-
//! generated headers in nuclear/<name>/ (--chemistry=nuclear_<name>), which only
//! nuclear.cpp includes. The ODE variables are mass fractions X_i = s_i/rho.
//! Temperature is fixed by <chemistry> temperature, so this requires an isothermal EOS.

// C headers

// C++ headers
#include <array>
#include <string>

// Athena++ classes headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "network.hpp"

class Units;

//! \class ChemNetwork
//! \brief nuclear network
class ChemNetwork : public NetworkWrapper {
  friend class MeshBlock;
 public:
  ChemNetwork(MeshBlock *pmb, ParameterInput *pin);
  ~ChemNetwork();

  // a list of species name, used in output; taken from the generated network
  static const std::array<std::string, NSPECIES> species_names;

  void InitializeNextStep(const int k, const int j, const int i);
  void RHS(const Real t, const Real *y, const Real ED, Real *ydot);
  Real Edot(const Real t, const Real *y, const Real ED);
  void Jacobian_isothermal(const Real t, const Real *y, const Real *ydot,
                           AthenaArray<Real> &jac);

 private:
  PassiveScalars *pmy_spec_;
  MeshBlock *pmy_mb_;

  Real temperature_; // fixed temperature in K, read from input file
  Real rho_cgs_;     // density in g/cm^3, updated at InitializeNextStep
};
#endif // CHEMISTRY_NETWORK_NUCLEAR_HPP_
