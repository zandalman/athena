//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file nuclear.cpp
//! \brief implementation of functions in class ChemNetwork, using a pynucastro-generated
//! nuclear network at fixed temperature.

// this class header
#include "nuclear.hpp"

// C headers

// C++ header
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error

// Athena++ header
#include "../../defs.hpp"
#include "../../eos/eos.hpp"
#include "../../hydro/hydro.hpp"
#include "../../mesh/mesh.hpp"
#include "../../parameter_input.hpp"
#include "../../scalars/scalars.hpp"
#include "../../units/units.hpp"
#include "network.hpp"

// pynucastro-generated network (src/chemistry/network/nuclear/<name>/, C++20)
#ifndef SCREENING
#define SCREENING
#endif
#include <amrex_bridge.H>
#include <network_properties.H>
#include <actual_network.H>
#include <actual_rhs.H>

static_assert(NumSpec == NSPECIES,
              "NSPECIES in configure.py does not match the generated network");

namespace {
std::array<std::string, NSPECIES> GeneratedSpeciesNames() {
  std::array<std::string, NSPECIES> names;
  for (int n=0; n<NumSpec; ++n) {
    names[n] = spec_names[n];
  }
  return names;
}

// fill the pynucastro burn state from mass fractions
void FillBurnState(const Real rho, const Real T, const Real *x, burn_t &state) {
  state.rho = rho;
  state.T = T;
  for (int n=0; n<NumSpec; ++n) {
    state.xn[n] = x[n];
  }
  compute_ye(state);
}
} // namespace

// species names, in the order of the generated network. spec_names is defined above
// in this translation unit, so it is initialized first.
const std::array<std::string, NSPECIES> ChemNetwork::species_names =
  GeneratedSpeciesNames();

//----------------------------------------------------------------------------------------
//! \brief ChemNetwork constructor

ChemNetwork::ChemNetwork(MeshBlock *pmb, ParameterInput *pin) {
  pmy_spec_ = pmb->pscalars;
  pmy_mb_ = pmb;
  rho_cgs_ = 0.;

  if (NON_BAROTROPIC_EOS) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ChemNetwork constructor [nuclear]" << std::endl
        << "  this network is at fixed temperature; configure with --eos=isothermal"
        << std::endl;
    ATHENA_ERROR(msg);
  }

  // fixed temperature in K
  temperature_ = pin->GetReal("chemistry", "temperature");
  actual_network_init();
}

//----------------------------------------------------------------------------------------
//! \brief ChemNetwork destructor

ChemNetwork::~ChemNetwork() {
}

//----------------------------------------------------------------------------------------
//! \fn void ChemNetwork::InitializeNextStep(const int k, const int j, const int i)
//! \brief Set the density of the cell in cgs.

void ChemNetwork::InitializeNextStep(const int k, const int j, const int i) {
  Real rho, rho_floor;
  rho = pmy_mb_->phydro->w(IDN, k, j, i);
  rho_floor = pmy_mb_->peos->GetDensityFloor();
  rho = (rho > rho_floor) ?  rho : rho_floor;
  rho_cgs_ = rho * pmy_mb_->pmy_mesh->punit->code_density_cgs;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ChemNetwork::RHS(const Real t, const Real *y, const Real ED,
//!                       Real *ydot)
//! \brief RHS: right-hand-side of ODE.
//!
//! y are mass fractions X_i. The generated network returns dY_i/dt in s^-1 with
//! molar abundances Y_i = X_i/A_i, so dX_i/dt = A_i dY_i/dt.
//! all input/output variables are in code units

void ChemNetwork::RHS(const Real t, const Real *y, const Real ED, Real *ydot) {
  burn_t state;
  FillBurnState(rho_cgs_, temperature_, y, state);
  Array1D<Real, 1, NumSpec> ydot_molar;
  Real enu_weak;
  actual_rhs(state, ydot_molar, enu_weak);
  const Real code_time_cgs = pmy_mb_->pmy_mesh->punit->code_time_cgs;
  for (int n=0; n<NumSpec; ++n) {
    ydot[n] = aion[n] * ydot_molar(n+1) * code_time_cgs;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ChemNetwork::Jacobian_isothermal(const Real t, const Real *y,
//!                                           const Real *ydot, AthenaArray<Real> &jac)
//! \brief analytic Jacobian d(Xdot_i)/dX_j = A_i/A_j d(Ydot_i)/dY_j, in code units.
//!
//! The generated Jacobian neglects the composition dependence of the screening factors.

void ChemNetwork::Jacobian_isothermal(const Real t, const Real *y, const Real *ydot,
                                      AthenaArray<Real> &jac) {
  burn_t state;
  FillBurnState(rho_cgs_, temperature_, y, state);
  MathArray2D<1, NumSpec, 1, NumSpec> jac_molar;
  actual_jac(state, jac_molar);
  const Real code_time_cgs = pmy_mb_->pmy_mesh->punit->code_time_cgs;
  for (int i=0; i<NumSpec; ++i) {
    for (int j=0; j<NumSpec; ++j) {
      jac(i, j) = aion[i] * jac_molar(i+1, j+1) * aion_inv[j] * code_time_cgs;
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Real ChemNetwork::Edot(const Real t, const Real *y, const Real ED)
//! \brief energy equation dED/dt; zero at fixed temperature (never called with an
//! isothermal EOS)

Real ChemNetwork::Edot(const Real t, const Real *y, const Real ED) {
  return 0.;
}
