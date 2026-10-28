//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file nuc_onezone.cpp
//! \brief one-zone nuclear burning test: uniform, static gas at fixed density and
//! temperature (isothermal EOS), burned by the nuclear network in src/chemistry/.
//!
//! The timestep is set by a user function, dt = max(dt_init, dt_fac*time), so that
//! steps are logarithmically spaced in time. Set <hydro> iso_sound_speed small so that
//! the hydro timestep does not limit it.
//======================================================================================

// C headers

// C++ headers
#include <algorithm>  // max
#include <cmath>      // abs
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // std::runtime_error()
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../defs.hpp"
#include "../eos/eos.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"

namespace {
Real dt_init, dt_fac;
Real OneZoneTimeStep(MeshBlock *pmb);
} // namespace

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief enroll the log-spaced timestep
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  dt_init = pin->GetReal("problem", "dt_init");
  dt_fac = pin->GetOrAddReal("problem", "dt_fac", 0.05);
  EnrollUserTimeStepFunction(OneZoneTimeStep);
  return;
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief uniform density at rest; initial mass fractions from <problem> X_<species>
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  if (!CHEMISTRY_ENABLED) {
    std::stringstream msg;
    msg << "### FATAL ERROR in nuc_onezone.cpp ProblemGenerator" << std::endl
        << "  configure with --chemistry=<nuclear network>" << std::endl;
    ATHENA_ERROR(msg);
  }

  const Real rho = pin->GetReal("problem", "rho");

  // initial mass fractions, normalized to sum to 1
  Real x_init[NSPECIES];
  Real x_sum = 0.;
  for (int ispec=0; ispec<NSPECIES; ++ispec) {
    x_init[ispec] = pin->GetOrAddReal("problem",
        "X_"+pscalars->chemnet.species_names[ispec], 0.);
    x_sum += x_init[ispec];
  }
  if (std::abs(x_sum - 1.) > 1e-6) {
    std::stringstream msg;
    msg << "### FATAL ERROR in nuc_onezone.cpp ProblemGenerator" << std::endl
        << "  initial mass fractions sum to " << x_sum << ", not 1" << std::endl;
    ATHENA_ERROR(msg);
  }

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        phydro->u(IDN, k, j, i) = rho;
        phydro->u(IM1, k, j, i) = 0.;
        phydro->u(IM2, k, j, i) = 0.;
        phydro->u(IM3, k, j, i) = 0.;
        for (int ispec=0; ispec<NSPECIES; ++ispec) {
          pscalars->s(ispec, k, j, i) = x_init[ispec]/x_sum*rho;
        }
      }
    }
  }
  return;
}

namespace {
//----------------------------------------------------------------------------------------
//! \fn Real OneZoneTimeStep(MeshBlock *pmb)
//! \brief logarithmically spaced timestep, dt = max(dt_init, dt_fac*time)

Real OneZoneTimeStep(MeshBlock *pmb) {
  return std::max(dt_init, dt_fac*pmb->pmy_mesh->time);
}
} // namespace
