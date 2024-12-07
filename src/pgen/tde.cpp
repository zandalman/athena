//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file tde.cpp
//! \brief Problem generator for tidal disruption event problems.
//!
//! Problem generator for tidal disruption event problems.
//========================================================================================

// C headers

// C++ headers
#include <cmath>      
#include <cstdio>     
#include <iostream>  
#include <sstream> 
#include <stdexcept> 
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"
#include "../units/units.hpp"

struct pgentde {
  Real G;          // Gravitational constant
  Real gamma;      // Adiabatic index
  Real n;          // Polytrope index
  Real Q;          // BH-to-star mass ratio
  Real beta;       // Penetration factor
  Real M_star;     // Star mass
  Real R_star;     // Star radius
  Real M_BH;       // BH mass
  Real r_t;        // Tidal radius
  Real r_p;        // Pericenter radius
  Real rho_c;      // Central density
  Real alpha;      // Scale length
  Real K;          // Entropy
  Real beta_start; // Start radius in units of tidal radii
  int num_le;      // Number of samples for Lane-Emden equation
  AthenaArray<Real> rho_star;   // Stellar density profile
  AthenaArray<Real> accel_grav; // Stellar gravitational field profile
};

pgentde* tde = new pgentde();

//----------------------------------------------------------------------------------------
//! \fn void calcLaneEmden(const Real xi, const Real th, const Real phi, const Real n, Real &dthdxi, Real &dphidxi)
//! \brief calcLaneEmden: Compute the derivatives in the Lane-Emden equation
//! \param xi      Dimensionless radius
//! \param th      Dimensionless density
//! \param phi     Dimensionless density derivative
//! \param n       Polytrope index
//! \param dthdxi  Derivative of the dimensionless density
//! \param dphidxi Derivative of the dimensionless density derivative

void calcLaneEmden(const Real xi, const Real th, const Real phi, const Real n, Real &dthdxi, Real &dphidxi) {
  dthdxi = -phi / (xi*xi);
  dphidxi = std::pow(fmax(th, 0.0), n) * xi*xi;
}

//----------------------------------------------------------------------------------------
//! \fn void rk4LaneEmden(const Real n, const Real dxi, Real &xi, Real &th, Real &phi)
//! \brief rk4LaneEmden: Increment xi, th, and phi in the Lane-Emden equation using the RK4 integrator
//! \param n   Polytrope index
//! \param dxi Step-size in xi
//! \param xi  Dimensionless radius
//! \param th  Dimensionless density
//! \param phi Dimensionless density derivative
void rk4LaneEmden(const Real n, const Real dxi, Real &xi, Real &th, Real &phi) {
  const Real dxi_h = dxi / 2.0; // half step-size
  Real k1_th, k2_th, k3_th, k4_th;
  Real k1_phi, k2_phi, k3_phi, k4_phi;
  calcLaneEmden(xi,         th,                 phi,                  n, k1_th, k1_phi);
  calcLaneEmden(xi + dxi_h, th + k1_th * dxi_h, phi + k1_phi * dxi_h, n, k2_th, k2_phi);
  calcLaneEmden(xi + dxi_h, th + k2_th * dxi_h, phi + k2_phi * dxi_h, n, k3_th, k3_phi);
  calcLaneEmden(xi + dxi,   th + k3_th * dxi,   phi + k3_phi * dxi,   n, k4_th, k4_phi);
  th  = th + dxi / 6.0 * (k1_th  + 2.0 * k2_th  + 2.0 * k3_th  + k4_th);
  phi = phi + dxi / 6.0 * (k1_phi + 2.0 * k2_phi + 2.0 * k3_phi + k4_phi);
  xi = xi + dxi;
}

//----------------------------------------------------------------------------------------
//! \fn void calcOrbit(const Real time)
//! \brief calcOrbit: Compute radial position of star in its orbit
//! \param time Time
//! \return Radial position of star
Real calcOrbit(const Real time) {
  Real A = sqrt(tde->G * tde->M_BH / (2.0 * tde->r_p*tde->r_p*tde->r_p));
  return tde->r_p * (-1.0 + 2.0 * cosh(2.0/3.0 * asinh(3.0/2.0 * A * time)));
}

//----------------------------------------------------------------------------------------
//! \fn Real tidalAccel(Real x, Real time)
//! \brief tidalAccel: Compute acceleration due to SMBH tidal field
//! \param x Position
//! \param time Time
//! \return Acceleration due to SMBH tidal field
Real tidalAccel(Real x, Real time) {
  Real r_star = calcOrbit(time);
  return -tde->G * tde->M_BH / (r_star*r_star*r_star) * x;
}

//----------------------------------------------------------------------------------------
//! \fn Real interp(Real xoR, AthenaArray<Real> &arr)
//! \brief interp: Interpolate the value of an array at a given position
//! \param xoR Position over radius
//! \param arr Array to interpolate
//! \return Interpolated value
Real interp(Real xoR, AthenaArray<Real> &arr) {
  if ( xoR <= 0.0 ) {
    return arr(0);
  } else if ( xoR >= 1.0 ) {
    return arr(tde->num_le);
  } else {
    int idx_low = static_cast<int>(std::floor(xoR * tde->num_le));
    int idx_high = idx_low + 1;
    Real xoR_low = static_cast<Real>(idx_low) / static_cast<Real>(tde->num_le);
    Real xoR_high = static_cast<Real>(idx_high) / static_cast<Real>(tde->num_le);
    Real val_low = arr(idx_low);
    Real val_high = arr(idx_high);
    Real iparam = (xoR - xoR_low) / (xoR_high - xoR_low); // interpolation parameter
    return val_low * (1.0 - iparam) + val_high * iparam;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real selfGravAccel(Real x0oR)
//! \brief selfGravAccel: Compute acceleration from self-gravity
//! \param x0oR Initial position over radius
//! \return Acceleration due to self-gravity
Real selfGravAccel(Real x0oR) {
  return -interp(x0oR, tde->accel_grav);
}

//----------------------------------------------------------------------------------------
//! \fn Real epsFus(Real rho_cgs, Real temp, Real x_H)
//! \brief epsFus: Compute the specific energy generation rate due to Hydrogen fusion
//! Including contributions from the proton-proton chain and the CNO cycle (although the latter dominates)
//! \param rho_cgs Density in cgs units
//! \param temp    Temperature
//! \param x_H     Hydrogen mass fraction
//! \return The specific energy generation rate due to Hydrogen fusion
Real epsFus(Real rho_cgs, Real temp, Real x_H) {
  Real temp6 = temp / 1.0e6;
  Real temp7 = temp / 1.0e7;
  Real eps_pp = 2.38e6 * rho_cgs * x_H*x_H * std::pow(temp6, -2.0/3.0) * std::exp(-33.8 * std::pow(temp6, -1.0/3.0));
  Real eps_cno = 4.4e27 * rho_cgs * x_H * Constants::Z_sol * std::pow(temp7, -2.0/3.0) * std::exp(-70.7 * std::pow(temp7, -1.0/3.0));
  // Real temp8 = temp / 1.0e8;
  // Real eps_3a = 5.4e11 * rho_cgs*rho_cgs * Y_sol*Y_sol*Y_sol / (temp8*temp8*temp8) * std::exp(-44.0 / temp8);
  return eps_pp + eps_cno;
}

//----------------------------------------------------------------------------------------
//! \fn void tdeSrcFunc(...)
//! Including contributions from tides, self-gravity, and Hydrogen fusion
//! \brief tdeSrcFunc: Custom source function
void tdeSrcFunc(
  MeshBlock *pmb, 
  const Real time, 
  const Real dt,
  const AthenaArray<Real> &prim, 
  const AthenaArray<Real> &prim_scalar,
  const AthenaArray<Real> &bcc, 
  AthenaArray<Real> &cons,
  AthenaArray<Real> &cons_scalar
) {

  for (int i=pmb->is; i<=pmb->ie; i++) {
    for (int j=pmb->js; j<=pmb->je; j++) {
      for (int k=pmb->ks; k<=pmb->ke; k++) {
        
        Real x = pmb->pcoord->x1v(i);
        Real x0oR = pmb->pscalars->r(0, k, j, i);
        Real rho = prim(IDN, k, j, i);
        Real egas = prim(IEN, k, j, i);
        Real x_H = pmb->pscalars->r(1, k, j, i);

        EquationOfState *peos = pmb->peos;
        Real temp = peos->TempFromRhoEg(rho, egas);
        Real rho_cgs = rho * pmb->pmy_mesh->punit->code_density_cgs;
        Real eps_fus_cgs = epsFus(rho_cgs, temp, x_H);
        Real eps_fus = eps_fus_cgs / pmb->pmy_mesh->punit->code_specenergydensity_cgs;
        Real ener_cno = 26.7e6 * pmb->pmy_mesh->punit->eV_code;
        Real m_H = pmb->pmy_mesh->punit->hydrogen_mass_code;

        Real accel_tidal = tidalAccel(x, time);
        Real accel_grav = selfGravAccel(x0oR);
        Real accel = accel_tidal + accel_grav;
        Real pdot = rho * accel;

        cons(IM1, k, j, i) += dt * pdot;
        cons(IEN, k, j, i) += dt * pdot * prim(IVX, k, j, i);
        cons(IEN, k, j, i) += dt * rho * eps_fus;
        cons_scalar(1, k, j, i) += -dt * rho * eps_fus * 4.0 * m_H / ener_cno;
      }
    }
  }

}

//----------------------------------------------------------------------------------------
//! \fn Real calcRstar(MeshBlock *pmb, int iout)
//! \brief calcRstar: Compute the radius of the center of mass of the star
Real calcRstar(MeshBlock *pmb, int iout) {
  Real r_star = calcOrbit(pmb->pmy_mesh->time);
  return r_star * pmb->pmy_mesh->punit->code_length_cgs;
}

//----------------------------------------------------------------------------------------
//! \fn Real calcRhoMax(MeshBlock *pmb, int iout)
//! \brief calcRhoMax: Compute the maximum density
Real calcRhoMax(MeshBlock *pmb, int iout) {
  Real rho_max = 0.0;
  for (int i=pmb->is; i<=pmb->ie; i++) {
    for (int j=pmb->js; j<=pmb->je; j++) {
      for (int k=pmb->ks; k<=pmb->ke; k++) {
        rho_max = fmax(rho_max, pmb->phydro->w(IDN, k, j, i));
      }
    }
  }
  return rho_max * pmb->pmy_mesh->punit->code_density_cgs;
}

//----------------------------------------------------------------------------------------
//! \fn Real calcPresMax(MeshBlock *pmb, int iout)
//! \brief calcPresMax: Compute the maximum pressure
Real calcPresMax(MeshBlock *pmb, int iout) {
  Real pres_max = 0.0;
  for (int i=pmb->is; i<=pmb->ie; i++) {
    for (int j=pmb->js; j<=pmb->je; j++) {
      for (int k=pmb->ks; k<=pmb->ke; k++) {
        pres_max = fmax(pres_max, pmb->phydro->w(IEN, k, j, i));
      }
    }
  }
  return pres_max * pmb->pmy_mesh->punit->code_pressure_cgs;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {

  AllocateUserHistoryOutput(3);
  EnrollUserHistoryOutput(0, calcRstar, "r_star", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, calcRhoMax, "rho_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(2, calcPresMax, "pres_max", UserHistoryOperation::max);
  EnrollUserExplicitSourceFunction(tdeSrcFunc);

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Problem Generator for tidal disruption event problems
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  std::stringstream msg;

  // populate the TDE struct
  tde->G = pmy_mesh->punit->grav_const_code;
  tde->n = pin->GetOrAddReal("problem", "n_poly", 1.5);
  tde->Q = pin->GetOrAddReal("problem", "Q", 1.0e6);
  tde->beta = pin->GetOrAddReal("problem", "beta", 1.0);
  tde->M_star = pin->GetOrAddReal("problem", "Mstar", 1.0);
  tde->R_star = pin->GetOrAddReal("problem", "Rstar", 1.0);
  tde->gamma = 1.0 + 1.0 / tde->n;
  tde->M_BH = tde->Q * tde->M_star;
  tde->r_t = tde->R_star * std::pow(tde->Q, 1.0/3.0);
  tde->r_p = tde->r_t / tde->beta;
  tde->beta_start = tde->beta * pin->GetOrAddReal("problem", "r_start", 1.0);
  Real gamma_gas = pin->GetOrAddReal("hydro", "gamma", 5.0/3.0);
  Real dxi, xi, th, phi;

  // compute the central density, scale length, and entropy
  dxi = tde->R_star / 1.0e6;
  xi = dxi, th = 1.0, phi = 0.0;
  while ( th >= 0.0 ) { rk4LaneEmden(tde->n, dxi, xi, th, phi); }
  tde->rho_c = tde->M_star / (4.0 * M_PI * tde->R_star*tde->R_star*tde->R_star) * xi*xi*xi / phi;
  tde->alpha = tde->R_star / xi;
  tde->K = 4.0 * M_PI * tde->G * tde->alpha*tde->alpha * std::pow(tde->rho_c, 1.0 - 1.0 / tde->n) / (tde->n + 1.0);

  tde->num_le = 10000;
  tde->rho_star.NewAthenaArray(tde->num_le + 1);
  tde->accel_grav.NewAthenaArray(tde->num_le + 1);

  dxi = tde->R_star / tde->alpha / static_cast<Real>(tde->num_le);
  xi = dxi, th = 1.0, phi = 0.0;
  for ( int i=0; i<=tde->num_le; i++ ) {
    rk4LaneEmden(tde->n, dxi, xi, th, phi);
    tde->rho_star(i) = tde->rho_c * std::pow(fmax(th, 0.0), tde->n);
    tde->accel_grav(i) = tde->n * tde->gamma * tde->K / tde->alpha * std::pow(tde->rho_c, tde->gamma - 1.0) * phi / (xi*xi);
  }
  
  for (int i=is; i<=ie; i++) {
    for (int j=js; j<=je; j++) {
      for (int k=js; k<=ke; k++) {
    
        // compute the star density and pressure
        Real x = pcoord->x1v(i);
        Real rho = interp(x / tde->R_star, tde->rho_star);
        Real pres = tde->K * std::pow(rho, tde->gamma);

        // compute the ambient medium density and pressure
        Real dfloor = 1.0e-4 / pmy_mesh->punit->code_density_cgs;
        Real pfloor = 1.0e7 / pmy_mesh->punit->code_pressure_cgs;
        if ( x > tde->R_star ) {
          dfloor *= tde->R_star*tde->R_star / (x*x);
          pfloor *= tde->R_star*tde->R_star / (x*x);
        }

        // set the initial conservative variables
        phydro->u(IDN, k, j, i) = fmax(dfloor, rho);
        phydro->u(IM1, k, j, i) = 0.0;
        phydro->u(IM2, k, j, i) = 0.0;
        phydro->u(IM3, k, j, i) = 0.0;
        phydro->u(IEN, k, j, i) = fmax(pfloor, pres) / (gamma_gas - 1.0);

        // set the initial passive scalars
        if ( NSCALARS > 0 ) {
          pscalars->s(0, k, j, i) = x / tde->R_star * fmax(dfloor, rho);  // initial Lagrangian position
          pscalars->s(1, k, j, i) = Constants::X_sol * fmax(dfloor, rho); // Hydrogen density
        }
      }
    }
  }

  return;
}
