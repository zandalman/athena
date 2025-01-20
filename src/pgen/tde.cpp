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
  int num_tau;     // Number of samples for affine model evolution
  Real tau_max;    // Maximum dimensionless time
  AthenaArray<Real> rho_star;   // Stellar density profile
  AthenaArray<Real> accel_grav; // Stellar gravitational field profile
  AthenaArray<Real> area;       // Area factor
  AthenaArray<Real> areadot;    // Area factor derivative
};

pgentde* tde = new pgentde();

struct laneEmden {
  Real th;
  Real phi;

  laneEmden operator+(const laneEmden& other) const {
    laneEmden le;
    le.th = th + other.th;
    le.phi = phi + other.phi;
    return le;
  }

  laneEmden operator*(Real scalar) const {
    laneEmden le;
    le.th = scalar * th;
    le.phi = scalar * phi;
    return le;
  }

  friend laneEmden operator*(Real scalar, const laneEmden& le) {
    return le * scalar;
  }
};

struct affineModel {
  Real dxdx0;
  Real dxdy0;
  Real dydx0;
  Real dydy0;
  Real dxdotdx0;
  Real dxdotdy0;
  Real dydotdx0;
  Real dydotdy0;

  affineModel operator+(const affineModel& other) const {
    affineModel am;
    am.dxdx0 = dxdx0 + other.dxdx0;
    am.dxdy0 = dxdy0 + other.dxdy0;
    am.dydx0 = dydx0 + other.dydx0;
    am.dydy0 = dydy0 + other.dydy0;
    am.dxdotdx0 = dxdotdx0 + other.dxdotdx0;
    am.dxdotdy0 = dxdotdy0 + other.dxdotdy0;
    am.dydotdx0 = dydotdx0 + other.dydotdx0;
    am.dydotdy0 = dydotdy0 + other.dydotdy0;
    return am;
  }

  affineModel operator*(Real scalar) const {
    affineModel am;
    am.dxdx0 = scalar * dxdx0;
    am.dxdy0 = scalar * dxdy0;
    am.dydx0 = scalar * dydx0;
    am.dydy0 = scalar * dydy0;
    am.dxdotdx0 = scalar * dxdotdx0;
    am.dxdotdy0 = scalar * dxdotdy0;
    am.dydotdx0 = scalar * dydotdx0;
    am.dydotdy0 = scalar * dydotdy0;
    return am;
  }

  friend affineModel operator*(Real scalar, const affineModel& am) {
    return am * scalar;
  }
};

//----------------------------------------------------------------------------------------
//! \fn void calcLaneEmden(const Real xi, const laneEmden le, laneEmden &dledxi)
//! \brief calcLaneEmden: Compute derivatives in the Lane-Emden equation with respect to xi
//! \param xi      Dimensionless radius
//! \param le      Lane-Emden parameters
//! \param dledxi  Lane-Emden parameter derivatives
void calcLaneEmden(const Real xi, const laneEmden le, laneEmden &dledxi) {
  dledxi.th = -le.phi / (xi*xi);
  dledxi.phi = std::pow(fmax(le.th, 0.0), tde->n) * xi*xi;
}

//----------------------------------------------------------------------------------------
//! \fn void calcAffineModel(Real tau, const affineModel &am, affineModel &amdot)
//! \brief calcAffineModel: Compute the derivatives in the affine model with respect to tau
//! \param tau    Dimensionless time
//! \param am     Affine model parameters
//! \param amdot  Affine model parameter derivatives
void calcAffineModel(Real tau, const affineModel &am, affineModel &amdot) {
  Real tanhTau = tanh(tau);
  Real phi = 2.0 * atan(sinh(tau));
  Real sinPhi = sin(phi);
  Real cosPhi = cos(phi);
  Real dfacdx0 = cosPhi * am.dxdx0 + sinPhi * am.dydx0;
  Real dfacdy0 = cosPhi * am.dxdy0 + sinPhi * am.dydy0;

  amdot.dxdx0 = am.dxdotdx0;
  amdot.dxdy0 = am.dxdotdy0;
  amdot.dydx0 = am.dydotdx0;
  amdot.dydy0 = am.dydotdy0;
  amdot.dxdotdx0 = 3.0 * tanhTau * am.dxdotdx0 - 2.0 * am.dxdx0 + 6.0 * cosPhi * dfacdx0;
  amdot.dxdotdy0 = 3.0 * tanhTau * am.dxdotdy0 - 2.0 * am.dxdy0 + 6.0 * cosPhi * dfacdy0;
  amdot.dydotdx0 = 3.0 * tanhTau * am.dydotdx0 - 2.0 * am.dydx0 + 6.0 * sinPhi * dfacdx0;
  amdot.dydotdy0 = 3.0 * tanhTau * am.dydotdy0 - 2.0 * am.dydy0 + 6.0 * sinPhi * dfacdy0;
}

//----------------------------------------------------------------------------------------
//! \fn void rk4(Callable dydx, const Real dx, Real &x, T &y)
//! \brief rk4: Advance an integration by one step using the 4th-order Runge-Kutta method
//! \param dydx Derivative function
//! \param dx   Step size
//! \param x    Independent variable
//! \param y    Dependent variable(s)
template <typename Callable, typename T>
void rk4(Callable dydx, const Real dx, Real &x, T &y) {
  const Real dx_h = dx / 2.0;
  T k1, k2, k3, k4;
  dydx(x,        y,             k1);
  dydx(x + dx_h, y + k1 * dx_h, k2);
  dydx(x + dx_h, y + k2 * dx_h, k3);
  dydx(x + dx,   y + k3 * dx,   k4);
  y = y + dx / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
  x = x + dx;
}

//----------------------------------------------------------------------------------------
//! \fn void calcOrbit(const Real time)
//! \brief calcOrbit: Compute radial coordinate of star in its parabolic orbit
//! \param time Time
//! \return Radial coordinate of star
Real calcOrbit(const Real time) {
  Real A = sqrt(tde->G * tde->M_BH / (2.0 * tde->r_p*tde->r_p*tde->r_p));
  return tde->r_p * 1.0/2.0 * (-1.0 + 2.0 * cosh(2.0/3.0 * asinh(3.0/2.0 * A * time)));
}

Real calcTau(const Real time, const Real r_star) {
  Real sgn_tau = time > 0.0 ? 1.0 : -1.0;
  return sgn_tau * acosh(sqrt(r_star / tde->r_p));
}

//----------------------------------------------------------------------------------------
//! \fn Real tidalAccel(Real x, Real time)
//! \brief tidalAccel: Compute acceleration due to SMBH tidal field
//! \param x Position
//! \param time Time
//! \return Acceleration due to SMBH tidal field
Real tidalAccel(Real x, Real r_star) {
  return -tde->G * tde->M_BH / (r_star*r_star*r_star) * x;
}

//----------------------------------------------------------------------------------------
//! \fn Real interp(Real xoR, AthenaArray<Real> &arr)
//! \brief interp: Interpolate the value of an array at a given position
//! \param xoR Position over radius
//! \param arr Array to interpolate
//! \return Interpolated value
Real interp(const Real x, const int size, const Real xmin, const Real xmax, const AthenaArray<Real> &arr) {
  if ( x <= xmin ) return arr(0);
  if ( x >= xmax ) return arr(size - 1);
  const Real dx = (xmax - xmin) / (static_cast<Real>(size) - 1.0);
  const int idx = static_cast<int>(std::floor((x - xmin) / dx));
  const Real xlow = static_cast<Real>(idx) * dx;
  const Real iparam = (x - xlow) / dx; // interpolation parameter
  return arr(idx) * (1.0 - iparam) + arr(idx + 1) * iparam;
}

//----------------------------------------------------------------------------------------
//! \fn Real selfGravAccel(Real x0oR)
//! \brief selfGravAccel: Compute acceleration from self-gravity
//! \param x0oR Initial position over radius
//! \return Acceleration due to self-gravity
Real selfGravAccel(const Real x0oR) {
  return -interp(x0oR, tde->num_le, 0.0, 1.0, tde->accel_grav);
}

//----------------------------------------------------------------------------------------
//! \fn Real logAreaDot(const Real tau, const Real r_star)
//! \brief logAreaDot: Compute the time derivative of the log area factor
//! \param tau    Dimensionless time
//! \param r_star Radial coordinate of the star
//! \return Time derivative of the log area factor
Real logAreaDot(const Real tau, const Real r_star) {
  Real area = interp(tau, tde->num_tau, -tde->tau_max, tde->tau_max, tde->area);
  Real dtau_dt = sqrt(tde->G * tde->M_BH / (2.0 * r_star*r_star*r_star));
  Real darea_dtau = interp(tau, tde->num_tau, -tde->tau_max, tde->tau_max, tde->areadot);
  Real areadot = darea_dtau * dtau_dt;
  return areadot / area;
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

  Real mu = 0.6;
  Real gamma_gas = 5.0/3.0;
  Real r_star = calcOrbit(time);
  Real tau = calcTau(time, r_star);

  Units *units = pmb->pmy_mesh->punit;
  Real eps_thm_over_temp = units->k_boltzmann_code / (gamma_gas - 1.0) / (mu * units->hydrogen_mass_code);

  Real logrhodot_area = 0.0;
  // if ( r_star < tde->r_t ) { 
  //   logrhodot_area = -logAreaDot(tau, r_star);
  // }

  for (int i=pmb->is; i<=pmb->ie; i++) {
    for (int j=pmb->js; j<=pmb->je; j++) {
      for (int k=pmb->ks; k<=pmb->ke; k++) {
        
        Real x = pmb->pcoord->x1v(i);
        Real x0oR = pmb->pscalars->r(0, k, j, i);
        Real rho = prim(IDN, k, j, i);
        Real vel = prim(IVX, k, j, i);
        Real egas = prim(IEN, k, j, i);

        std::cout << rho << ", " << x0oR << std::endl;

        EquationOfState *peos = pmb->peos;
        Real temp = peos->TempFromRhoEg(rho, egas);
        Real eps_thm = eps_thm_over_temp * temp;

        Real vdot_tidal = tidalAccel(x, r_star);
        Real vdot_grav = selfGravAccel(x0oR);
        Real vdot = vdot_tidal + vdot_grav;
        Real rhodot = rho * logrhodot_area;

        cons(IDN, k, j, i) += dt * rhodot;
        cons(IM1, k, j, i) += dt * (rho * vdot + vel * rhodot);
        cons(IEN, k, j, i) += dt * (
          rho * vel * vdot 
          + 0.5 * vel*vel * rhodot
          + eps_thm * rhodot
        );

        for (int iscal=0; iscal<NSCALARS; iscal++) {
          cons_scalar(iscal, k, j, i) += dt * rhodot * prim_scalar(iscal, k, j, i);
        }
      }
    }
  }

}

//----------------------------------------------------------------------------------------
//! \fn Real calcRstar(MeshBlock *pmb, int iout)
//! \brief calcRstar: Compute the radius of the center of mass of the star
Real calcRstarOut(MeshBlock *pmb, int iout) {
  Real r_star = calcOrbit(pmb->pmy_mesh->time);
  return r_star * pmb->pmy_mesh->punit->code_length_cgs;
}

//----------------------------------------------------------------------------------------
//! \fn Real calcRhoMax(MeshBlock *pmb, int iout)
//! \brief calcRhoMax: Compute the maximum density
Real calcRhoMaxOut(MeshBlock *pmb, int iout) {
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
Real calcPresMaxOut(MeshBlock *pmb, int iout) {
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

Real calcTauOut(MeshBlock *pmb, int iout) {
  Real time = pmb->pmy_mesh->time;
  Real r_star = calcOrbit(time);
  return calcTau(time, r_star);
}

Real calcAreaOut(MeshBlock *pmb, int iout) {
  Real time = pmb->pmy_mesh->time;
  Real r_star = calcOrbit(time);
  Real tau = calcTau(time, r_star);
  return interp(tau, tde->num_tau, -tde->tau_max, tde->tau_max, tde->area);
}

void Mesh::InitUserMeshData(ParameterInput *pin) {

  AllocateUserHistoryOutput(5);
  EnrollUserHistoryOutput(0, calcRstarOut, "r_star", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, calcRhoMaxOut, "rho_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(2, calcPresMaxOut, "pres_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(3, calcTauOut, "tau", UserHistoryOperation::max);
  EnrollUserHistoryOutput(4, calcAreaOut, "area", UserHistoryOperation::max);
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
  tde->tau_max = acosh(sqrt(tde->beta));

  Real gamma_gas = pin->GetOrAddReal("hydro", "gamma", 5.0/3.0);
  Real dxi, xi;
  Real dtau, tau;
  laneEmden le;

  // populate the affine model struct
  affineModel am;
  am.dxdx0 = 1.0;
  am.dxdy0 = 0.0;
  am.dydx0 = 0.0;
  am.dydy0 = 1.0;
  am.dxdotdx0 = 0.0;
  am.dxdotdy0 = 0.0;
  am.dydotdx0 = 0.0;
  am.dydotdy0 = 0.0;

  // compute the central density, scale length, and entropy
  dxi = tde->R_star / 1.0e6;
  le.th = 1.0;
  le.phi = 0.0;
  while ( le.th >= 0.0 ) { 
    rk4<decltype(calcLaneEmden), laneEmden>(calcLaneEmden, dxi, xi, le);
  }
  tde->rho_c = tde->M_star / (4.0 * M_PI * tde->R_star*tde->R_star*tde->R_star) * xi*xi*xi / le.phi;
  tde->alpha = tde->R_star / xi;
  tde->K = 4.0 * M_PI * tde->G * tde->alpha*tde->alpha * std::pow(tde->rho_c, 1.0 - 1.0 / tde->n) / (tde->n + 1.0);

  // create arrays for stellar profile
  tde->num_le = 16384;
  tde->rho_star.NewAthenaArray(tde->num_le);
  tde->accel_grav.NewAthenaArray(tde->num_le);

  // compute stellar profile
  dxi = tde->R_star / tde->alpha / static_cast<Real>(tde->num_le - 1);
  tde->rho_star(0) = tde->rho_c;
  tde->accel_grav(0) = 0.0;
  le.th = 1.0;
  le.phi = 0.0;
  for ( int i=1; i<tde->num_le; i++ ) {
    xi = dxi + static_cast<Real>(i) * dxi;
    tde->rho_star(i) = tde->rho_c * std::pow(fmax(le.th, 0.0), tde->n);
    tde->accel_grav(i) = tde->n * tde->gamma * tde->K / tde->alpha * std::pow(tde->rho_c, tde->gamma - 1.0) * le.phi / (xi*xi);
    rk4<decltype(calcLaneEmden), laneEmden>(calcLaneEmden, dxi, xi, le);
  }

  // create arrays for affine model
  tde->num_tau = 16384;
  tde->area.NewAthenaArray(tde->num_tau);
  tde->areadot.NewAthenaArray(tde->num_tau);

  // compute the affine model
  dtau = 2.0 * tde->tau_max / static_cast<Real>(tde->num_tau - 1);
  for ( int i=0; i<tde->num_tau; i++ ) {
    tau = -tde->tau_max + static_cast<Real>(i) * dtau;
    tde->area(i) = am.dxdx0 * am.dydy0 - am.dxdy0 * am.dydx0;
    tde->areadot(i) = am.dxdotdx0 * am.dydy0 + am.dxdx0 * am.dydotdy0 - am.dxdotdy0 * am.dydx0 - am.dxdy0 * am.dydotdx0;
    rk4<decltype(calcAffineModel), affineModel>(calcAffineModel, dtau, tau, am);
  }
  
  for (int i=is; i<=ie; i++) {
    for (int j=js; j<=je; j++) {
      for (int k=js; k<=ke; k++) {
    
        // compute the star density and pressure
        Real x = pcoord->x1v(i);
        Real rho = interp(x / tde->R_star, tde->num_le, 0.0, 1.0, tde->rho_star);
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
        pscalars->s(0, k, j, i) = x / tde->R_star * fmax(dfloor, rho);  // initial Lagrangian position
        pscalars->s(1, k, j, i) = Constants::X_sol * fmax(dfloor, rho); // Hydrogen density
      }
    }
  }

  return;
}
