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
  Real gamg;       // Adiabatic index
  Real gamp;       // Adiabatic index (from polytrope)
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
  Real th;  // Dimensionless density
  Real phi; // Dimensionless density derivative

  // Overload the + operator
  laneEmden operator+(const laneEmden& other) const {
    laneEmden le;
    le.th = th + other.th;
    le.phi = phi + other.phi;
    return le;
  }

  // Overload the * operator
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
  Real dxdx0;    // Jacobian of the affine transformation
  Real dxdy0;    // ( dxdx0 dxdy0 )
  Real dydx0;    // ( dydx0 dydy0 )
  Real dydy0;    
  Real dxdotdx0; // Jacobian dimensionless time derivative of the affine transformation
  Real dxdotdy0; // ___d ( dxdx0 dxdy0 )
  Real dydotdx0; // dtau ( dydx0 dydy0 )
  Real dydotdy0;

  // Overload the + operator
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

  // Overload the * operator
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
//! \brief calcLaneEmden: Compute derivatives in the Lane-Emden equation with respect to xi.
//! \param xi      Dimensionless radius
//! \param le      Lane-Emden parameters
//! \param dledxi  Lane-Emden parameter derivatives
void calcLaneEmden(const Real xi, const laneEmden le, laneEmden &dledxi) {
  dledxi.th = -le.phi / (xi*xi);
  dledxi.phi = std::pow(fmax(le.th, 0.0), tde->n) * xi*xi;
}

//----------------------------------------------------------------------------------------
//! \fn void calcAffineModel(Real tau, const affineModel &am, affineModel &amdot)
//! See Coughlin&Nixon2022 Section 3
//! \brief calcAffineModel: Compute the derivatives in the affine model with respect to tau.
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
//! \brief rk4: Advance an integration by one step using the 4th-order Runge-Kutta method.
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
//! \fn Real calcOrbit(const Real time)
//! \brief calcOrbit: Compute radial coordinate of star in its parabolic orbit.
//! The formula is derived from Barker's equation.
//! \param time Time
//! \return Radial coordinate of star
Real calcOrbit(const Real time) {
  Real A = sqrt(tde->G * tde->M_BH / (2.0 * tde->r_p*tde->r_p*tde->r_p));
  return tde->r_p * (-1.0 + 2.0 * cosh(2.0/3.0 * asinh(3.0/2.0 * A * time)));
}

//----------------------------------------------------------------------------------------
//! \fn Real calcTau(const Real time, const Real r_star)
//! \brief calcOrbit: Compute dimensionless time.
//! See Coughlin&Nixon2022 Section 2.
//! \param time Time
//! \param r_star Radial coordinate of the star
//! \return Dimensionless time
Real calcTau(const Real time, const Real r_star) {
  Real sgn_tau = time > 0.0 ? 1.0 : -1.0;
  return sgn_tau * acosh(sqrt(r_star / tde->r_p));
}

//----------------------------------------------------------------------------------------
//! \fn Real interp(const Real x, const int size, const Real xmin, const Real xmax, const AthenaArray<Real> &arr)
//! \brief interp: Interpolate from an array of values at a given point.
//! We assume the point array has uniform spacing.
//! \param x    Interpolation point
//! \param size Size of the array
//! \param xmin Minimum value of point array
//! \param xmax Maximum value of point array
//! \param arr  Array of values
//! \return Interpolated value
Real interp(const Real x, const int size, const Real xmin, const Real xmax, const AthenaArray<Real> &arr) {
  if ( x <= xmin ) return arr(0); // check if point outside of point array
  if ( x >= xmax ) return arr(size - 1);
  const Real dx = (xmax - xmin) / (static_cast<Real>(size) - 1.0);
  const int idx = static_cast<int>(std::floor((x - xmin) / dx));
  const Real xlow = xmin + static_cast<Real>(idx) * dx;
  const Real iparam = (x - xlow) / dx; // interpolation parameter
  return arr(idx) * (1.0 - iparam) + arr(idx + 1) * iparam;
}

//----------------------------------------------------------------------------------------
//! \fn Real logAreaDot(const Real tau, const Real r_star)
//! \brief logAreaDot: Compute the time derivative of the log area factor.
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
//! \brief tdeSrcFunc: Custom source function
//! Including contributions from tides, self-gravity, and in-plane stretching.
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
  // compute orbital radius and dimenionless time
  Real r_star = calcOrbit(time);
  Real tau = calcTau(time, r_star);

  // compute gas specific internal energy over temperature
  // eps / temp = k_B / (gam - 1) / (mu m_H)
  EquationOfState *peos = pmb->peos;
  Units *units = pmb->pmy_mesh->punit;
  Real eps_over_temp = units->k_boltzmann_code / (tde->gamg - 1.0) / (tde->mu * units->hydrogen_mass_code);

  // compute change in log rho due to area factor
  Real logrhodot = 0.0;
  if ( r_star < tde->r_t ) { 
    logrhodot = -logAreaDot(tau, r_star);
  }

  Real x, x0oR, rho, vel, pres, temp, eps, vdot, rhodot;
  for (int i=pmb->is; i<=pmb->ie; i++) {

    // get fluid variables
    x = pmb->pcoord->x1v(i);
    x0oR = pmb->pscalars->r(0, 0, 0, i);
    rho = prim(IDN, 0, 0, i);
    vel = prim(IVX, 0, 0, i);
    pres = prim(IPR, 0, 0, i);

    // compute gas specific internal energy
    temp = peos->TempFromRhoP(rho, pres);
    eps = eps_over_temp * temp;

    // compute velocity and density time derivatives
    vdot = -tde->G * tde->M_BH / (r_star*r_star*r_star) * x; // tidal contribution
    vdot += -interp(x0oR, tde->num_le, 0.0, 1.0, tde->accel_grav); // self-gravity contribution
    rhodot = rho * logrhodot;

    // add source terms
    cons(IDN, 0, 0, i) += dt * rhodot;
    cons(IM1, 0, 0, i) += dt * (rho * vdot + vel * rhodot);
    cons(IEN, 0, 0, i) += dt * (
      rho * vel * vdot 
      + 0.5 * vel*vel * rhodot
      + eps * rhodot
    );
    for (int iscal=0; iscal<NSCALARS; iscal++) {
      cons_scalar(iscal, 0, 0, i) += dt * rhodot * prim_scalar(iscal, 0, 0, i);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real calcRstarOut(MeshBlock *pmb, int iout)
//! \brief calcRstarOut: Compute the radial coordinate of the star in its parabolic orbit.
Real calcRstarOut(MeshBlock *pmb, int iout) {
  Real r_star = calcOrbit(pmb->pmy_mesh->time);
  return r_star * pmb->pmy_mesh->punit->code_length_cgs;
}

//----------------------------------------------------------------------------------------
//! \fn Real calcTauOut(MeshBlock *pmb, int iout)
//! \brief calcTauOut: Compute the dimensionless time.
Real calcTauOut(MeshBlock *pmb, int iout) {
  Real time = pmb->pmy_mesh->time;
  Real r_star = calcOrbit(time);
  return calcTau(time, r_star);
}

//----------------------------------------------------------------------------------------
//! \fn Real calcAreaOut(MeshBlock *pmb, int iout)
//! \brief calcAreaOut: Compute the in-plane stretching factor.
Real calcAreaOut(MeshBlock *pmb, int iout) {
  Real time = pmb->pmy_mesh->time;
  Real r_star = calcOrbit(time);
  Real tau = calcTau(time, r_star);
  return interp(tau, tde->num_tau, -tde->tau_max, tde->tau_max, tde->area);
}

//----------------------------------------------------------------------------------------
//! \fn Real calcRhoMaxOut(MeshBlock *pmb, int iout)
//! \brief calcRhoMaxOut: Compute the maximum density.
Real calcRhoMaxOut(MeshBlock *pmb, int iout) {
  Real rho_max = 0.0;
  for (int i=pmb->is; i<=pmb->ie; i++) {
    rho_max = std::fmax(rho_max, pmb->phydro->w(IDN, 0, 0, i));
  }
  return rho_max;
}

//----------------------------------------------------------------------------------------
//! \fn Real calcPresMaxOut(MeshBlock *pmb, int iout)
//! \brief calcPresMaxOut: Compute the maximum pressure.
Real calcPresMaxOut(MeshBlock *pmb, int iout) {
  Real pres_max = 0.0;
  for (int i=pmb->is; i<=pmb->ie; i++) {
    pres_max = std::fmax(pres_max, pmb->phydro->w(IPR, 0, 0, i));
  }
  return pres_max;
}

Real calcRhoCOut(MeshBlock *pmb, int iout) {
  Real rho_c = 0.0;
  if ( pmb->pcoord->x1f(pmb->is) == 0.0 ) {
    rho_c = pmb->phydro->w(IDN, 0, 0, 0);
  }
  return rho_c;
}

Real calcTempCOut(MeshBlock *pmb, int iout) {
  Real rho_c, pres_c;
  Real temp_c = 0.0;
  if ( pmb->pcoord->x1f(pmb->is) == 0.0 ) {
    rho_c = pmb->phydro->w(IDN, 0, 0, 0);
    pres_c = pmb->phydro->w(IPR, 0, 0, 0);
    temp_c = pmb->peos->TempFromRhoP(rho_c, pres_c);
  }
  return temp_c;
}

template <int n10ths>
Real calc10thCoord(MeshBlock *pmb, int iout) {
  constexpr Real x0_arr[] = {
    0.132728, 0.176674, 0.213271, 0.247949, 0.283294,
    0.321520, 0.365572, 0.421027, 0.503631, 1.000000
  }; // from spherical polytrope
  const Real x0_t = x0_arr[n10ths];
  Real x0_min = pmb->pscalars->r(0, 0, 0, pmb->is);
  Real x0_max = pmb->pscalars->r(0, 0, 0, pmb->ie);
  Real x_t = 0.0, diff = x0_max - x0_min, x0;
  if ( x0_t > x0_min && x0_t < x0_max ) {
    for (int i=pmb->is; i<=pmb->ie; i++) {
      x0 = pmb->pscalars->r(0, 0, 0, i);
      if ( fabs(x0 - x0_t) < diff ) {
        diff = fabs(x0 - x0_t);
        x_t = pmb->pcoord->x1v(i);
      }
    }
  }
  return x_t;
}

template <int n10ths>
Real calc10thRho(MeshBlock *pmb, int iout) {
  // constexpr Real x0_arr[] = {
  //   0.268022, 0.348876, 0.411954, 0.467946, 0.521182,
  //   0.574469, 0.630587, 0.693703, 0.773791, 1.000000
  // }; // from spherical polytrope
  constexpr Real x0_arr[] = {
    0.132728, 0.176674, 0.213271, 0.247949, 0.283294,
    0.321520, 0.365572, 0.421027, 0.503631, 1.000000
  }; // from spherical polytrope
  const Real x0_t = x0_arr[n10ths];
  Real x0_min = pmb->pscalars->r(0, 0, 0, pmb->is);
  Real x0_max = pmb->pscalars->r(0, 0, 0, pmb->ie);
  Real rho_t = 0.0, diff = x0_max - x0_min, x0;
  if ( x0_t > x0_min && x0_t < x0_max ) {
    for (int i=pmb->is; i<=pmb->ie; i++) {
      x0 = pmb->pscalars->r(0, 0, 0, i);
      if ( fabs(x0 - x0_t) < diff ) {
        diff = fabs(x0 - x0_t);
        rho_t = pmb->phydro->w(IDN, 0, 0, i);
      }
    }
  }
  return rho_t;
}

template <int n10ths>
Real calc10thTemp(MeshBlock *pmb, int iout) {
  constexpr Real x0_arr[] = {
    0.132728, 0.176674, 0.213271, 0.247949, 0.283294,
    0.321520, 0.365572, 0.421027, 0.503631, 1.000000
  }; // from spherical polytrope
  const Real x0_t = x0_arr[n10ths];
  Real x0_min = pmb->pscalars->r(0, 0, 0, pmb->is);
  Real x0_max = pmb->pscalars->r(0, 0, 0, pmb->ie);
  Real temp_t = 0.0, diff = x0_max - x0_min, x0;
  Real rho_t = 0.0, pres_t = 0.0;
  if ( x0_t > x0_min && x0_t < x0_max ) {
    for (int i=pmb->is; i<=pmb->ie; i++) {
      x0 = pmb->pscalars->r(0, 0, 0, i);
      if ( fabs(x0 - x0_t) < diff ) {
        diff = fabs(x0 - x0_t);
        rho_t = pmb->phydro->w(IDN, 0, 0, i);
        pres_t = pmb->phydro->w(IPR, 0, 0, i);
        temp_t = pmb->peos->TempFromRhoP(rho_t, pres_t);
      }
    }
  }
  return temp_t;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {

  AllocateUserHistoryOutput(5);
  EnrollUserHistoryOutput(0, calcRstarOut, "r_star", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, calcTauOut, "tau", UserHistoryOperation::max);
  EnrollUserHistoryOutput(2, calcAreaOut, "area", UserHistoryOperation::max);
  EnrollUserHistoryOutput(3, calcRhoMaxOut, "rho_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(4, calcPresMaxOut, "pres_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(5, calcRhoCOut, "rho_c", UserHistoryOperation::max);
  EnrollUserHistoryOutput(6, calcTempCOut, "temp_c", UserHistoryOperation::max);
  EnrollUserHistoryOutput(7, calc10thCoord<0>, "z1t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(8, calc10thCoord<1>, "z2t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(9, calc10thCoord<2>, "z3t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(10, calc10thCoord<3>, "z4t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(11, calc10thCoord<4>, "z5t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(12, calc10thCoord<5>, "z6t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(13, calc10thCoord<6>, "z7t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(14, calc10thCoord<7>, "z8t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(15, calc10thCoord<8>, "z9t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(16, calc10thRho<0>, "rho1t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(17, calc10thRho<1>, "rho2t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(18, calc10thRho<2>, "rho3t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(19, calc10thRho<3>, "rho4t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(20, calc10thRho<4>, "rho5t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(21, calc10thRho<5>, "rho6t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(22, calc10thRho<6>, "rho7t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(23, calc10thRho<7>, "rho8t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(24, calc10thRho<8>, "rho9t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(25, calc10thTemp<0>, "temp1t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(26, calc10thTemp<1>, "temp2t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(27, calc10thTemp<2>, "temp3t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(28, calc10thTemp<3>, "temp4t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(29, calc10thTemp<4>, "temp5t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(30, calc10thTemp<5>, "temp6t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(31, calc10thTemp<6>, "temp7t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(32, calc10thTemp<7>, "temp8t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(33, calc10thTemp<8>, "temp9t", UserHistoryOperation::max);
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
  tde->gamp = 1.0 + 1.0 / tde->n;
  tde->M_BH = tde->Q * tde->M_star;
  tde->r_t = tde->R_star * std::pow(tde->Q, 1.0/3.0);
  tde->r_p = tde->r_t / tde->beta;
  tde->beta_start = tde->beta * pin->GetOrAddReal("problem", "r_start", 1.0);
  tde->tau_max = acosh(sqrt(tde->beta));
  tde->gamg = pin->GetOrAddReal("hydro", "gamma", 5.0/3.0);
  tde->mu = pin->GetOrAddReal("hydro", "mu", 0.6);

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
  xi = dxi;
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
    tde->accel_grav(i) = tde->n * tde->gamp * tde->K / tde->alpha * std::pow(tde->rho_c, tde->gamp - 1.0) * le.phi / (xi*xi);
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

    // compute the star density and pressure
    Real x = pcoord->x1v(i);
    Real rho = interp(x / tde->R_star, tde->num_le, 0.0, 1.0, tde->rho_star);
    Real pres = tde->K * std::pow(rho, tde->gamp);

    // compute the ambient medium density and pressure
    Real dfloor = 1.0e-4 / pmy_mesh->punit->code_density_cgs;
    Real pfloor = 1.0e7 / pmy_mesh->punit->code_pressure_cgs;
    if ( x > tde->R_star ) {
      dfloor *= tde->R_star*tde->R_star / (x*x);
      pfloor *= tde->R_star*tde->R_star / (x*x);
    }

    // set the initial conservative variables
    phydro->u(IDN, 0, 0, i) = fmax(dfloor, rho);
    phydro->u(IM1, 0, 0, i) = 0.0;
    phydro->u(IM2, 0, 0, i) = 0.0;
    phydro->u(IM3, 0, 0, i) = 0.0;
    phydro->u(IEN, 0, 0, i) = fmax(pfloor, pres) / (gamma_gas - 1.0);

    // set the initial passive scalars
    pscalars->s(0, 0, 0, i) = x / tde->R_star * fmax(dfloor, rho);  // initial Lagrangian position
  }

  return;
}
