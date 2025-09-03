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
  std::array<Real, 12> x0_t_list;  // mass coordinates at which to record outputs
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
  
  // initialize variables
  Real z, x0oR, rho, vel, pres, temp, egas, vdot, rhodot, q;
  EquationOfState *peos = pmb->peos;
  
  // calculate affine model parameters
  Real r_star = calcOrbit(time);
  Real tau = calcTau(time, r_star);
  Real logrhodot = r_star < tde->r_t ? -logAreaDot(tau, r_star) : 0.0;

  for (int i=pmb->is; i<=pmb->ie; i++) {

    // get primatives
    z = pmb->pcoord->x1v(i);
    rho = prim(IDN, 0, 0, i);
    vel = prim(IVX, 0, 0, i);
    pres = prim(IPR, 0, 0, i);
    egas = pmb->peos->EgasFromRhoP(rho, pres);
    q = 1.0 + pres / egas; // d(lnegas)/d(lnrho)|s
    x0oR = pmb->pscalars->r(0, 0, 0, i);

    // compute time derivatives
    vdot = -tde->G * tde->M_BH * z / (r_star*r_star*r_star);       // tides
    vdot += -interp(x0oR, tde->num_le, 0.0, 1.0, tde->accel_grav); // self-gravity
    rhodot = rho * logrhodot;

    // add source terms
    cons(IDN, 0, 0, i) += dt * rhodot;
    cons(IM1, 0, 0, i) += dt * (rho * vdot + vel * rhodot);
    cons(IEN, 0, 0, i) += dt * (
      rho * vel * vdot 
      + 0.5 * vel*vel * rhodot
      + egas * rhodot / rho * q // extra factor accounts for d(lneps)/d(lnrho)|s
    );
    for (int iscal=0; iscal<NSCALARS; iscal++) {
      cons_scalar(iscal, 0, 0, i) += dt * rhodot * prim_scalar(iscal, 0, 0, i);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real calcEdotTide(MeshBlock *pmb, int iout)
//! \brief calcEdotTide: Calculate the energy source term contribution from tides.
Real calcEdotTide(MeshBlock *pmb, int iout) {
  
  // initialize variables
  Real z, dz, rho, vel, vdot;
  Real Edot = 0.0;
  
  // calculate affine model parameters
  Real time = pmb->pmy_mesh->time;
  Real r_star = calcOrbit(time);

  // loop over cells
  for (int i=pmb->is; i<=pmb->ie; i++) {
    
    // get primitives
    z = pmb->pcoord->x1v(i);
    dz = pmb->pcoord->x1f(i+1) - pmb->pcoord->x1f(i);
    rho = pmb->phydro->w(IDN, 0, 0, i);
    vel = pmb->phydro->w(IVX, 0, 0, i);

    // compute Edot
    vdot = -tde->G * tde->M_BH * z / (r_star*r_star*r_star);
    Edot += -dz * rho * vel * vdot;
  }
  return Edot;
}

//----------------------------------------------------------------------------------------
//! \fn Real calcEdotGrav(MeshBlock *pmb, int iout)
//! \brief calcEdotGrav: Calculate the energy source term contribution from self-gravity.
Real calcEdotGrav(MeshBlock *pmb, int iout) {
  
  // initialize variables
  Real z, dz, rho, vel, vdot, x0oR;
  Real Edot = 0.0;

  // loop over cells
  for (int i=pmb->is; i<=pmb->ie; i++) {
    
    // get primitives
    z = pmb->pcoord->x1v(i);
    dz = pmb->pcoord->x1f(i+1) - pmb->pcoord->x1f(i);
    rho = pmb->phydro->w(IDN, 0, 0, i);
    vel = pmb->phydro->w(IVX, 0, 0, i);
    x0oR = pmb->pscalars->r(0, 0, 0, i);

    // compute Edot
    vdot = -interp(x0oR, tde->num_le, 0.0, 1.0, tde->accel_grav);
    Edot += -dz * rho * vel * vdot;
  }
  return Edot;
}

//----------------------------------------------------------------------------------------
//! \fn Real calcEdotArea(MeshBlock *pmb, int iout)
//! \brief calcEdotArea: Calculate the energy source term contribution from in-plane stretching.
Real calcEdotArea(MeshBlock *pmb, int iout) {
  
  // initialize variables
  Real z, dz, rho, vel, pres, egas, q, rhodot;
  Real Edot = 0.0;

  // calculate affine model parameters
  Real time = pmb->pmy_mesh->time;
  Real r_star = calcOrbit(time);
  Real tau = calcTau(time, r_star);
  Real logrhodot = r_star < tde->r_t ? -logAreaDot(tau, r_star) : 0.0;

  // loop over cells
  for (int i=pmb->is; i<=pmb->ie; i++) {
    
    // get primitives
    z = pmb->pcoord->x1v(i);
    dz = pmb->pcoord->x1f(i+1) - pmb->pcoord->x1f(i);
    rho = pmb->phydro->w(IDN, 0, 0, i);
    vel = pmb->phydro->w(IVX, 0, 0, i);
    pres = pmb->phydro->w(IPR, 0, 0, i);
    egas = pmb->peos->EgasFromRhoP(rho, pres);
    q = 1.0 + pres / egas;

    // compute Edot
    rhodot = rho * logrhodot;
    Edot += -dz * rhodot * (0.5 * vel*vel + egas / rho * q);
  }
  return Edot;
}

//----------------------------------------------------------------------------------------
//! \fn Real calcEkin(MeshBlock *pmb, int iout)
//! \brief calcEkin: Calculate the total kinetic energy.
Real calcEkin(MeshBlock *pmb, int iout) {
  Real dx, rho, vel;
  Real Ekin = 0.0;
  for (int i=pmb->is; i<=pmb->ie; i++) {
    dx = pmb->pcoord->x1f(i+1) - pmb->pcoord->x1f(i);
    rho = pmb->phydro->w(IDN, 0, 0, i);
    vel = pmb->phydro->w(IVX, 0, 0, i);
    Ekin += dx * 0.5 * rho * vel*vel;
  }
  return Ekin;
}

//----------------------------------------------------------------------------------------
//! \fn Real calcEth(MeshBlock *pmb, int iout)
//! \brief calcEth: Calculate the total thermal energy.
Real calcEth(MeshBlock *pmb, int iout) {
  EquationOfState *peos = pmb->peos;
  Real dx, rho, pres;
  Real Eth = 0.0;
  for (int i=pmb->is; i<=pmb->ie; i++) {
    dx = pmb->pcoord->x1f(i+1) - pmb->pcoord->x1f(i);
    rho = pmb->phydro->w(IDN, 0, 0, i);
    pres = pmb->phydro->w(IPR, 0, 0, i);
    Eth += dx * peos->EgasFromRhoP(rho, pres);
  }
  return Eth;
}

//----------------------------------------------------------------------------------------
//! \fn Real calcRstarOut(MeshBlock *pmb, int iout)
//! \brief calcRstarOut: Compute the radial coordinate of the star in its parabolic orbit.
Real calcRstarOut(MeshBlock *pmb, int iout) {
  return calcOrbit(pmb->pmy_mesh->time);
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

Real calcRhoC(MeshBlock *pmb, int iout) {
  Real rho_c = 0.0;
  if ( pmb->pcoord->x1f(pmb->is) == 0.0 ) {
    rho_c = pmb->phydro->w(IDN, 0, 0, 0);
  }
  return rho_c;
}

Real calcPresC(MeshBlock *pmb, int iout) {
  Real pres_c = 0.0;
  if ( pmb->pcoord->x1f(pmb->is) == 0.0 ) {
    pres_c = pmb->phydro->w(IPR, 0, 0, 0);
  }
  return pres_c;
}

Real calcTempC(MeshBlock *pmb, int iout) {
  Real rho_c, pres_c;
  Real temp_c = 0.0;
  if ( pmb->pcoord->x1f(pmb->is) == 0.0 ) {
    rho_c = pmb->phydro->w(IDN, 0, 0, 0);
    pres_c = pmb->phydro->w(IPR, 0, 0, 0);
    temp_c = pmb->peos->TempFromRhoP(rho_c, pres_c);
  }
  return temp_c;
}

template <int idx>
Real calcCoordLag(MeshBlock *pmb, int iout) {
  
  // initialize variables
  const Real x0_t = tde->x0_t_list[idx];
  Real x0_min = pmb->pscalars->r(0, 0, 0, pmb->is);
  Real x0_max = pmb->pscalars->r(0, 0, 0, pmb->ie);
  Real x = 0.0;
  Real x_prev = 0.0;
  Real x_t = 0.0;
  Real x0, x0_prev, iparam;

  // only proceed if meshblock contains target Lagrangian coordinate
  if ( x0_t >= x0_min && x0_t <= x0_max ) {

    // find mass coordinate
    for (int i=pmb->is; i<=pmb->ie; i++) {
      x0 = pmb->pscalars->r(0, 0, 0, i);
      x = pmb->pcoord->x1v(i);
      if ( x0 > x0_t ) break;
      x0_prev = x0;
      x_prev = x;
    }

    // interpolate value
    iparam = (x0_t - x0_prev) / (x0 - x0_prev);
    x_t = x_prev * (1.0 - iparam) + x * iparam;

  }

  return x_t;
}

template <int idx>
Real calcRhoLag(MeshBlock *pmb, int iout) {
  
  // initialize variables
  const Real x0_t = tde->x0_t_list[idx];
  Real x0_min = pmb->pscalars->r(0, 0, 0, pmb->is);
  Real x0_max = pmb->pscalars->r(0, 0, 0, pmb->ie);
  Real rho = 0.0;
  Real rho_prev = 0.0;
  Real rho_t = 0.0;
  Real x0, x0_prev, iparam;

  // only proceed if meshblock contains target Lagrangian coordinate
  if ( x0_t > x0_min && x0_t < x0_max ) {

    // find mass coordinate
    for (int i=pmb->is; i<=pmb->ie; i++) {
      x0 = pmb->pscalars->r(0, 0, 0, i);
      rho = pmb->phydro->w(IDN, 0, 0, i);
      if ( x0 > x0_t ) break;
      x0_prev = x0;
      rho_prev = rho;
    }

    // interpolate value
    iparam = (x0_t - x0_prev) / (x0 - x0_prev);
    rho_t = rho_prev * (1.0 - iparam) + rho * iparam;

  }

  return rho_t;
}

template <int idx>
Real calcPresLag(MeshBlock *pmb, int iout) {
  
  // initialize variables
  const Real x0_t = tde->x0_t_list[idx];
  Real x0_min = pmb->pscalars->r(0, 0, 0, pmb->is);
  Real x0_max = pmb->pscalars->r(0, 0, 0, pmb->ie);
  Real pres = 0.0;
  Real pres_prev = 0.0;
  Real pres_t = 0.0;
  Real x0, x0_prev, iparam;

  // only proceed if meshblock contains target Lagrangian coordinate
  if ( x0_t > x0_min && x0_t < x0_max ) {

    // find mass coordinate
    for (int i=pmb->is; i<=pmb->ie; i++) {
      x0 = pmb->pscalars->r(0, 0, 0, i);
      pres = pmb->phydro->w(IPR, 0, 0, i);
      if ( x0 > x0_t ) break;
      x0_prev = x0;
      pres_prev = pres;
    }

    // interpolate value
    iparam = (x0_t - x0_prev) / (x0 - x0_prev);
    pres_t = pres_prev * (1.0 - iparam) + pres * iparam;

  }

  return pres_t;
}

template <int idx>
Real calcTempLag(MeshBlock *pmb, int iout) {
  
  // initialize variables
  const Real x0_t = tde->x0_t_list[idx];
  Real x0_min = pmb->pscalars->r(0, 0, 0, pmb->is);
  Real x0_max = pmb->pscalars->r(0, 0, 0, pmb->ie);
  Real temp = 0.0;
  Real temp_prev = 0.0;
  Real temp_t = 0.0;
  Real rho, pres;
  Real x0, x0_prev, iparam;

  // only proceed if meshblock contains target Lagrangian coordinate
  if ( x0_t > x0_min && x0_t < x0_max ) {

    // find mass coordinate
    for (int i=pmb->is; i<=pmb->ie; i++) {
      x0 = pmb->pscalars->r(0, 0, 0, i);
      rho = pmb->phydro->w(IDN, 0, 0, i);
      pres = pmb->phydro->w(IPR, 0, 0, i);
      temp = pmb->peos->TempFromRhoP(rho, pres);
      if ( x0 > x0_t ) break;
      x0_prev = x0;
      temp_prev = temp;
    }

    // interpolate value
    iparam = (x0_t - x0_prev) / (x0 - x0_prev);
    temp_t = temp_prev * (1.0 - iparam) + temp * iparam;

  }

  return temp_t;
}

template <int idx>
void EnrollCalcCoord(Mesh *pmy_mesh, int num_out) {
  std::ostringstream label;
  label << "z" << idx;
  pmy_mesh->EnrollUserHistoryOutput(num_out-1 + idx, calcCoordLag<idx>, label.str().c_str(), UserHistoryOperation::max);
  EnrollCalcCoord<idx-1>(pmy_mesh, num_out);
}

template <>
void EnrollCalcCoord<0>(Mesh *pmy_mesh, int num_out) {}

template <int idx>
void EnrollCalcRho(Mesh *pmy_mesh, int num_out) {
  std::ostringstream label;
  label << "rho" << idx;
  pmy_mesh->EnrollUserHistoryOutput(num_out-1 + idx, calcRhoLag<idx>, label.str().c_str(), UserHistoryOperation::max);
  EnrollCalcRho<idx-1>(pmy_mesh, num_out);
}

template <>
void EnrollCalcRho<0>(Mesh *pmy_mesh, int num_out) {}

template <int idx>
void EnrollCalcPres(Mesh *pmy_mesh, int num_out) {
  std::ostringstream label;
  label << "pres" << idx;
  pmy_mesh->EnrollUserHistoryOutput(num_out-1 + idx, calcPresLag<idx>, label.str().c_str(), UserHistoryOperation::max);
  EnrollCalcPres<idx-1>(pmy_mesh, num_out);
}

template <>
void EnrollCalcPres<0>(Mesh *pmy_mesh, int num_out) {}

template <int idx>
void EnrollCalcTemp(Mesh *pmy_mesh, int num_out) {
  std::ostringstream label;
  label << "temp" << idx;
  pmy_mesh->EnrollUserHistoryOutput(num_out-1 + idx, calcTempLag<idx>, label.str().c_str(), UserHistoryOperation::max);
  EnrollCalcTemp<idx-1>(pmy_mesh, num_out);
}

template <>
void EnrollCalcTemp<0>(Mesh *pmy_mesh, int num_out) {}

void Mesh::InitUserMeshData(ParameterInput *pin) {

  constexpr int num_out = 11;
  AllocateUserHistoryOutput(num_out + 12 + 12 + 12 + 12);
  EnrollUserHistoryOutput(0, calcRstarOut, "r_star", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, calcTauOut, "tau", UserHistoryOperation::max);
  EnrollUserHistoryOutput(2, calcAreaOut, "area", UserHistoryOperation::max);
  EnrollUserHistoryOutput(3, calcRhoC, "rho_c", UserHistoryOperation::max);
  EnrollUserHistoryOutput(4, calcPresC, "pres_c", UserHistoryOperation::max);
  EnrollUserHistoryOutput(5, calcTempC, "temp_c", UserHistoryOperation::max);
  EnrollUserHistoryOutput(6, calcEdotTide, "Edot_tide", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(7, calcEdotGrav, "Edot_grav", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(8, calcEdotArea, "Edot_area", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(9, calcEkin, "Ekin", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(10, calcEth, "Eth", UserHistoryOperation::sum);
  EnrollCalcCoord<12>(this, num_out);
  EnrollCalcRho<12>(this, num_out + 12);
  EnrollCalcPres<12>(this, num_out + 12 + 12);
  EnrollCalcTemp<12>(this, num_out + 12 + 12 + 12);
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
  le.th = 1.0 - 1.0/6.0 * dxi;
  le.phi = -1.0/3.0 * dxi*dxi*dxi;
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
  le.th = 1.0 - 1.0/6.0 * dxi;
  le.phi = -1.0/3.0 * dxi*dxi*dxi;
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
  if ( tde->n == 1.5 ) {
    tde->x0_t_list = {
      0.2680, 0.3489, 0.4120, 0.4680, 0.5212, 
      0.5745, 0.6306, 0.6937, 0.7738, 0.8311,
      0.9128, 0.9656
    };
  } else if ( tde->n == 3.0 ) {
    tde->x0_t_list = {
      0.1327, 0.1767, 0.2133, 0.2479, 0.2833,
      0.3215, 0.3656, 0.4210, 0.5036, 0.5748,
      0.7057, 0.8294
    };
  }

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
    Real dx = pcoord->x1f(i+1) - pcoord->x1f(i);
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
