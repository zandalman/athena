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
  Real n;                    // polytrope index
  int num_time;              // number of samples for affine model evolution
  Real size;                 // initial vertical size of the stream
  Real gm1;                  // gamma minus one
  Real dfloor;               // density floor
  Real pfloor;               // pressure floor
  AthenaArray<Real> time;    // time
  AthenaArray<Real> r;       // radius
  AthenaArray<Real> area;    // area factor
  AthenaArray<Real> areadot; // area factor derivative
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
  Real x;      // x-coordinate
  Real xdot;   // x-velocity
  Real y;      // y-coordinate
  Real ydot;   // y-velocity
  Real lam;    // longitudial stretching
  Real Om;     // angular frequency
  Real L;      // longitudinal length
  Real alpha;  // orientation
  Real Dlt;    // horizontal size
  Real Dltdot; // horizontal stretching
  Real vpar;   // in-plane shear

  // Overload the + operator
  affineModel operator+(const affineModel& other) const {
    affineModel am;
    am.x = x + other.x;
    am.xdot = xdot + other.xdot;
    am.y = y + other.y;
    am.ydot = ydot + other.ydot;
    am.lam = lam + other.lam;
    am.Om = Om + other.Om;
    am.L = L + other.L;
    am.alpha = alpha + other.alpha;
    am.Dlt = Dlt + other.Dlt;
    am.Dltdot = Dltdot + other.Dltdot;
    am.vpar = vpar + other.vpar;
    return am;
  }

  // Overload the * operator
  affineModel operator*(Real scalar) const {
    affineModel am;
    am.x = scalar * x;
    am.xdot = scalar * xdot;
    am.y = scalar * y;
    am.ydot = scalar * ydot;
    am.lam = scalar * lam;
    am.Om = scalar * Om;
    am.L = scalar * L;
    am.alpha = scalar * alpha;
    am.Dlt = scalar * Dlt;
    am.Dltdot = scalar * Dltdot;
    am.vpar = scalar * vpar;
    return am;
  }

  friend affineModel operator*(Real scalar, const affineModel& am) {
    return am * scalar;
  }
};

//----------------------------------------------------------------------------------------
//! \fn void calcLaneEmden(const Real xi, const laneEmden le, laneEmden &dledxi)
//! \brief calcLaneEmden: Compute derivatives in the cylindrical Lane-Emden equation with respect to xi.
//! \param xi      Dimensionless radius
//! \param le      Lane-Emden parameters
//! \param dledxi  Lane-Emden parameter derivatives
void calcLaneEmden(const Real xi, const laneEmden le, laneEmden &dledxi) {
  dledxi.th = -le.phi / xi;
  dledxi.phi = std::pow(fmax(le.th, 0.0), tde->n) * xi;
}

//----------------------------------------------------------------------------------------
//! \fn void calcAffineModel(Real time, const affineModel &am, affineModel &amdot)
//! \brief calcAffineModel: Compute the derivatives in the affine model with respect to time.
//! \param time   Time
//! \param am     Affine model parameters
//! \param amdot  Affine model parameter derivatives
void calcAffineModel(Real time, const affineModel &am, affineModel &amdot) {
  
  Real r = sqrt(am.x*am.x + am.y*am.y);
  Real cos_th = am.x / r;
  Real sin_th = am.y / r;
  Real cos_alpha = cos(am.alpha);
  Real sin_alpha = sin(am.alpha);
  Real cos_dlt = cos_th * cos_alpha + sin_th * sin_alpha;
  Real sin_dlt = sin_th * cos_alpha - cos_th * sin_alpha;

  amdot.x = am.xdot;
  amdot.y = am.ydot;
  amdot.xdot = -am.x/(r*r*r);
  amdot.ydot = -am.y/(r*r*r);
  amdot.lam = am.Om*am.Om - am.lam*am.lam - (1.0 - 3.0 * cos_dlt*cos_dlt) / (r*r*r);
  amdot.Om = -2.0 * am.lam * am.Om + 3.0/(r*r*r) * cos_dlt * sin_dlt;
  amdot.L = am.lam * am.L;
  amdot.alpha = am.Om;
  amdot.Dlt = am.Dltdot;
  amdot.Dltdot = -am.Dlt * (1.0 - 3.0 * sin_dlt*sin_dlt) / (r*r*r) - am.Om*am.Om * am.Dlt - 2.0 * am.Om * am.vpar;
  amdot.vpar = 3.0/(r*r*r) * am.Dlt * cos_dlt * sin_dlt + am.Dltdot * am.Om - am.lam * (am.Om * am.Dlt + am.vpar);
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

void calcIparam(const Real x, const int size, const AthenaArray<Real> &arrx, int &idx, Real &iparam) {
  if ( x <= arrx(0) ) {
    // clamp values below array range
    idx = 0, iparam = 0.0;
  } else if ( x >= arrx(size - 1)) {
    // clamp values above array range
    idx = size - 2, iparam = 1.0;
  } else {
    // binary search
    idx = 0;
    int right = size - 1, mid;
    while ( right - idx > 1 ) {
      mid = (idx + right) / 2;
      if ( arrx(mid) <= x ) idx = mid;
      else right = mid;
    }
    // calculate interpolation parameter
    iparam = (x - arrx(idx)) / (arrx(idx+1) - arrx(idx));
  }
}

Real interp(const int idx, const Real iparam, const AthenaArray<Real> &arry) {
  return arry(idx) + iparam * (arry(idx+1) - arry(idx));
}

Real angleToTime(const double f, const double ecc, const double a, const double per) {
  Real cos_u = (cos(f) + ecc) / (1.0 + ecc * cos(f));
  Real sin_u = sqrt(1.0 - cos_u*cos_u);
  Real u = atan2(sin_u, cos_u);
  Real sgn = static_cast<Real>(2 * (f > 0.0) - 1);
  Real num_per = floor(0.5 + f / (2.0 * M_PI));
  return sgn * (u - ecc * sin_u) * sqrt(a*a*a) + num_per * per;
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

  int idx;
  Real iparam;
  calcIparam(time, tde->num_time, tde->time, idx, iparam);
  Real r = interp(idx, iparam, tde->r);
  Real area = interp(idx, iparam, tde->area);
  Real areadot = interp(idx, iparam, tde->areadot);
  Real x, rho, vel, pres, vdot, rhodot;

  for (int i=pmb->is; i<=pmb->ie; i++) {
    
    x = pmb->pcoord->x1v(i);
    rho = prim(IDN, 0, 0, i);
    vel = prim(IVX, 0, 0, i);
    pres = prim(IPR, 0, 0, i);
    vdot = -x / (r*r*r);
    rhodot = -rho * areadot / area;

    // don't apply source function to floor material
    if ( rho < tde->dfloor ) continue;

    cons(IDN, 0, 0, i) += dt * rhodot;
    cons(IM1, 0, 0, i) += dt * (rho * vdot + vel * rhodot);
    cons(IEN, 0, 0, i) += dt * (
      rho * vel * vdot 
      + 0.5 * vel*vel * rhodot
      + pres / tde->gm1 * rhodot / rho
    );

    for (int iscal=0; iscal<NSCALARS; iscal++) {
      cons_scalar(iscal, 0, 0, i) += dt * rhodot * prim_scalar(iscal, 0, 0, i);
    }
  }

}

Real calcEdotsrc(MeshBlock *pmb, int iout) {
  
  int idx;
  Real iparam;
  // Real dt = pmb->pmy_mesh->dt;
  Real time = pmb->pmy_mesh->time;
  calcIparam(time, tde->num_time, tde->time, idx, iparam);
  Real r = interp(idx, iparam, tde->r);
  Real area = interp(idx, iparam, tde->area);
  Real areadot = interp(idx, iparam, tde->areadot);
  Real x, dx, rho, vel, pres, vdot, rhodot;
  Real Edotsrc = 0.0;
  
  for (int i=pmb->is; i<=pmb->ie; i++) {
    
    x = pmb->pcoord->x1v(i);
    dx = pmb->pcoord->x1f(i+1) - pmb->pcoord->x1f(i);
    rho = pmb->phydro->w(IDN, 0, 0, i);
    vel = pmb->phydro->w(IVX, 0, 0, i);
    pres = pmb->phydro->w(IPR, 0, 0, i);
    vdot = -x / (r*r*r);
    rhodot = -rho * areadot / area;

    // don't apply source function to floor material
    if ( rho < tde->dfloor ) continue;

    Edotsrc += -dx * (
      rho * vel * vdot
      + 0.5 * vel*vel * rhodot
      + pres / tde->gm1 * rhodot / rho
    );
  }
  
  return Edotsrc;
}

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

Real calcEth(MeshBlock *pmb, int iout) {
  Real dx, pres;
  Real Eth = 0.0;
  for (int i=pmb->is; i<=pmb->ie; i++) {
    dx = pmb->pcoord->x1f(i+1) - pmb->pcoord->x1f(i);
    pres = pmb->phydro->w(IPR, 0, 0, i);
    Eth += dx * pres / tde->gm1;
  }
  return Eth;
}

Real calcS(MeshBlock *pmb, int iout) {
  Real dx, rho, pres;
  Real mass = 0.0;
  Real S = 0.0;
  for (int i=pmb->is; i<=pmb->ie; i++) {
    dx = pmb->pcoord->x1f(i+1) - pmb->pcoord->x1f(i);
    rho = pmb->phydro->w(IDN, 0, 0, i);
    pres = pmb->phydro->w(IPR, 0, 0, i);
    S += dx * pres / pow(rho, tde->gm1);
  }
  return S;
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

Real calcPresCOut(MeshBlock *pmb, int iout) {
  Real pres_c = 0.0;
  if ( pmb->pcoord->x1f(pmb->is) == 0.0 ) {
    pres_c = pmb->phydro->w(IPR, 0, 0, 0);
  }
  return pres_c;
}

template <int n10ths>
Real calc10thCoord(MeshBlock *pmb, int iout) {
  constexpr Real x0_arr[] = {
    0.177623, 0.256908, 0.322507, 0.382746, 0.441380,
    0.501125, 0.564952, 0.637650, 0.731012, 1.000000
  }; // from cylindrical polytrope
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

void Mesh::InitUserMeshData(ParameterInput *pin) {

  AllocateUserHistoryOutput(17);
  EnrollUserHistoryOutput(0, calcRhoMaxOut, "rho_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, calcPresMaxOut, "pres_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(2, calcRhoCOut, "rho_c", UserHistoryOperation::max);
  EnrollUserHistoryOutput(3, calcPresCOut, "pres_c", UserHistoryOperation::max);
  EnrollUserHistoryOutput(4, calcEdotsrc, "Edotsrc", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(5, calcEkin, "Ekin", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(6, calcEth, "Eth", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(7, calcS, "S", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(8, calc10thCoord<0>, "z1t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(9, calc10thCoord<1>, "z2t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(10, calc10thCoord<2>, "z3t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(11, calc10thCoord<3>, "z4t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(12, calc10thCoord<4>, "z5t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(13, calc10thCoord<5>, "z6t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(14, calc10thCoord<6>, "z7t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(15, calc10thCoord<7>, "z8t", UserHistoryOperation::max);
  EnrollUserHistoryOutput(16, calc10thCoord<8>, "z9t", UserHistoryOperation::max);
  EnrollUserExplicitSourceFunction(tdeSrcFunc);

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Problem Generator for tidal disruption event problems
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  std::stringstream msg;
  
  // floors
  tde->dfloor = 1.0e-10;
  tde->pfloor = 5.0e-20;
  
  // populate the affine model struct
  affineModel am;
  am.x = pin->GetReal("problem", "x0");
  am.xdot = pin->GetReal("problem", "vx0");
  am.y = pin->GetReal("problem", "y0");
  am.ydot = pin->GetReal("problem", "vy0");
  am.lam = pin->GetReal("problem", "lam0");
  am.Om = pin->GetReal("problem", "Om0");
  am.L = pin->GetReal("problem", "L0");
  am.alpha = pin->GetReal("problem", "alpha0");
  am.Dlt = pin->GetReal("problem", "Dlt0");
  am.Dltdot = pin->GetReal("problem", "Dltdot0");
  am.vpar = pin->GetReal("problem", "vpar0");

  // retrieve other parameters
  Real R_star = pin->GetReal("problem", "Rstar");
  Real rho0 = pin->GetReal("problem", "rho0");
  Real H0 = pin->GetReal("problem", "H0");
  Real Hdot0 = pin->GetReal("problem", "Hdot0");
  Real K0 = pin->GetReal("problem", "K0");
  Real gamg = pin->GetOrAddReal("hydro", "gamma", 5.0/3.0);
  Real fmax = M_PI;
  tde->size = R_star * H0;
  tde->gm1 = gamg - 1.0;

  // compute orbital parameters
  Real r0 = sqrt(am.x*am.x + am.y*am.y);
  Real vel0 = sqrt(am.xdot*am.xdot + am.ydot*am.ydot);
  Real eps = -1.0/r0 + 0.5 * vel0*vel0;
  Real h = am.x * am.ydot - am.y * am.xdot;
  Real ecc = sqrt(1.0 + 2.0 * eps * h*h);
  Real a = -1.0 / (2.0 * eps);
  Real per = 2.0 * M_PI * sqrt(a*a*a);
  Real f0 = -acos((a * (1.0 - ecc*ecc) / r0 - 1.0) / ecc);
  Real time0 = angleToTime(f0, ecc, a, per);

  // create arrays for stream slice profile
  int num_le = 16384;
  AthenaArray<Real> r_sl, rho_sl;
  r_sl.NewAthenaArray(num_le);
  rho_sl.NewAthenaArray(num_le);

  // define polytrope parameters
  tde->n = 1.5;
  Real gam = 1.0 + 1.0/tde->n;
  const Real xi_max = 2.647; // from cylindrical polytrope
  Real alpha = tde->size / xi_max;

  // compute stream slice profile
  laneEmden le;
  Real dxi, xi;
  dxi = xi_max / static_cast<Real>(num_le - 1);
  r_sl(0) = 0.0;
  rho_sl(0) = rho0 / (am.L * H0 * am.Dlt);
  le.th = 1.0;
  le.phi = 0.0;
  for ( int i=1; i<num_le; i++ ) {
    xi = dxi + static_cast<Real>(i) * dxi;
    r_sl(i) = alpha * xi;
    rho_sl(i) = rho_sl(0) * std::pow(std::fmax(le.th, 0.0), tde->n);
    rk4<decltype(calcLaneEmden), laneEmden>(calcLaneEmden, dxi, xi, le);
  }

  // create arrays for affine model
  const int fine_ratio = 4;
  tde->num_time = 16384;
  tde->time.NewAthenaArray(tde->num_time);
  tde->r.NewAthenaArray(tde->num_time);
  tde->area.NewAthenaArray(tde->num_time);
  tde->areadot.NewAthenaArray(tde->num_time);

  // set initial values
  tde->time(0) = time0;
  tde->r(0) = r0;
  tde->area(0) = am.Dlt * am.L;
  tde->areadot(0) = tde->area(0) * (am.Dltdot / am.Dlt + am.lam);

  // compute the affine model
  Real df = (fmax - f0) / static_cast<Real>(fine_ratio * tde->num_time - 1);
  Real f, time, time_old, dt;
  time = time0;
  for ( int i=1; i<tde->num_time; i++ ) {
    for ( int j=0; j<fine_ratio; j++ ) {
      if ( i == 1 and j == 0 ) continue;
      f = f0 + static_cast<Real>(fine_ratio * (i - 1) + j) * df;
      time_old = time;
      time = angleToTime(f, ecc, a, per);
      dt = time - time_old;
      rk4<decltype(calcAffineModel), affineModel>(calcAffineModel, dt, time_old, am);
    }
    tde->time(i) = time;
    tde->r(i) = sqrt(am.x*am.x + am.y*am.y);
    tde->area(i) = am.Dlt * am.L;
    tde->areadot(i) = tde->area(i) * (am.Dltdot / am.Dlt + am.lam);
  }
  
  for (int i=is; i<=ie; i++) {
    
    // compute the star density and pressure
    int idx;
    Real iparam;
    Real x = pcoord->x1v(i);
    calcIparam(x, num_le, r_sl, idx, iparam);
    Real rho = interp(idx, iparam, rho_sl);
    Real pres = K0 * std::pow(rho, gam);
    Real vel = Hdot0 / H0 * x;

    // compute the ambient medium density and pressure (may need to adjust these)
    Real dfloor = tde->dfloor;
    Real pfloor = tde->pfloor;
    if ( x > R_star * H0 ) {
      dfloor *= R_star*R_star * H0*H0 / (x*x);
      pfloor *= R_star*R_star * H0*H0 / (x*x);
    }
    rho = std::fmax(dfloor, rho);
    pres = std::fmax(pfloor, pres);

    // set the initial conservative variables
    phydro->u(IDN, 0, 0, i) = rho;
    phydro->u(IM1, 0, 0, i) = rho * vel;
    phydro->u(IM2, 0, 0, i) = 0.0;
    phydro->u(IM3, 0, 0, i) = 0.0;
    phydro->u(IEN, 0, 0, i) = 0.5 * rho * vel*vel + pres / tde->gm1;

    // set the initial passive scalars
    pscalars->s(0, 0, 0, i) = x / tde->size * rho;  // initial Lagrangian position
}

  return;
}
