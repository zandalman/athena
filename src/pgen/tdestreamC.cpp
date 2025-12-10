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
#include <fstream>
#include <stdexcept> 
#include <string>
#include <hdf5.h>

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

#if SINGLE_PRECISION_ENABLED
#define H5T_REAL H5T_NATIVE_FLOAT
#else
#define H5T_REAL H5T_NATIVE_DOUBLE
#endif

struct pgentde {
  Real n;                        // polytrope index
  int num_time;                  // number of samples for affine model evolution
  Real dfloor;                   // density floor
  Real pfloor;                   // pressure floor
  Real mtot;                     // total mass
  Real rho_init;                 // initial density
  Real area0;                    // initial area factor
  AthenaArray<Real> time;        // time
  AthenaArray<Real> r;           // radius
  AthenaArray<Real> area;        // area factor
  AthenaArray<Real> areadot;     // area factor time derivative
  std::array<Real, 11> mcoord;   // mass coordinates at which to record outputs
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
  Real r;      // radius
  Real f;      // true anomaly
  Real lam;    // longitudial stretching
  Real Om;     // angular frequency
  Real L;      // longitudinal length
  Real alpha;  // orientation
  Real H;      // vertical size
  Real Hdot;   // vertical stretching
  Real Dlt;    // horizontal size
  Real Dltdot; // horizontal stretching
  Real vpar;   // in-plane shear
  Real K;

  // Overload the + operator
  affineModel operator+(const affineModel& other) const {
    affineModel am;
    am.r = r;
    am.f = f;
    am.lam = lam + other.lam;
    am.Om = Om + other.Om;
    am.L = L + other.L;
    am.alpha = alpha + other.alpha;
    am.H = H + other.H;
    am.Hdot = Hdot + other.Hdot;
    am.Dlt = Dlt + other.Dlt;
    am.Dltdot = Dltdot + other.Dltdot;
    am.vpar = vpar + other.vpar;
    am.K = K + other.K;
    return am;
  }

  // Overload the * operator
  affineModel operator*(Real scalar) const {
    affineModel am;
    am.r = r;
    am.f = f;
    am.lam = scalar * lam;
    am.Om = scalar * Om;
    am.L = scalar * L;
    am.alpha = scalar * alpha;
    am.H = scalar * H;
    am.Hdot = scalar * Hdot;
    am.Dlt = scalar * Dlt;
    am.Dltdot = scalar * Dltdot;
    am.vpar = scalar * vpar;
    am.K = scalar * K;
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

template <typename Callable, typename T>
void rk4(Callable dydx, const Real dx, Real &x, T &y, EquationOfState* peos) {
  const Real dx_h = dx / 2.0;
  T k1, k2, k3, k4;
  dydx(x,        y,             k1, peos);
  dydx(x + dx_h, y + k1 * dx_h, k2, peos);
  dydx(x + dx_h, y + k2 * dx_h, k3, peos);
  dydx(x + dx,   y + k3 * dx,   k4, peos);
  y = y + dx / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
  x = x + dx;
}

//----------------------------------------------------------------------------------------
//! \fn void calcIparam(const Real x, const int size, const AthenaArray<Real> &arrx, int &idx, Real &iparam)
//! \brief calcIparam: Compute the interpolation parameter and index using a binary search.
//! Given an array arry of y-values, the interpolated y-value is 
//! y = (1.0 - iparam) * arry(idx) + iparam * arry(idx+1)
//! \param x       The x-value to interpolate
//! \param size    The size of the array
//! \param arrx    The array of x-values to interpolate
//! \param idx     The interpolation index
//! \param iparam  The interpolation parameter
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

//----------------------------------------------------------------------------------------
//! \fn Real interp(const int idx, const Real iparam, const AthenaArray<Real> &arry)
//! \brief interp: Interpolate an array.
//! \param idx    The interpolation index
//! \param iparam The interpolation parameter
//! \param arry   The array of y-values
//! \return The interpolated y-value
Real interp(const int idx, const Real iparam, const AthenaArray<Real> &arry) {
  return arry(idx) + iparam * (arry(idx+1) - arry(idx));
}

//----------------------------------------------------------------------------------------
//! \fn Real angleToTime(const double f, const double ecc, const double a, const double per)
//! \brief angleToTime: Calculate orbital time from true anomaly.
//! \param f   The true anomaly.
//! \param ecc The orbital eccentricity.
//! \param a   The semi-major axis.
//! \param per The orbital period.
//! \return The orbital time.
Real angleToTime(const double f, const double ecc, const double a, const double per) {
  Real cos_u = (cos(f) + ecc) / (1.0 + ecc * cos(f)); // cosine eccentric anomaly
  Real sin_u = sqrt(1.0 - cos_u*cos_u);               // sine eccentric anomaly
  Real u = atan2(sin_u, cos_u);                       // eccentric anomaly
  Real sgn = static_cast<Real>(2 * (f > 0.0) - 1);
  Real num_per = floor(0.5 + f / (2.0 * M_PI));
  return sgn * (u - ecc * sin_u) * sqrt(a*a*a) + num_per * per;
}

void readAM(hid_t file, const char* field, AthenaArray<Real> &array, int &num) {
  hid_t dataset = H5Dopen(file, field, H5P_DEFAULT);
  hid_t dataspace = H5Dget_space(dataset);
  hsize_t dims[1];
  H5Sget_simple_extent_dims(dataspace, dims, NULL);
  num = static_cast<int>(dims[0]);
  array.NewAthenaArray(num);
  H5Dread(dataset, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, array.data());
  H5Sclose(dataspace);
  H5Dclose(dataset);
}

//----------------------------------------------------------------------------------------
//! \fn void tdeSrcFunc(...)
//! \brief tdeSrcFunc: Custom source function.
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
  EquationOfState *peos = pmb->peos;
  Real z, rho, vel, pres, egas, q, vdot, rhodot, mask, fac;
  
  // do interpolation
  int idx;
  Real iparam;
  calcIparam(time, tde->num_time, tde->time, idx, iparam);
  Real r = interp(idx, iparam, tde->r);
  Real area = interp(idx, iparam, tde->area);
  Real areadot = interp(idx, iparam, tde->areadot);

  // loop over cells
  for (int i=pmb->is; i<=pmb->ie; i++) {
    
    // get primitives
    z = pmb->pcoord->x1v(i);
    rho = prim(IDN, 0, 0, i);
    vel = prim(IVX, 0, 0, i);
    pres = prim(IPR, 0, 0, i);
    egas = peos->EgasFromRhoP(rho, pres);
    q = 1.0 + pres / egas; // d(lnegas)/d(lnrho)|s
    
    // compute time derivatives
    vdot = -z / (r*r*r);
    rhodot = -rho * areadot / area;
    mask = prim_scalar(0, 0, 0, i);
    fac = exp(-(1.0 - mask) / 0.02);

    // update conserved variables
    cons(IDN, 0, 0, i) += fac * dt * rhodot;
    cons(IM1, 0, 0, i) += fac * dt * (rho * vdot + vel * rhodot);
    cons(IEN, 0, 0, i) += fac * dt * (
      rho * vel * vdot 
      + 0.5 * vel*vel * rhodot
      + egas * rhodot / rho * q
    );
    cons_scalar(0, 0, 0, i) += fac * dt * rhodot * mask;
  }

}

//----------------------------------------------------------------------------------------
//! \fn Real calcEdotTide(MeshBlock *pmb, int iout)
//! \brief calcEdotTide: Calculate the energy source term contribution from tides.
Real calcEdotTide(MeshBlock *pmb, int iout) {
  
  // initialize variables
  Real z, dz, rho, vel, vdot, mask, fac;
  Real Edot = 0.0;
  
  // do interpolation
  int idx;
  Real iparam;
  Real time = pmb->pmy_mesh->time;
  calcIparam(time, tde->num_time, tde->time, idx, iparam);
  Real r = interp(idx, iparam, tde->r);

  // loop over cells
  for (int i=pmb->is; i<=pmb->ie; i++) {
    
    // get primitives
    z = pmb->pcoord->x1v(i);
    dz = pmb->pcoord->x1f(i+1) - pmb->pcoord->x1f(i);
    rho = pmb->phydro->w(IDN, 0, 0, i);
    vel = pmb->phydro->w(IVX, 0, 0, i);

    // compute Edot
    vdot = -z / (r*r*r);
    mask = pmb->pscalars->r(0, 0, 0, i);
    fac = exp(-(1.0 - mask) / 0.02);
    Edot += -fac * dz * rho * vel * vdot;
  }
  return Edot;
}

//----------------------------------------------------------------------------------------
//! \fn Real calcEdotArea(MeshBlock *pmb, int iout)
//! \brief calcEdotArea: Calculate the energy source term contribution from in-plane stretching.
Real calcEdotArea(MeshBlock *pmb, int iout) {
  
  // initialize variables
  Real z, dz, rho, vel, pres, egas, q, rhodot, mask, fac;
  Real Edot = 0.0;

  // do interpolation
  int idx;
  Real iparam;
  Real time = pmb->pmy_mesh->time;
  calcIparam(time, tde->num_time, tde->time, idx, iparam);
  Real area = interp(idx, iparam, tde->area);
  Real areadot = interp(idx, iparam, tde->areadot);

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
    rhodot = -rho * areadot / area;
    mask = pmb->pscalars->r(0, 0, 0, i);
    fac = exp(-(1.0 - mask) / 0.02);
    Edot += -fac * dz * rhodot * (0.5 * vel*vel + egas / rho * q);
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
//! \fn Real calcRhoC(MeshBlock *pmb, int iout)
//! \brief calcRhoC: Calculate the central density.
Real calcRhoC(MeshBlock *pmb, int iout) {
  return pmb->phydro->w(IDN, 0, 0, 0);
}

//----------------------------------------------------------------------------------------
//! \fn Real calcRhoC(MeshBlock *pmb, int iout)
//! \brief calcRhoC: Calculate the central pressure.
Real calcPresC(MeshBlock *pmb, int iout) {
  return pmb->phydro->w(IPR, 0, 0, 0);
}

//----------------------------------------------------------------------------------------
//! \fn Real calcH(MeshBlock *pmb, int iout)
//! \brief calcH: Calculate the dimensionless height.
Real calcH(MeshBlock *pmb, int iout) {
  int idx;
  Real iparam;
  Real time = pmb->pmy_mesh->time;
  calcIparam(time, tde->num_time, tde->time, idx, iparam);
  Real area = interp(idx, iparam, tde->area);
  Real rho_c = pmb->phydro->w(IDN, 0, 0, 0);
  return tde->rho_init / (rho_c * area);
}

//----------------------------------------------------------------------------------------
//! \fn Real calclogHdot(MeshBlock *pmb, int iout)
//! \brief calclogHdot: Calculate the dimensionless height derivative.
//! Maximum likelihood estimator for dvz/dz weighted by mass of each cell.
Real calclogHdot(MeshBlock *pmb, int iout) {
  Real dz, z, rho, vel;
  Real num = 0.0;
  Real den = 0.0;
  for (int i=pmb->is; i<=pmb->ie; i++) {
    dz = pmb->pcoord->x1f(i+1) - pmb->pcoord->x1f(i);
    z = pmb->pcoord->x1v(i);
    rho = pmb->phydro->w(IDN, 0, 0, i);
    vel = pmb->phydro->w(IVX, 0, 0, i);
    num += rho * dz * z * vel;
    den += rho * dz * z*z;
  }
  return num / den;
}

//----------------------------------------------------------------------------------------
//! \fn Real calcCoord(MeshBlock *pmb, int iout)
//! \brief calcCoord: Calculate the Eulerian coordinate of each mass coordinate.
template <int idx>
Real calcCoord(MeshBlock *pmb, int iout) {
  
  // initialize variables
  Real mfrac = tde->mcoord[idx];
  Real z_prev = 0.0;
  Real mass = 0.0;
  Real mass_prev = 0.0;
  Real rho, z, dz, iparam, mask, fac;

  // do interpolation
  int idx2;
  Real time = pmb->pmy_mesh->time;
  calcIparam(time, tde->num_time, tde->time, idx2, iparam);
  Real area = interp(idx2, iparam, tde->area);
  Real mtot = tde->mtot * tde->area0 / area;

  // find mass coordinate
  for (int i=pmb->is; i<=pmb->ie; i++) {
    rho = pmb->phydro->w(IDN, 0, 0, i);
    z = pmb->pcoord->x1v(i);
    dz = pmb->pcoord->x1f(i+1) - pmb->pcoord->x1f(i);
    mass += rho * dz;
    if (mass / mtot > mfrac) break;
    z_prev = z;
    mass_prev = mass;
  }

  // interpolate value
  iparam = (mfrac * mtot - mass_prev) / (mass - mass_prev);
  return z_prev * (1.0 - iparam) + z * iparam;
}

//----------------------------------------------------------------------------------------
//! \fn Real calcRho(MeshBlock *pmb, int iout)
//! \brief calcRho: Calculate the density of each mass coordinate.
template <int idx>
Real calcRho(MeshBlock *pmb, int iout) {
  
  // initialize variables
  Real mfrac = tde->mcoord[idx];
  Real rho_prev = 0.0;
  Real mass = 0.0;
  Real mass_prev = 0.0;
  Real rho, dz, iparam, mask, fac;

  // do interpolation
  int idx2;
  Real time = pmb->pmy_mesh->time;
  calcIparam(time, tde->num_time, tde->time, idx2, iparam);
  Real area = interp(idx2, iparam, tde->area);
  Real mtot = tde->mtot * tde->area0 / area;

  // find mass coordinate
  for (int i=pmb->is; i<=pmb->ie; i++) {
    rho = pmb->phydro->w(IDN, 0, 0, i);
    dz = pmb->pcoord->x1f(i+1) - pmb->pcoord->x1f(i);
    mass += rho * dz;
    if (mass / mtot > mfrac) break;
    rho_prev = rho;
    mass_prev = mass;
  }

  // interpolate value
  iparam = (mfrac * mtot - mass_prev) / (mass - mass_prev);
  return exp(log(rho_prev) * (1.0 - iparam) + log(rho) * iparam);
}

//----------------------------------------------------------------------------------------
//! \fn Real calcPres(MeshBlock *pmb, int iout)
//! \brief calcPres: Calculate the pressure of each mass coordinate.
template <int idx>
Real calcPres(MeshBlock *pmb, int iout) {
  
  // initialize variables
  Real mfrac = tde->mcoord[idx];
  Real pres_prev = 0.0;
  Real mass = 0.0;
  Real mass_prev = 0.0;
  Real rho, pres, dz, iparam, mask, fac;

  // do interpolation
  int idx2;
  Real time = pmb->pmy_mesh->time;
  calcIparam(time, tde->num_time, tde->time, idx2, iparam);
  Real area = interp(idx2, iparam, tde->area);
  Real mtot = tde->mtot * tde->area0 / area;

  // find mass coordinate
  for (int i=pmb->is; i<=pmb->ie; i++) {
    rho = pmb->phydro->w(IDN, 0, 0, i);
    pres = pmb->phydro->w(IPR, 0, 0, i);
    dz = pmb->pcoord->x1f(i+1) - pmb->pcoord->x1f(i);
    mass += rho * dz;
    if (mass / mtot > mfrac) break;
    pres_prev = pres;
    mass_prev = mass;
  }

  // interpolate value
  iparam = (mfrac * mtot - mass_prev) / (mass - mass_prev);
  return exp(log(pres_prev) * (1.0 - iparam) + log(pres) * iparam);
}

template <int idx>
void EnrollCalcCoord(Mesh *pmy_mesh, int num_out) {
  std::ostringstream label;
  label << "z" << idx;
  pmy_mesh->EnrollUserHistoryOutput(num_out-1 + idx, calcCoord<idx>, label.str().c_str(), UserHistoryOperation::max);
  EnrollCalcCoord<idx-1>(pmy_mesh, num_out);
}

template <>
void EnrollCalcCoord<0>(Mesh *pmy_mesh, int num_out) {}

template <int idx>
void EnrollCalcRho(Mesh *pmy_mesh, int num_out) {
  std::ostringstream label;
  label << "rho" << idx;
  pmy_mesh->EnrollUserHistoryOutput(num_out-1 + idx, calcRho<idx>, label.str().c_str(), UserHistoryOperation::max);
  EnrollCalcRho<idx-1>(pmy_mesh, num_out);
}

template <>
void EnrollCalcRho<0>(Mesh *pmy_mesh, int num_out) {}

template <int idx>
void EnrollCalcPres(Mesh *pmy_mesh, int num_out) {
  std::ostringstream label;
  label << "pres" << idx;
  pmy_mesh->EnrollUserHistoryOutput(num_out-1 + idx, calcPres<idx>, label.str().c_str(), UserHistoryOperation::max);
  EnrollCalcPres<idx-1>(pmy_mesh, num_out);
}

template <>
void EnrollCalcPres<0>(Mesh *pmy_mesh, int num_out) {}

void Mesh::InitUserMeshData(ParameterInput *pin) {

  constexpr int num_out = 8;
  AllocateUserHistoryOutput(num_out + 11 + 11 + 11);
  EnrollUserHistoryOutput(0, calcRhoC, "rho_c", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, calcPresC, "pres_c", UserHistoryOperation::max);
  EnrollUserHistoryOutput(2, calcEdotTide, "Edot_tide", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(3, calcEdotArea, "Edot_area", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(4, calcEkin, "Ekin", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(5, calcEth, "Eth", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(6, calcH, "H", UserHistoryOperation::max);
  EnrollUserHistoryOutput(7, calclogHdot, "logHdot", UserHistoryOperation::max);
  EnrollCalcCoord<11>(this, num_out);
  EnrollCalcRho<11>(this, num_out + 11);
  EnrollCalcPres<11>(this, num_out + 11 + 11);
  EnrollUserExplicitSourceFunction(tdeSrcFunc);

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Problem Generator for tidal disruption event problems
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  std::stringstream msg;
  
  // problem name
  std::string pname = pin->GetString("job", "problem_id");

  // affine model solution filename
  std::string am_name = pin->GetString("problem", "am");
  
  // floors
  tde->dfloor = pin->GetReal("hydro", "dfloor");
  tde->pfloor = pin->GetReal("hydro", "pfloor");

  // retrieve other parameters
  Real rho_init = pin->GetReal("problem", "rho_init");
  Real rho0 = pin->GetReal("problem", "rho0");
  Real s_eq = pin->GetReal("problem", "s_eq");
  Real H0 = pin->GetReal("problem", "H0");
  Real Hdot0 = pin->GetReal("problem", "Hdot0");
  Real K0 = pin->GetReal("problem", "K0");
  Real gamg = pin->GetOrAddReal("hydro", "gamma", 5.0/3.0);
  Real z0 = s_eq * H0;
  tde->rho_init = rho_init;

  tde->mcoord = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99};

  // create arrays for stream slice profile
  int num_le = 16384;
  AthenaArray<Real> r_sl, rho_sl;
  r_sl.NewAthenaArray(num_le);
  rho_sl.NewAthenaArray(num_le);

  // define polytrope parameters
  tde->n = 1.5;
  Real gam = 1.0 + 1.0 / tde->n;
  const Real xi_max = 2.6477; // from cylindrical polytrope
  Real alpha = z0 / xi_max;

  // compute stream slice profile
  laneEmden le;
  Real dxi, xi;
  Real cutoff = 1e-7;
  Real z_cutoff;
  dxi = xi_max / static_cast<Real>(num_le - 1);
  r_sl(0) = 0.0;
  rho_sl(0) = rho0;
  le.th = 1.0 - 0.25 * dxi*dxi; // from Taylor expansion around xi=0
  le.phi = -0.5 * dxi*dxi*dxi;
  for ( int i=1; i<num_le; i++ ) {
    xi = dxi + static_cast<Real>(i) * dxi;
    r_sl(i) = alpha * xi;
    rho_sl(i) = rho0 * std::pow(std::fmax(le.th, 0.0), tde->n);
    if ( rho_sl(i) < cutoff * rho0 ) z_cutoff = r_sl(i);
    rk4<decltype(calcLaneEmden), laneEmden>(calcLaneEmden, dxi, xi, le);
  }

  // read affine model solution
  hid_t file = H5Fopen(am_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  readAM(file, "time", tde->time, tde->num_time);
  readAM(file, "r", tde->r, tde->num_time);
  readAM(file, "area", tde->area, tde->num_time);
  readAM(file, "areadot", tde->areadot, tde->num_time);
  H5Fclose(file);

  int idx2;
  Real iparam2;
  Real time0 = pin->GetReal("time", "start_time");
  calcIparam(time0, tde->num_time, tde->time, idx2, iparam2);
  tde->area0 = interp(idx2, iparam2, tde->area);

  int idx;
  Real iparam, z, dz, exp_floor;
  Real rho, pres, vel, egas, mask;
  Real dz_floor = 0.02 * s_eq;
  tde->mtot = 0.0;

  for (int i=is; i<=ie; i++) {
    
    // compute the star density and pressure
    z = pcoord->x1v(i);
    dz = pcoord->x1f(i+1) - pcoord->x1f(i);
    calcIparam(z, num_le, r_sl, idx, iparam);
    rho = std::fmax(interp(idx, iparam, rho_sl), tde->dfloor);
    pres = std::fmax(K0 * std::pow(rho, gam), tde->pfloor);
    egas = peos->EgasFromRhoP(rho, pres);
    vel = Hdot0 / H0 * z;
    mask = 1.0;

    if ( rho < cutoff * rho_sl(0) && z > z_cutoff ) {
      exp_floor = exp(-(z - z_cutoff) / dz_floor);
      vel = Hdot0 / H0 * z * exp_floor;
      mask = exp_floor;
    } else {
      tde->mtot += rho * dz;
    }

    // set the initial conservative variables
    phydro->u(IDN, 0, 0, i) = rho;
    phydro->u(IM1, 0, 0, i) = rho * vel;
    phydro->u(IM2, 0, 0, i) = 0.0;
    phydro->u(IM3, 0, 0, i) = 0.0;
    phydro->u(IEN, 0, 0, i) = 0.5 * rho * vel*vel + egas;

    // set the initial passive scalars
    pscalars->s(0, 0, 0, i) = rho * mask; // floor mask
  }

  return;
}
