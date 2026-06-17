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
#include <cfloat>

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
#include "../utils/utils.hpp"

struct pgen_globals {
  Real at;
  Real gam;
  Real rho_b;
  Real pres_b;
  Real vel;
  Real boxlen;
  Real boxwid;
};

pgen_globals* gl = new pgen_globals();

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

//----------------------------------------------------------------------------------------
//! \fn void calcLaneEmden(const Real xi, const laneEmden le, laneEmden &dledxi)
//! \brief calcLaneEmden: Compute derivatives in the cylindrical Lane-Emden equation with respect to xi.
//! \param xi      Dimensionless radius
//! \param le      Lane-Emden parameters
//! \param dledxi  Lane-Emden parameter derivatives
void calcLaneEmden(const Real xi, const laneEmden le, laneEmden &dledxi) {
  dledxi.th = -le.phi / xi;
  dledxi.phi = std::pow(fmax(le.th, 0.0), 1.5) * xi;
};

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
};

Real interp(const Real x, const int size, const Real xmin, const Real xmax, const Real dx, const AthenaArray<Real> &arry) {
  if ( x <= xmin ) {
    // clamp values below array range
    return arry(0);
  } else if ( x >= xmax) {
    // clamp values above array range
    return arry(size - 1);
  } else {
    // interpolate
    Real u = (x - xmin) / dx;
    int idx = static_cast<int>(floor(u));
    Real iparam = u - idx;
    return arry(idx) + iparam * (arry(idx + 1) - arry(idx));
  }
};

void tide(
  MeshBlock *pmb, 
  const Real time, 
  const Real dt,
  const AthenaArray<Real> &prim, 
  const AthenaArray<Real> &prim_scalar,
  const AthenaArray<Real> &bcc, 
  AthenaArray<Real> &cons,
  AthenaArray<Real> &cons_scalar
) {

  Real rho;
  Real y, z, s, sclip;
  Real vy, vz;
  Real pxdot, pydot, pzdot;
  Real func;
  Real boxwid = gl->boxwid;
  
  for (int k=pmb->ks; k<=pmb->ke; k++) {
    for (int j=pmb->js; j<=pmb->je; j++) {
      for (int i=pmb->is; i<=pmb->ie; i++) {

        y = pmb->pcoord->x2v(j);
        z = pmb->pcoord->x3v(k);
        s = sqrt(y*y + z*z);
        vy = prim(IVY, k, j, i);
        vz = prim(IVZ, k, j, i);
        rho = prim(IDN, k, j, i);
        
        // momentum change
        if (s > 0.0) {
          
          pxdot = -rho * 

          pydot = -rho * gl->at * func * y/s * dt;
          pzdot = -rho * gl->at * func * z/s * dt;
        } else {
          pydot = 0.0;
          pzdot = 0.0;
        }
        
        // momentum and energy source terms
        cons(IM2, k, j, i) += pydot;
        cons(IM3, k, j, i) += pzdot;
        cons(IEN, k, j, i) += pydot * vy + pzdot * vz + (pydot*pydot + pzdot*pzdot) / (2.0 * rho);

      }
    }
  }

};

void calcCharBC(
  Real sign,
  Real rho_i, Real v1_i, Real v2_i, Real v3_i, Real pres_i,
  Real rho_b, Real v1_b, Real v2_b, Real v3_b, Real pres_b,
  Real& rho, Real& v1, Real& v2, Real& v3, Real& pres
) {
  
  // compute sound speed
  Real gam = gl->gam;
  Real cs_i = sqrt(gam * pres_i / rho_i);
  Real cs_b = sqrt(gam * pres_b / rho_b);
  
  // Riemann invariants
  Real Wp_i = v1_i + 2.0 * cs_i / (gam - 1.0);
  Real Wm_i = v1_i - 2.0 * cs_i / (gam - 1.0);
  Real Wp_b = v1_b + 2.0 * cs_b / (gam - 1.0);
  Real Wm_b = v1_b - 2.0 * cs_b / (gam - 1.0);

  // entropy invariants
  Real s_i = pres_i / pow(rho_i, gam);
  Real s_b = pres_b / pow(rho_b, gam);

  // acoustic waves
  Real Wp = sign * (v1_i + cs_i) > 0.0 ? Wp_i : Wp_b;
  Real Wm = sign * (v1_i - cs_i) > 0.0 ? Wm_i : Wm_b;
  Real s = sign * v1_i > 0.0 ? s_i : s_b;

  // reconstruct vz and cs
  v1 = 0.5 * (Wp + Wm);
  Real cs = 0.25 * (gam - 1.0) * (Wp - Wm);
  rho = rho_b;
  pres = rho * cs*cs / gam;

  // shear waves
  v2 = sign * v1_i > 0.0 ? v2_i : v2_b;
  v3 = sign * v1_i > 0.0 ? v3_i : v3_b;
}

void calcBgd(
  Real s, 
  Real& rho_b, Real& vx_b, Real& vy_b, Real& vz_b, Real& pres_b
) {
  rho_b  = gl->rho_b;
  vx_b   = gl->vel;
  vy_b   = 0.0;
  vz_b   = 0.0;
  pres_b = gl->pres_b;
}

void bndCharZin(
  MeshBlock *pmb, 
  Coordinates *pco, 
  AthenaArray<Real> &prim,
  FaceField &b, 
  Real time, 
  Real dt, 
  int il, int iu, int jl, 
  int ju, int kl, int ku, 
  int ngh
){
  Real y, z, s;
  Real rho_i, vx_i, vy_i, vz_i, pres_i;
  Real rho_b, vx_b, vy_b, vz_b, pres_b;
  Real rho, vx, vy, vz, pres;

  for (int k = kl-ngh; k <= kl-1; k++) {
    for (int j = jl; j <= ju; j++) {
      for (int i = il; i <= iu; i++) {

        // get interior state
        rho_i  = prim(IDN, kl, j, i);
        vx_i   = prim(IVX, kl, j, i);
        vy_i   = prim(IVY, kl, j, i);
        vz_i   = prim(IVZ, kl, j, i);
        pres_i = prim(IPR, kl, j, i);

        // background state
        y =  pmb->pcoord->x2f(j);
        z =  pmb->pcoord->x3f(k);
        s = sqrt(y*y + z*z);
        calcBgd(s, rho_b, vx_b, vy_b, vz_b, pres_b);
        
        // characteristic BC
        calcCharBC(
          -1.0, 
          rho_i, vz_i, vx_i, vy_i, pres_i,
          rho_b, vz_b, vx_b, vy_b, pres_b,
          rho, vz, vx, vy, pres
        );

        // fill ghost cell
        prim(IDN, k, j, i) = rho;
        prim(IVX, k, j, i) = vx;
        prim(IVY, k, j, i) = vy;
        prim(IVZ, k, j, i) = vz;
        prim(IPR, k, j, i) = pres;
      }
    }
  }
}

void bndCharZout(
  MeshBlock *pmb, 
  Coordinates *pco, 
  AthenaArray<Real> &prim,
  FaceField &b, 
  Real time, 
  Real dt, 
  int il, int iu, int jl, 
  int ju, int kl, int ku, 
  int ngh
){
  Real y, z, s;
  Real rho_i, vx_i, vy_i, vz_i, pres_i;
  Real rho_b, vx_b, vy_b, vz_b, pres_b;
  Real rho, vx, vy, vz, pres;

  for (int k=ku+1; k<=ku+ngh; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {

        // get interior state
        rho_i  = prim(IDN, ku, j, i);
        vx_i   = prim(IVX, ku, j, i);
        vy_i   = prim(IVY, ku, j, i);
        vz_i   = prim(IVZ, ku, j, i);
        pres_i = prim(IPR, ku, j, i);

        // background state
        y =  pmb->pcoord->x2f(j);
        z =  pmb->pcoord->x3f(k);
        s = sqrt(y*y + z*z);
        calcBgd(s, rho_b, vx_b, vy_b, vz_b, pres_b);
        
        // characteristic BC
        calcCharBC(
          +1.0, 
          rho_i, vz_i, vx_i, vy_i, pres_i,
          rho_b, vz_b, vx_b, vy_b, pres_b,
          rho, vz, vx, vy, pres
        );

        // fill ghost cell
        prim(IDN, k, j, i) = rho;
        prim(IVX, k, j, i) = vx;
        prim(IVY, k, j, i) = vy;
        prim(IVZ, k, j, i) = vz;
        prim(IPR, k, j, i) = pres;
      }
    }
  }

}

void bndCharYin(
  MeshBlock *pmb, 
  Coordinates *pco, 
  AthenaArray<Real> &prim,
  FaceField &b, 
  Real time, 
  Real dt, 
  int il, int iu, int jl, 
  int ju, int kl, int ku, 
  int ngh
){
  Real y, z, s;
  Real rho_i, vx_i, vy_i, vz_i, pres_i;
  Real rho_b, vx_b, vy_b, vz_b, pres_b;
  Real rho, vx, vy, vz, pres;

  for (int k=kl; k<=ku; k++) {
    for (int j=jl-ngh; j<=jl-1; j++) {
      for (int i=il; i<=iu; i++) {

        // get interior state
        rho_i = prim(IDN, k, jl, i);
        vx_i = prim(IVX, k, jl, i);
        vy_i = prim(IVY, k, jl, i);
        vz_i = prim(IVZ, k, jl, i);
        pres_i = prim(IPR, k, jl, i);

        // background state
        y =  pmb->pcoord->x2f(j);
        z =  pmb->pcoord->x3f(k);
        s = sqrt(y*y + z*z);
        calcBgd(s, rho_b, vx_b, vy_b, vz_b, pres_b);
        
        // characteristic BC
        calcCharBC(
          -1.0, 
          rho_i, vy_i, vz_i, vx_i, pres_i,
          rho_b, vy_b, vz_b, vx_b, pres_b,
          rho, vy, vz, vx, pres
        );

        // fill ghost cell
        prim(IDN, k, j, i) = rho;
        prim(IVX, k, j, i) = vx;
        prim(IVY, k, j, i) = vy;
        prim(IVZ, k, j, i) = vz;
        prim(IPR, k, j, i) = pres;
      }
    }
  }

}

void bndCharYout(
  MeshBlock *pmb, 
  Coordinates *pco, 
  AthenaArray<Real> &prim,
  FaceField &b, 
  Real time, 
  Real dt, 
  int il, int iu, int jl, 
  int ju, int kl, int ku, 
  int ngh
){
  Real y, z, s;
  Real rho_i, vx_i, vy_i, vz_i, pres_i;
  Real rho_b, vx_b, vy_b, vz_b, pres_b;
  Real rho, vx, vy, vz, pres;

  for (int k=kl; k<=ku; k++) {
    for (int j=ju+1; j<=ju+ngh; j++) {
      for (int i=il; i<=iu; i++) {

        // set interior state
        rho_i = prim(IDN, k, ju, i);
        vx_i = prim(IVX, k, ju, i);
        vy_i = prim(IVY, k, ju, i);
        vz_i = prim(IVZ, k, ju, i);
        pres_i = prim(IPR, k, ju, i);

        // background state
        y =  pmb->pcoord->x2f(j);
        z =  pmb->pcoord->x3f(k);
        s = sqrt(y*y + z*z);
        calcBgd(s, rho_b, vx_b, vy_b, vz_b, pres_b);
        
        // characteristic BC
        calcCharBC(
          +1.0, 
          rho_i, vy_i, vz_i, vx_i, pres_i,
          rho_b, vy_b, vz_b, vx_b, pres_b,
          rho, vy, vz, vx, pres
        );

        // fill ghost cell
        prim(IDN, k, j, i) = rho;
        prim(IVX, k, j, i) = vx;
        prim(IVY, k, j, i) = vy;
        prim(IVZ, k, j, i) = vz;
        prim(IPR, k, j, i) = pres;
      }
    }
  }

}

void bndDiodeZin(
  MeshBlock *pmb, 
  Coordinates *pco, 
  AthenaArray<Real> &prim,
  FaceField &b, 
  Real time, 
  Real dt, 
  int il, int iu, int jl, 
  int ju, int kl, int ku, 
  int ngh
){

  Real z = pmb->pcoord->x3f(kl);
  Real dz = pmb->pcoord->dx3f(kl);
  
  for (int k=kl-ngh; k<=kl-1; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {

        // set ghost zone primatives: diode
        prim(IDN, k, j, i) = prim(IDN, kl, j, i);
        prim(IVX, k, j, i) = prim(IVX, kl, j, i);
        prim(IVY, k, j, i) = prim(IVY, kl, j, i);
        prim(IVZ, k, j, i) = std::min(0.0, prim(IVZ, kl, j, i));
        prim(IPR, k, j, i) = prim(IPR, kl, j, i) + prim(IDN, kl, j, i) * gl->at * z * dz;

      }
    }
  }

}

void bndDiodeZout(
  MeshBlock *pmb, 
  Coordinates *pco, 
  AthenaArray<Real> &prim,
  FaceField &b, 
  Real time, 
  Real dt, 
  int il, int iu, int jl, 
  int ju, int kl, int ku, 
  int ngh
){

  Real z = pmb->pcoord->x3f(ku+1);
  Real dz = pmb->pcoord->dx3f(ku+1);

  for (int k=ku+1; k<=ku+ngh; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {

        // set ghost zone primatives: diode
        prim(IDN, k, j, i) = prim(IDN, ku, j, i);
        prim(IVX, k, j, i) = prim(IVX, ku, j, i);
        prim(IVY, k, j, i) = prim(IVY, ku, j, i);
        prim(IVZ, k, j, i) = std::max(0.0, prim(IVZ, ku, j, i));
        prim(IPR, k, j, i) = prim(IPR, ku, j, i) + prim(IDN, ku, j, i) * gl->at * z * dz;

      }
    }
  }

}

void bndDiodeYin(
  MeshBlock *pmb, 
  Coordinates *pco, 
  AthenaArray<Real> &prim,
  FaceField &b, 
  Real time, 
  Real dt, 
  int il, int iu, int jl, 
  int ju, int kl, int ku, 
  int ngh
){

  Real y = pmb->pcoord->x2f(jl);
  Real dy = pmb->pcoord->dx2f(jl);

  for (int k=kl; k<=ku; k++) {
    for (int j=jl-ngh; j<=jl-1; j++) {
      for (int i=il; i<=iu; i++) {

        // set ghost zone primatives: diode
        prim(IDN, k, j, i) = prim(IDN, k, jl, i);
        prim(IVX, k, j, i) = prim(IVX, k, jl, i);
        prim(IVY, k, j, i) = std::min(0.0, prim(IVY, k, jl, i));
        prim(IVZ, k, j, i) = prim(IVZ, k, jl, i);
        prim(IPR, k, j, i) = prim(IPR, k, jl, i) + prim(IDN, k, jl, i) * gl->at * y * dy;

      }
    }
  }

}

void bndDiodeYout(
  MeshBlock *pmb, 
  Coordinates *pco, 
  AthenaArray<Real> &prim,
  FaceField &b, 
  Real time, 
  Real dt, 
  int il, int iu, int jl, 
  int ju, int kl, int ku, 
  int ngh
){

  Real y = pmb->pcoord->x2f(ju+1);
  Real dy = pmb->pcoord->dx2f(ju+1);

  for (int k=kl; k<=ku; k++) {
    for (int j=ju+1; j<=ju+ngh; j++) {
      for (int i=il; i<=iu; i++) {

        // set ghost zone primatives: diode
        prim(IDN, k, j, i) = prim(IDN, k, ju, i);
        prim(IVX, k, j, i) = prim(IVX, k, ju, i);
        prim(IVY, k, j, i) = std::max(0.0, prim(IVY, k, ju, i));
        prim(IVZ, k, j, i) = prim(IVZ, k, ju, i);
        prim(IPR, k, j, i) = prim(IPR, k, ju, i) + prim(IDN, k, ju, i) * gl->at * y * dy;

      }
    }
  }

}

Real mix(MeshBlock *pmb, int iout) {
  
  Real dx, dy, dz, dV;
  Real rho, mask;
  Real mix = 0.0;
  
  for(int k=pmb->ks; k<=pmb->ke; k++) {
    for(int j=pmb->js; j<=pmb->je; j++) {
      for(int i=pmb->is; i<=pmb->ie; i++) {

        dx = pmb->pcoord->dx1v(i);
        dy = pmb->pcoord->dx2v(j);
        dz = pmb->pcoord->dx3v(k);
        dV = dx * dy * dz;
        rho = pmb->phydro->w(IDN, k, j, i);
        mask = pmb->pscalars->r(0, k, j, i);
        if ((mask > 0.0) && (mask < 1.0)) {
          mix += -(mask * log(mask) + (1.0 - mask) * log(1.0 - mask)) * rho * dV;
        }
      }
    }
  }
  return mix;
}

Real mom_s(MeshBlock *pmb, int iout) {
  
  Real dx, dy, dz, dV;
  Real rho, vx, mask;
  Real mom_s = 0.0;
  
  for(int k=pmb->ks; k<=pmb->ke; k++) {
    for(int j=pmb->js; j<=pmb->je; j++) {
      for(int i=pmb->is; i<=pmb->ie; i++) {

        dx = pmb->pcoord->dx1v(i);
        dy = pmb->pcoord->dx2v(j);
        dz = pmb->pcoord->dx3v(k);
        dV = dx * dy * dz;
        rho = pmb->phydro->w(IDN, k, j, i);
        vx = pmb->phydro->w(IVX, k, j, i);
        mask = pmb->pscalars->r(0, k, j, i);
        mom_s += mask * vx * rho * dV;
      }
    }
  }
  return mom_s;
}

Real mass1(MeshBlock *pmb, int iout) {
  
  Real dx, dy, dz, dV;
  Real rho, mix, mask;
  Real mass1 = 0.0;
  
  for(int k=pmb->ks; k<=pmb->ke; k++) {
    for(int j=pmb->js; j<=pmb->je; j++) {
      for(int i=pmb->is; i<=pmb->ie; i++) {

        dx = pmb->pcoord->dx1v(i);
        dy = pmb->pcoord->dx2v(j);
        dz = pmb->pcoord->dx3v(k);
        dV = dx * dy * dz;
        rho = pmb->phydro->w(IDN, k, j, i);
        mask = pmb->pscalars->r(0, k, j, i);
        if ((mask > 0.0) && (mask < 1.0)) {
          mix = -(mask * log(mask) + (1.0 - mask) * log(1.0 - mask));
          if (mix > 0.1 * log(2.0)) {
            mass1 += rho * dV;
          }
        }
      }
    }
  }
  return mass1;
}

Real mass5(MeshBlock *pmb, int iout) {
  
  Real dx, dy, dz, dV;
  Real rho, mix, mask;
  Real mass5 = 0.0;
  
  for(int k=pmb->ks; k<=pmb->ke; k++) {
    for(int j=pmb->js; j<=pmb->je; j++) {
      for(int i=pmb->is; i<=pmb->ie; i++) {

        dx = pmb->pcoord->dx1v(i);
        dy = pmb->pcoord->dx2v(j);
        dz = pmb->pcoord->dx3v(k);
        dV = dx * dy * dz;
        rho = pmb->phydro->w(IDN, k, j, i);
        mask = pmb->pscalars->r(0, k, j, i);
        if ((mask > 0.0) && (mask < 1.0)) {
          mix = -(mask * log(mask) + (1.0 - mask) * log(1.0 - mask));
          if (mix > 0.5 * log(2.0)) {
            mass5 += rho * dV;
          }
        }
      }
    }
  }
  return mass5;
}

Real mass9(MeshBlock *pmb, int iout) {
  
  Real dx, dy, dz, dV;
  Real rho, mix, mask;
  Real mass9 = 0.0;
  
  for(int k=pmb->ks; k<=pmb->ke; k++) {
    for(int j=pmb->js; j<=pmb->je; j++) {
      for(int i=pmb->is; i<=pmb->ie; i++) {

        dx = pmb->pcoord->dx1v(i);
        dy = pmb->pcoord->dx2v(j);
        dz = pmb->pcoord->dx3v(k);
        dV = dx * dy * dz;
        rho = pmb->phydro->w(IDN, k, j, i);
        mask = pmb->pscalars->r(0, k, j, i);
        if ((mask > 0.0) && (mask < 1.0)) {
          mix = -(mask * log(mask) + (1.0 - mask) * log(1.0 - mask));
          if (mix > 0.9 * log(2.0)) {
            mass9 += rho * dV;
          }
        }
      }
    }
  }
  return mass9;
}

Real iner(MeshBlock *pmb, int iout) {
  
  Real y, z, ssq;
  Real dx, dy, dz, dV;
  Real rho, mix, mask;
  Real iner = 0.0;
  
  for(int k=pmb->ks; k<=pmb->ke; k++) {
    for(int j=pmb->js; j<=pmb->je; j++) {
      for(int i=pmb->is; i<=pmb->ie; i++) {

        y = pmb->pcoord->x2v(j);
        z = pmb->pcoord->x3v(k);
        ssq = y*y + z*z;
        dx = pmb->pcoord->dx1v(i);
        dy = pmb->pcoord->dx2v(j);
        dz = pmb->pcoord->dx3v(k);
        dV = dx * dy * dz;
        rho = pmb->phydro->w(IDN, k, j, i);
        mask = pmb->pscalars->r(0, k, j, i);
        iner += mask * ssq * rho * dV;
      }
    }
  }
  return iner;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {

  //EnrollUserBoundaryFunction(BoundaryFace::inner_x3, bndCharZin);
  //EnrollUserBoundaryFunction(BoundaryFace::outer_x3, bndCharZout);
  //EnrollUserBoundaryFunction(BoundaryFace::inner_x2, bndCharYin);
  //EnrollUserBoundaryFunction(BoundaryFace::outer_x2, bndCharYout);
  
  AllocateUserHistoryOutput(6);
  EnrollUserHistoryOutput(0, mix, "mix", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(1, mom_s, "mom_s", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(2, mass1, "mass1", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(3, mass5, "mass5", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(4, mass9, "mass9", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(5, iner, "iner", UserHistoryOperation::sum);
  EnrollUserExplicitSourceFunction(tide);
  return;

};

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Problem Generator for the TDE self-intersection problem
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  std::stringstream msg;
  
  // polytrope properties
  const Real xi_max  = 2.6477;
  const Real phi_max = 1.0611;
  Real mass_rhoc_max = 2.0 * M_PI * phi_max/(xi_max*xi_max);
  
  // dimensionless parameters
  Real gam    = pin->GetReal("hydro", "gamma");    // adiabatic index
  Real dlt    = pin->GetReal("problem", "dlt");    // stream-to-background density ratio
  Real mach_s = pin->GetOrAddReal("problem", "mach_s", -1.0); // mach number relative to stream
  Real mach_b = pin->GetReal("problem", "mach_b"); // mach number relative to background
  Real tide   = pin->GetReal("problem", "tide");   // tidal acceleration relative to ram pressure acceleration
  Real amp    = pin->GetReal("problem", "amp");    // velocity perturbation amplitude
  Real sig1   = pin->GetReal("problem", "sig1");  // shear layer cylindrical width
  Real sig2   = pin->GetReal("problem", "sig2");  // velocity perturbation cylindrical width
  Real psi    = pin->GetReal("problem", "psi") * M_PI/180.0;   // background flow angle
  int kmin   = pin->GetOrAddInteger("problem", "kmin", 2);     // minimum longitudinal wavenumber
  int kmax   = pin->GetOrAddInteger("problem", "kmax", 64);     // maximum longitudinal wavenumber
  int mmin   = pin->GetOrAddInteger("problem", "mmin", 0);     // minimum azimuthal wavenunber
  int mmax   = pin->GetOrAddInteger("problem", "mmax", 4);     // maximum azimuthal wavenumber

  // box length
  Real xmin = pin->GetReal("mesh", "x1min");
  Real xmax = pin->GetReal("mesh", "x1max");
  Real ymin = pin->GetReal("mesh", "x2min");
  Real ymax = pin->GetReal("mesh", "x2max");
  Real boxlen = xmax - xmin;
  Real boxwid = ymax - ymin;
  gl->boxlen = boxlen;
  gl->boxwid = boxwid;

  // settings
  std::string profile = pin->GetString("problem", "profile");
  std::string seeding = pin->GetString("problem", "seeding");

  // random number generator seed
  std::int64_t iseed;
  if (seeding == "mode") {
    iseed = -6435624;
  } else if (seeding == "white") {
    iseed = -1 - gid;
  } else {
    msg << "Seeding option " << seeding << " not supported" << std::endl;
    msg << "Options: mode, white" << std::endl;
    ATHENA_ERROR(msg);
  }

  // we use units were rho_b = pres_b = 1.0
  if (mach_s == -1.0) { mach_s = mach_b * sqrt(dlt); }
  Real cs_b   = sqrt(gam);
  Real vel    = mach_b * cs_b;
  Real cs_s   = vel / mach_s;
  Real rho_s  = dlt;
  Real pres_s = rho_s * cs_s*cs_s / gam;
  Real at     = tide * vel*vel / dlt;
  Real K_s    = pres_s / pow(rho_s, 5.0/3.0);

  // set globals
  gl->at = at;
  gl->vel = vel;
  gl->rho_b = 1.0;
  gl->pres_b = 1.0;

  // print info
  if (Globals::my_rank == 0 && gid == 0) {
    std::cout << "vel=" << vel << " rho_s=" << rho_s << " pres_s=" << pres_s << " at=" << at << std::endl;
    std::cout << "nk=" << kmax-kmin+1 << " nm=" << mmax-mmin+1 << std::endl;
  }

  // initialzie polytrope
  Real xi;
  laneEmden le;
  AthenaArray<Real> rho_rhoc, mass_rhoc;
  const int num_le = 16384;
  Real dxi = xi_max / static_cast<Real>(num_le - 1);
  Real ds = 1.0 / static_cast<Real>(num_le - 1);
  le.th = 1.0 - 0.25 * dxi*dxi;
  le.phi = 0.5 * dxi*dxi*dxi;
  rho_rhoc.NewAthenaArray(num_le);
  mass_rhoc.NewAthenaArray(num_le);
  rho_rhoc(0) = 1.0;
  mass_rhoc(0) = 0.0;

  // integrate Lane-Emden equation
  for ( int i=1; i<num_le; i++ ) {
    xi = dxi + static_cast<Real>(i) * dxi;
    rho_rhoc(i) = std::pow(std::fmax(le.th, 0.0), 1.5);
    mass_rhoc(i) = 2.0 * M_PI * le.phi/(xi_max*xi_max);
    rk4<decltype(calcLaneEmden), laneEmden>(calcLaneEmden, dxi, xi, le);
  }
  
  Real x, y, z, s, phi;
  Real rho, pres, cs, mask;
  Real vx, vy, vz;
  Real sfunc_tanh, sfunc_exp, vpert;
  Real kk, mm;
  Real phase, norm;
  Real rho_sp, rho_bp, pres_sp, pres_bp;

  // compute random phases
  AthenaArray<Real> random_phase;
  random_phase.NewAthenaArray(kmax-kmin+1, mmax-mmin+1);
  for (kk = kmin; kk <= kmax; kk++) {
    for (mm = mmin; mm <= mmax; mm++) {
      random_phase(kk-kmin, mm-mmin) = 2.0 * M_PI * ran2(&iseed);
    }
  }

  // tidal flattening function (for later)
  // Real sclip, func;
  // sclip = std::min(s, 1.0);
  // func = sin(2.0*M_PI*sclip/boxwid) * cos(M_PI*sclip/boxwid)*cos(M_PI*sclip/boxwid) * boxwid/(2.0*M_PI);

  
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        
        x = pcoord->x1v(i);
        y = pcoord->x2v(j);
        z = pcoord->x3v(k);
        s = sqrt(y*y + z*z);
        phi = atan2(z, y);

        sfunc_tanh = 0.5 * (1.0 - tanh((s - 1.0) / sig1));
        sfunc_exp = exp(-(s - 1.0)*(s - 1.0) / (2.0 * sig2*sig2));
        
        if (profile == "tophat") {
          rho_bp = 1.0;
          pres_bp = 1.0;
          rho_sp = rho_s;
          pres_sp = pres_s;
        } else if (profile == "polytrope") {
          rho_bp  = 1.0;
          pres_bp = 1.0;
          rho_sp  = rho_s * interp(s, num_le, 0.0, 1.0, ds, rho_rhoc);
          pres_sp = K_s * pow(rho_sp, 5.0/3.0);
        } else {
          msg << "Profile option " << profile << " not supported" << std::endl;
          msg << "Options: tophat, polytrope" << std::endl;
          ATHENA_ERROR(msg);
        }

        rho  = rho_bp + (rho_sp - rho_bp) * sfunc_tanh;
        pres = pres_bp + (pres_sp - pres_bp) * sfunc_tanh;
        vx   = -vel * (1.0 - sfunc_tanh) * cos(psi);
        vy   = vel * (1.0 - sfunc_tanh) * sin(psi);
        vz   = 0.0;
        mask = sfunc_tanh;
        cs = sqrt(gam * pres / rho);

        // velocity perturbation
        if (seeding == "mode") {
          vpert = 0.0;
          for (kk = kmin; kk <= kmax; kk++) {
            for (mm = mmin; mm <= mmax; mm++) {
              phase = 2.0*M_PI * static_cast<Real>(kk) * x/gl->boxlen + static_cast<Real>(mm) * phi + random_phase(kk-kmin, mm-mmin);
              vpert += cos(phase);
            }
          }
          norm = sqrt(static_cast<Real>(kmax-kmin+1) * static_cast<Real>(mmax-mmin+1));
          vy += amp/norm * vel * sfunc_exp * vpert * cos(phi);
          vz += amp/norm * vel * sfunc_exp * vpert * sin(phi);
        } else if (seeding == "white") {
          vx += amp * cs * sfunc_tanh * (ran2(&iseed) - 0.5);
          vy += amp * cs * sfunc_tanh * (ran2(&iseed) - 0.5);
          vz += amp * cs * sfunc_tanh * (ran2(&iseed) - 0.5);
        }
        
        // set the initial conservative variables
        phydro->u(IDN, k, j, i) = rho;
        phydro->u(IM1, k, j, i) = vx * rho;
        phydro->u(IM2, k, j, i) = vy * rho;
        phydro->u(IM3, k, j, i) = vz * rho;
        phydro->u(IEN, k, j, i) = 0.5 * rho * (vx*vx + vy*vy + vz*vz) + pres / (gam - 1.0);
        pscalars->s(0, k, j, i) = mask * rho;

      }
    }
  }

  return;
}
