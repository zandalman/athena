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

struct pgen_poly {
  int size;
  Real dfrac;
  AthenaArray<Real> rho_rhoc;
  Real dt_max;
  Real delay;
  Real pres_inj;
};

struct pgen_stream {
  Real rho;
  Real K;
  Real y;
  Real xdot;
  Real ydot;
  Real yp_max;
  Real zp_max;
  Real dvx_dy;
  Real dvy_dy;
  Real dvz_dz;
};

pgen_poly* poly = new pgen_poly();
pgen_stream* s1 = new pgen_stream();
pgen_stream* s2 = new pgen_stream();

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

void TR13Accel(Real x, Real y, Real z, Real xdot, Real ydot, Real zdot, Real& xddot, Real& yddot, Real &zddot) {
  
  Real r, rm2, rv;
  Real rxvx, rxvy, rxvz, rxvsq;
  Real rinv5, fac1, fac2;
  
  r = sqrt(x*x + y*y + z*z);
  rm2 = r - 2.0;
  rv = x * xdot + y * ydot + z * zdot;
  rxvx = y * zdot - z * ydot;
  rxvy = z * xdot - x * zdot;
  rxvz = x * ydot - y * xdot;
  rxvsq = rxvx*rxvx + rxvy*rxvy + rxvz*rxvz;
  rinv5 = 1.0 / (r*r*r*r*r);
  fac1 = -rinv5 * (rm2*rm2 + 3.0*rxvsq);
  fac2 = 2.0 * rv / (r * r * rm2);
  xddot = x * fac1 + xdot * fac2;
  yddot = y * fac1 + ydot * fac2;
  zddot = z * fac1 + zdot * fac2;
}

//----------------------------------------------------------------------------------------
//! \fn void gravAccel(...)
//! \brief gravAccel: Gravitational acceleration from the generalized Newtonian potential of T&R13 ( https://arxiv.org/pdf/1303.4068 )
void gravAccel(
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
  Real x, y, z;
  Real xdot, ydot, zdot;
  Real xddot, yddot, zddot;
  Real pxdot, pydot, pzdot;
  Real mask;
  
  for (int k=pmb->ks; k<=pmb->ke; k++) {
    for (int j=pmb->js; j<=pmb->je; j++) {
      for (int i=pmb->is; i<=pmb->ie; i++) {

        // density
        rho = prim(IDN, k, j, i);
        
        // positions
        x = pmb->pcoord->x1v(i);
        y = pmb->pcoord->x2v(j);
        z = pmb->pcoord->x3v(k);

        // velocities
        xdot = prim(IVX, k, j, i);
        ydot = prim(IVY, k, j, i);
        zdot = prim(IVZ, k, j, i);

        // mask out floor material
        mask = 1.0 - pmb->pscalars->r(0, k, j, i);

        // compute acceleration
        TR13Accel(x, y, z, xdot, ydot, zdot, xddot, yddot, zddot);
        
        // momentum change       
        pxdot = mask * rho * xddot * dt;
        pydot = mask * rho * yddot * dt;
        pzdot = mask * rho * zddot * dt;
        
        // momentum and energy source terms
        cons(IM1, k, j, i) += pxdot;
        cons(IM2, k, j, i) += pydot;
        cons(IM3, k, j, i) += pzdot;
        cons(IEN, k, j, i) += pxdot * xdot + pydot * ydot + pzdot * zdot
                              + (pxdot*pxdot + pydot*pydot + pzdot*pzdot) / (2.0 * rho);

      }
    }
  }

};

Real myTimeStep(MeshBlock *pmb) { return poly->dt_max; }

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

  for (int k=kl-ngh; k<=kl-1; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {

        // set ghost zone primatives: diode
        prim(IDN, k, j, i) = prim(IDN, kl, j, i);
        prim(IVX, k, j, i) = prim(IVX, kl, j, i);
        prim(IVY, k, j, i) = prim(IVY, kl, j, i);
        prim(IVZ, k, j, i) = std::min(0.0, prim(IVZ, kl, j, i));
        prim(IPR, k, j, i) = prim(IPR, kl, j, i);

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

  for (int k=ku+1; k<=ku+ngh; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {

        // set ghost zone primatives: diode
        prim(IDN, k, j, i) = prim(IDN, ku, j, i);
        prim(IVX, k, j, i) = prim(IVX, ku, j, i);
        prim(IVY, k, j, i) = prim(IVY, ku, j, i);
        prim(IVZ, k, j, i) = std::max(0.0, prim(IVZ, ku, j, i));
        prim(IPR, k, j, i) = prim(IPR, ku, j, i);

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

  for (int k=kl; k<=ku; k++) {
    for (int j=jl-ngh; j<=jl-1; j++) {
      for (int i=il; i<=iu; i++) {

        // set ghost zone primatives: diode
        prim(IDN, k, j, i) = prim(IDN, k, jl, i);
        prim(IVX, k, j, i) = prim(IVX, k, jl, i);
        prim(IVY, k, j, i) = std::min(0.0, prim(IVY, k, jl, i));
        prim(IVZ, k, j, i) = prim(IVZ, k, jl, i);
        prim(IPR, k, j, i) = prim(IPR, k, jl, i);

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

  for (int k=kl; k<=ku; k++) {
    for (int j=ju+1; j<=ju+ngh; j++) {
      for (int i=il; i<=iu; i++) {

        // set ghost zone primatives: diode
        prim(IDN, k, j, i) = prim(IDN, k, ju, i);
        prim(IVX, k, j, i) = prim(IVX, k, ju, i);
        prim(IVY, k, j, i) = std::max(0.0, prim(IVY, k, ju, i));
        prim(IVZ, k, j, i) = prim(IVZ, k, ju, i);
        prim(IPR, k, j, i) = prim(IPR, k, ju, i);

      }
    }
  }

}

void bndInjXin(
  MeshBlock *pmb, 
  Coordinates *pco,
  AthenaArray<Real> &prim, 
  FaceField &b,
  Real time, 
  Real dt,
  int il, int iu, int jl, 
  int ju, int kl, int ku, 
  int ngh
) {

  Real y, z;
  Real yp, zp;
  Real frac;
  Real rho, pres;
  Real xdot, ydot, zdot;
  Real taper;
  
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il-ngh; i<=il-1; i++) {

        // positions
        yp = pmb->pcoord->x2v(j) - s2->y;
        zp = pmb->pcoord->x3v(k);
        frac = yp*yp / (s2->yp_max*s2->yp_max) + zp*zp / (s2->zp_max*s2->zp_max);

        if ( frac < 1.0 ) { // in stream

          rho  = s2->rho * interp(frac, poly->size, 0.0, 1.0, poly->dfrac, poly->rho_rhoc);
          pres = s2->K * pow(rho, 5.0/3.0);
          xdot = s2->xdot + s2->dvx_dy * yp;
          ydot = s2->ydot + s2->dvy_dy * yp;
          zdot = s2->dvz_dz * zp;
          
          if ( frac < 0.8 ) {
          
            // set ghost zone primatives: stream injection
            prim(IDN, k, j, i) = rho;
            prim(IVX, k, j, i) = xdot;
            prim(IVY, k, j, i) = ydot;
            prim(IVZ, k, j, i) = zdot;
            prim(IPR, k, j, i) = 1.83685e-08 * s1->rho; //pres;

          } else {

            // set ghost zone primatives: taper
            taper = (frac - 0.8) / 0.2;
            prim(IDN, k, j, i) = (1.0 - taper) * rho + taper * prim(IDN, k, j, il);
            prim(IVX, k, j, i) = (1.0 - taper) * xdot + taper * std::min(0.0, prim(IVX, k, j, il));
            prim(IVY, k, j, i) = (1.0 - taper) * ydot + taper * prim(IVY, k, j, il);
            prim(IVZ, k, j, i) = (1.0 - taper) * zdot + taper * prim(IVZ, k, j, il);
            prim(IPR, k, j, i) = (1.0 - taper) * 1.83685e-08 * s1->rho + taper * prim(IPR, k, j, il);

          }

        } else { // out of stream

          // set ghost zone primatives: diode
          prim(IDN, k, j, i) = prim(IDN, k, j, il);
          prim(IVX, k, j, i) = std::min(0.0, prim(IVX, k, j, il));
          prim(IVY, k, j, i) = prim(IVY, k, j, il);
          prim(IVZ, k, j, i) = prim(IVZ, k, j, il);
          prim(IPR, k, j, i) = prim(IPR, k, j, il);

        }

      }
    }
  }

};

void bndInjXout(
  MeshBlock *pmb, 
  Coordinates *pco,
  AthenaArray<Real> &prim, 
  FaceField &b,
  Real time, 
  Real dt,
  int il, int iu, int jl, 
  int ju, int kl, int ku, 
  int ngh
) {

  Real y, z;
  Real yp, zp;
  Real frac;
  Real rho, pres;
  Real xdot, ydot, zdot;
  Real taper;
  
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=iu+1; i<=iu+ngh; i++) {

        // positions
        yp = pmb->pcoord->x2v(j) - s1->y;
        zp = pmb->pcoord->x3v(k);
        frac = yp*yp / (s1->yp_max*s1->yp_max) + zp*zp / (s1->zp_max*s1->zp_max);

        if ( frac < 1.0 && time > poly->delay ) { // in stream

          rho  = s1->rho * interp(frac, poly->size, 0.0, 1.0, poly->dfrac, poly->rho_rhoc);
          pres = s1->K * pow(rho, 5.0/3.0);
          xdot = s1->xdot + s1->dvx_dy * yp;
          ydot = s1->ydot + s1->dvy_dy * yp;
          zdot = s1->dvz_dz * zp;

          if ( frac < 0.8 ) {
          
            // set ghost zone primatives: stream injection
            prim(IDN, k, j, i) = rho;
            prim(IVX, k, j, i) = xdot;
            prim(IVY, k, j, i) = ydot;
            prim(IVZ, k, j, i) = zdot;
            prim(IPR, k, j, i) = 1.83685e-08 * s1->rho; //pres;

          } else {

            // set ghost zone primatives: taper
            taper = (frac - 0.8) / 0.2;
            prim(IDN, k, j, i) = (1.0 - taper) * rho + taper * prim(IDN, k, j, iu);
            prim(IVX, k, j, i) = (1.0 - taper) * xdot + taper * std::max(0.0, prim(IVX, k, j, iu));
            prim(IVY, k, j, i) = (1.0 - taper) * ydot + taper * prim(IVY, k, j, iu);
            prim(IVZ, k, j, i) = (1.0 - taper) * zdot + taper * prim(IVZ, k, j, iu);
            prim(IPR, k, j, i) = (1.0 - taper) * 1.83685e-08 * s1->rho + taper * prim(IPR, k, j, iu);

          }

        } else { // out of stream

          // set ghost zone primatives: diode
          prim(IDN, k, j, i) = prim(IDN, k, j, iu);
          prim(IVX, k, j, i) = std::max(0.0, prim(IVX, k, j, iu));
          prim(IVY, k, j, i) = prim(IVY, k, j, iu);
          prim(IVZ, k, j, i) = prim(IVZ, k, j, iu);
          prim(IPR, k, j, i) = prim(IPR, k, j, iu);

        }

      }
    }
  }

};

void Mesh::InitUserMeshData(ParameterInput *pin) {

  EnrollUserBoundaryFunction(BoundaryFace::inner_x3, bndDiodeZin);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x3, bndDiodeZout);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x2, bndDiodeYin);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x2, bndDiodeYout);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, bndInjXin);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x1, bndInjXout);
  EnrollUserExplicitSourceFunction(gravAccel);
  EnrollUserTimeStepFunction(myTimeStep);

  return;

};

// void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {

//   AllocateRealUserMeshDataField(3);
//   iuser_meshblock_data[0].NewAthenaArray(257, 257);
//   iuser_meshblock_data[1].NewAthenaArray(257, 257);
//   iuser_meshblock_data[2].NewAthenaArray(256, 256);
  
//   for(int j=0; j<257; j++) {
//     for(int i=0; i<257; i++) {
//       iuser_mesh_data[0](j, i) = static_cast<Real>(i)/256.0 * 2.0 - 1.0;           // cos(th)
//       iuser_mesh_data[1](j, i) = (static_cast<Real>(j)/256.0 * 2.0 - 1.0) * M_PI;  // phi
//       iuser_mesh_data[2](j, i) = 0.0;                                              // mass flux                  
//     }
//   }

// }

// void MeshBlock::UserWorkInLoop(void) {

//   Real x, y, z;
//   Real r, cos_th, phi;

//   for(int k=ks; k<=ke; k++) {
//     for(int j=js; j<=je; j++) {
//       for(int i=is; i<=ie; i++) {
  
//         // positions
//         x = pmb->pcoord->x1v(i) - xc;
//         y = pmb->pcoord->x2v(j) - yc;
//         z = pmb->pcoord->x3v(k);
        
//         r = sqrt(x*x + y*y + z*z);
//         if (r - )

//         cos_th = z / r;
//         phi = atan2(y, x);


      
//       }
//     }
//   }

// }

// void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  
//   Real x, y, z;
//   Real xdot, ydot, zdot;
//   Real rho, pres, vsq;
//   Real r, rv, rm2;
//   Real rxvx, rxvy, rxvz, rxvsq;
//   Real ener, h, be;
  
//   for(int k=ks; k<=ke; k++) {
//     for(int j=js; j<=je; j++) {
//       for(int i=is; i<=ie; i++) {
        
//         // positions
//         x = pmb->pcoord->x1v(i);
//         y = pmb->pcoord->x2v(j);
//         z = pmb->pcoord->x3v(k);

//         // velocities
//         xdot = prim(IVX, k, j, i);
//         ydot = prim(IVY, k, j, i);
//         zdot = prim(IVZ, k, j, i);
        
//         // fluid variables
//         rho = phydro->w(IDN, k, j, i);
//         pres = phydro->w(IPR, k, j, i);

//         // orbital energy and angular momentum
//         r = sqrt(x*x + y*y + z*z);
//         rm2 = r - 2.0;
//         rv = x * xdot + y * ydot + z * zdot;
//         rxvx = y * zdot - z * ydot;
//         rxvy = z * xdot - x * zdot;
//         rxvz = x * ydot - y * xdot;
//         rxvsq = rxvx*rxvx + rxvy*rxvy + rxvz*rxvz;
//         ener = 0.5 / (rm2*rm2) * (rv*rv + r*rm2 * rxvsq / (r*r)) - 1.0/r;
//         be = ener + gam / (gam - 1.0) * pres / rho;
        
//         user_out_var(2, k, j, i) = be;
//         user_out_var(2, k, j, i) = rho*rho;
//         user_out_var(3, k, j, i) = rho*pres;
//       }
//     }
//   }

// }

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Problem Generator for the TDE self-intersection problem
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  std::stringstream msg;

  Real dfloor = pin->GetReal("hydro", "dfloor");
  Real pfloor = pin->GetReal("hydro", "pfloor");
  Real gam = pin->GetReal("hydro", "gamma");
  poly->pres_inj = pfloor;

  // populate the outgoing stream struct
  s1->rho = pin->GetReal("problem", "rho1");
  s1->K = pin->GetReal("problem", "K1");
  s1->y = pin->GetReal("problem", "y1");
  s1->xdot = pin->GetReal("problem", "xdot1");
  s1->ydot = pin->GetReal("problem", "ydot1");
  s1->yp_max = pin->GetOrAddReal("problem", "yp_max1", 0.0);
  s1->zp_max = pin->GetOrAddReal("problem", "zp_max1", 0.0);
  s1->dvx_dy = pin->GetOrAddReal("problem", "dvx_dy1", 0.0);
  s1->dvy_dy = pin->GetOrAddReal("problem", "dvy_dy1", 0.0);
  s1->dvz_dz = pin->GetOrAddReal("problem", "dvz_dz1", 0.0);

  // populate the incoming stream struct
  s2->rho = pin->GetReal("problem", "rho2");
  s2->K = pin->GetReal("problem", "K2");
  s2->y = pin->GetReal("problem", "y2");
  s2->xdot = pin->GetReal("problem", "xdot2");
  s2->ydot = pin->GetReal("problem", "ydot2");
  s2->yp_max = pin->GetOrAddReal("problem", "yp_max2", 0.0);
  s2->zp_max = pin->GetOrAddReal("problem", "zp_max2", 0.0);
  s2->dvx_dy = pin->GetOrAddReal("problem", "dvx_dy2", 0.0);
  s2->dvy_dy = pin->GetOrAddReal("problem", "dvy_dy2", 0.0);
  s2->dvz_dz = pin->GetOrAddReal("problem", "dvz_dz2", 0.0);

  poly->dt_max = pin->GetOrAddReal("time", "dt_max", FLT_MAX);
  poly->delay = pin->GetOrAddReal("problem", "delay", 0.0);

  // polytrope parameters
  laneEmden le;
  Real xi;
  const int num_le = 16384;
  const Real xi_max = 2.6477;
  Real dxi = xi_max / static_cast<Real>(num_le - 1);

  // initialize polytrope arrays
  poly->size = num_le;
  poly->dfrac = 1.0 / static_cast<Real>(num_le - 1);
  poly->rho_rhoc.NewAthenaArray(num_le);
  le.th = 1.0 - 0.25 * dxi*dxi;
  le.phi = -0.5 * dxi*dxi*dxi;
  poly->rho_rhoc(0) = 1.0;

  // integrate Lane-Emden equation
  for ( int i=1; i<num_le; i++ ) {
    xi = dxi + static_cast<Real>(i) * dxi;
    poly->rho_rhoc(i) = std::pow(std::fmax(le.th, 0.0), 1.5);
    rk4<decltype(calcLaneEmden), laneEmden>(calcLaneEmden, dxi, xi, le);
  }
  
  Real x, rho, pres;
  
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        
        x = pcoord->x1v(i);
        rho = s1->rho;
        pres = 1.83685e-08 * rho; // p=nkT where T=1e5
        
        // set the initial conservative variables
        phydro->u(IDN, k, j, i) = rho / 1.e6;
        phydro->u(IM1, k, j, i) = 0.0;
        phydro->u(IM2, k, j, i) = 0.0;
        phydro->u(IM3, k, j, i) = 0.0;
        phydro->u(IEN, k, j, i) = pres / (gam - 1.0);
        pscalars->s(0, k, j, i) = rho / 1.e6;

      }
    }
  }

  return;
}
