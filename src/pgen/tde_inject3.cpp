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

#include <cmath>      
#include <cstdio>     
#include <iostream>  
#include <sstream> 
#include <fstream>
#include <stdexcept> 
#include <string>
#include <cfloat>
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

struct pgen_globals {
  Real dtmax;
  Real gam;
  Real rsink;
  Real rho_sink;
  Real egas_sink;
};

pgen_globals* myglobals = new pgen_globals();

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

void calc_orb(Real x, Real y, Real z, Real xdot, Real ydot, Real zdot, Real &egrav, Real &hgrav) {
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
  egrav = 0.5 / (rm2*rm2) * (rv*rv + rm2/r * rxvsq) - 1.0/r;
  hgrav = r/rm2 * sqrt(rxvsq);
}

Real mass_acc(MeshBlock *pmb, int iout) {
  return pmb->ruser_meshblock_data[6](0);
}

Real ener_acc(MeshBlock *pmb, int iout) {
  return pmb->ruser_meshblock_data[6](1);
}

Real amom_acc(MeshBlock *pmb, int iout) {
  return pmb->ruser_meshblock_data[6](2);
}

void allSource(
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
  Real x, y, z, r;
  Real dx, dy, dz, dV;
  Real xdot, ydot, zdot;
  Real xddot, yddot, zddot;
  Real pxdot, pydot, pzdot;
  Real vsq, pres, egas, mask;
  Real egrav, hgrav;
  
  for (int k=pmb->ks; k<=pmb->ke; k++) {
    for (int j=pmb->js; j<=pmb->je; j++) {
      for (int i=pmb->is; i<=pmb->ie; i++) {

        // injection
        if (pmb->ruser_meshblock_data[5](k-pmb->ks, j-pmb->js, i-pmb->is) != 0.0) {
          for (int n=0; n<5; n++) {
            cons(n, k, j, i) = pmb->ruser_meshblock_data[n](k-pmb->ks, j-pmb->js, i-pmb->is);
          }
          cons_scalar(0, k, j, i) = pmb->ruser_meshblock_data[0](k-pmb->ks, j-pmb->js, i-pmb->is);
          continue;
        }
        
        // coordinates
        x = pmb->pcoord->x1v(i);
        y = pmb->pcoord->x2v(j);
        z = pmb->pcoord->x3v(k);
        r = sqrt(x*x + y*y + z*z);

        // cell size
        dx = pmb->pcoord->dx1v(i);
        dy = pmb->pcoord->dx2v(j);
        dz = pmb->pcoord->dx3v(k);
        dV = dx * dy * dz;

        // primatives
        rho = prim(IDN, k, j, i);
        xdot = prim(IVX, k, j, i);
        ydot = prim(IVY, k, j, i);
        zdot = prim(IVZ, k, j, i);
        pres = prim(IPR, k, j, i);
        egas = pres / (myglobals->gam - 1.0);
        mask = pmb->pscalars->r(0, k, j, i);
        vsq = xdot*xdot + ydot*ydot + zdot*zdot;
        
        // sink
        if (r <= myglobals->rsink) {
          
          cons(IDN, k, j, i) = myglobals->rho_sink;
          cons(IM1, k, j, i) = 0.0;
          cons(IM2, k, j, i) = 0.0;
          cons(IM3, k, j, i) = 0.0;
          cons(IEN, k, j, i) = myglobals->egas_sink;
          cons_scalar(0, k, j, i) = 0.0;
          if (mask > 0.0) {
            calc_orb(x, y, z, xdot, ydot, zdot, egrav, hgrav);
            pmb->ruser_meshblock_data[6](0) += mask * rho * dV;
            pmb->ruser_meshblock_data[6](1) += mask * (rho * egrav + pres / (myglobals->gam - 1.0)) * dV;
            pmb->ruser_meshblock_data[6](2) += mask * rho * hgrav * dV;
          }
          continue;
        
        }
          
        // gravitational acceleration
        TR13Accel(x, y, z, xdot, ydot, zdot, xddot, yddot, zddot);
      
        // momentum change       
        pxdot = rho * xddot * dt;
        pydot = rho * yddot * dt;
        pzdot = rho * zddot * dt;
        
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

Real myTimeStep(MeshBlock *pmb) { 
  return myglobals->dtmax; 
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  
  AllocateUserHistoryOutput(3);
  EnrollUserHistoryOutput(0, mass_acc, "mass_acc", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(1, ener_acc, "ener_acc", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(2, amom_acc, "amom_acc", UserHistoryOperation::sum);
  EnrollUserExplicitSourceFunction(allSource);
  EnrollUserTimeStepFunction(myTimeStep);
  return;

};

void readArr1D(hid_t file, const char* field, AthenaArray<Real> &array) {
  hid_t dataset = H5Dopen(file, field, H5P_DEFAULT);
  hid_t dataspace = H5Dget_space(dataset);
  hsize_t dims[1];
  H5Sget_simple_extent_dims(dataspace, dims, NULL);
  int num = static_cast<int>(dims[0]);
  array.NewAthenaArray(num);
  H5Dread(dataset, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, array.data());
  H5Sclose(dataspace);
  H5Dclose(dataset);
}

void readArr4D(hid_t file, const char* field, AthenaArray<Real> &array, int &nz, int &ny, int &nx) {
  hid_t dataset = H5Dopen(file, field, H5P_DEFAULT);
  hid_t dataspace = H5Dget_space(dataset);
  hsize_t dims[4];
  H5Sget_simple_extent_dims(dataspace, dims, NULL);
  int nvar = static_cast<int>(dims[0]);
  nz = static_cast<int>(dims[1]);
  ny = static_cast<int>(dims[2]);
  nx = static_cast<int>(dims[3]);
  array.NewAthenaArray(nvar, nz, ny, nx);
  H5Dread(dataset, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, array.data());
  H5Sclose(dataspace);
  H5Dclose(dataset);
}

Real trilin(
  const AthenaArray<Real> &arr,
  int n,
  Real wx, Real wy, Real wz,
  int ix, int iy, int iz
) {
  Real all = (1.0-wx) * arr(n, iz, iy, ix) + wx * arr(n, iz, iy, ix+1);
  Real alu = (1.0-wx) * arr(n, iz, iy+1, ix) + wx * arr(n, iz, iy+1, ix+1);
  Real aul = (1.0-wx) * arr(n, iz+1, iy, ix) + wx * arr(n, iz+1, iy, ix+1);
  Real auu = (1.0-wx) * arr(n, iz+1, iy+1, ix) + wx * arr(n, iz+1, iy+1, ix+1);
  Real al  = (1.0-wy) * all + wy * alu;
  Real au  = (1.0-wy) * aul + wy * auu;
  return (1.0-wz) * al + wz * au;
}

void getInterpIdx(Real x, Real xmin, Real xmax, int nx, int &ix, Real &wx) {
  Real dx = (xmax - xmin) / static_cast<Real>(nx);
  Real fix = (x - xmin - dx/2.0) / (xmax - xmin - dx) * static_cast<Real>(nx-1);
  ix = static_cast<int>(std::floor(fix));
  wx = fix - static_cast<Real>(ix);
}

void resampleCons(
  MeshBlock *pmb,
  const AthenaArray<Real> &ximin, 
  const AthenaArray<Real> &ximax, 
  const AthenaArray<Real> &cons,
  int nx, int ny, int nz,
  Real gam, Real rinj
) {
  
  // initialize variables
  int ii, jj, kk;
  Real wx, wy, wz;
  int ix, iy, iz;  
  Real x, y, z;
  Real xc, yc, zc;
  Real rp;
  
  Real rho, Etot;               
  Real momx, momy, momz;
  Real momr, momth, momphi;

  zc = (ximin(0) + ximax(0)) / 2.0;
  yc = (ximin(1) + ximax(1)) / 2.0;
  xc = (ximin(2) + ximax(2)) / 2.0;

  // resample conserved variables to new grid
  for (int k=pmb->ks; k<=pmb->ke; k++) {
    for (int j=pmb->js; j<=pmb->je; j++) {
      for (int i=pmb->is; i<=pmb->ie; i++) {

        ii = i-pmb->is;
        jj = j-pmb->js;
        kk = k-pmb->ks;
        
        // get positions
        x = pmb->pcoord->x1v(i);
        y = pmb->pcoord->x2v(j);
        z = pmb->pcoord->x3v(k);
        rp = sqrt((x-xc)*(x-xc) + (y-yc)*(y-yc) + (z-zc)*(z-zc));

        // compute interpolation indices
        getInterpIdx(z, ximin(0), ximax(0), nz, iz, wz);
        getInterpIdx(y, ximin(1), ximax(1), ny, iy, wy);
        getInterpIdx(x, ximin(2), ximax(2), nx, ix, wx);

        // if out-of-bounds, zero new array
        if (ix<0 || ix>=nx-1 || iy<0 || iy>=ny-1 || iz<0 || iz>=nz-1) {   
          for (int n=0; n<6; n++) {
            pmb->ruser_meshblock_data[n](kk, jj, ii) = 0.0;
          }
        } else {
          
          // else do tri-linear interpolation of pn conservatives to new array
          rho  = trilin(cons, 0, wx, wy, wz, ix, iy, iz);
          momx = trilin(cons, 1, wx, wy, wz, ix, iy, iz);
          momy = trilin(cons, 2, wx, wy, wz, ix, iy, iz);
          momz = trilin(cons, 3, wx, wy, wz, ix, iy, iz);
          Etot = trilin(cons, 4, wx, wy, wz, ix, iy, iz);

          // set new array values
          pmb->ruser_meshblock_data[0](kk, jj, ii) = rho;
          pmb->ruser_meshblock_data[1](kk, jj, ii) = momx;
          pmb->ruser_meshblock_data[2](kk, jj, ii) = momy;
          pmb->ruser_meshblock_data[3](kk, jj, ii) = momz;
          pmb->ruser_meshblock_data[4](kk, jj, ii) = Etot;
          pmb->ruser_meshblock_data[5](kk, jj, ii) = static_cast<Real>(rp <= rinj);
        }
      }
    }
  }
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {

  // allocate user mesh block data
  AllocateRealUserMeshBlockDataField(7);
  ruser_meshblock_data[0].NewAthenaArray(block_size.nx3, block_size.nx2, block_size.nx1); // rho
  ruser_meshblock_data[1].NewAthenaArray(block_size.nx3, block_size.nx2, block_size.nx1); // mom1
  ruser_meshblock_data[2].NewAthenaArray(block_size.nx3, block_size.nx2, block_size.nx1); // mom2
  ruser_meshblock_data[3].NewAthenaArray(block_size.nx3, block_size.nx2, block_size.nx1); // mom3
  ruser_meshblock_data[4].NewAthenaArray(block_size.nx3, block_size.nx2, block_size.nx1); // Etot
  ruser_meshblock_data[5].NewAthenaArray(block_size.nx3, block_size.nx2, block_size.nx1); // mask
  
  ruser_meshblock_data[6].NewAthenaArray(3); // accretion tracking
  ruser_meshblock_data[6](0) = 0.0;
  ruser_meshblock_data[6](1) = 0.0;
  ruser_meshblock_data[6](2) = 0.0;

  // read input file
  Real rinj = pin->GetReal("problem", "rinj");
  Real gam = pin->GetOrAddReal("hydro", "gamma", 4.0/3.0);

  // read injection data file and store in user mesh block data
  int nx, ny, nz;
  AthenaArray<Real> ximin, ximax, cons;
  std::string filename = pin->GetString("problem", "inj_file");
  hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  readArr1D(file, "ximin", ximin);
  readArr1D(file, "ximax", ximax);
  readArr4D(file, "cons", cons, nz, ny, nx);
  H5Fclose(file);
  resampleCons(this, ximin, ximax, cons, nx, ny, nz, gam, rinj);
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Problem Generator for the TDE self-intersection problem
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  std::stringstream msg;

  // initialize variables
  Real x, y, z, r, datmo, patmo;
  
  // read input file
  Real rho_min = pin->GetReal("hydro", "rho_min");
  Real rho_pow = pin->GetReal("hydro", "rho_pow");
  Real pgas_min = pin->GetReal("hydro", "pgas_min");
  Real pgas_pow = pin->GetReal("hydro", "pgas_pow");
  Real gam = pin->GetReal("hydro", "gamma");
  Real rsink = pin->GetReal("problem", "rsink");

  // set globals
  myglobals->dtmax = pin->GetOrAddReal("time", "dt_max", FLT_MAX);
  myglobals->gam = gam;
  myglobals->rsink = rsink;
  myglobals->rho_sink = rho_min * pow(rsink, rho_pow);
  myglobals->egas_sink = pgas_min * pow(rsink, pgas_pow) / (gam - 1.0);
  
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {

        x = pcoord->x1v(i);
        y = pcoord->x2v(j);
        z = pcoord->x3v(k);
        r = std::max(rsink, sqrt(x*x + y*y + z*z));
        datmo = 1.e2 * rho_min * pow(r, rho_pow);
        patmo = 1.e2 * pgas_min * pow(r, pgas_pow);

        // set the initial conservative variables
        phydro->u(IDN, k, j, i) = datmo;
        phydro->u(IM1, k, j, i) = 0.0;
        phydro->u(IM2, k, j, i) = 0.0;
        phydro->u(IM3, k, j, i) = 0.0;
        phydro->u(IEN, k, j, i) = patmo / (gam - 1.0);
        pscalars->r(0, k, j, i) = 0.0;
      }
    }
  }

  return;
}
