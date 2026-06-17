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
};

pgen_globals* myglobals = new pgen_globals();

void sph2Cart(Real r, Real th, Real phi, Real &x, Real &y, Real &z) {
  x = r * sin(th) * cos(phi);
  y = r * sin(th) * sin(phi);
  z = r * cos(th);
}

void cart2Sph(Real x, Real y, Real z, Real vx, Real vy, Real vz, Real &vr, Real &vth, Real &vphi) {
  Real s = sqrt(x*x + y*y);
  Real r = sqrt(x*x + y*y + z*z);
  vr = (x * vx + y * vy + z * vz) / r;
  vth = ((x * vx + y * vy) * z - s*s * vz) / (r * s);
  vphi = (-y * vx + x * vy) / s;
}

void TR13Accel(Real r, Real th, Real phi, Real vr, Real vth, Real vphi, Real& ar, Real& ath, Real &aphi) {
  Real rm2, rm3, ri, ri2, ri4, rm2i;
  Real sin_th, cos_th, sin_thi;
  rm2 = r - 2.0;
  rm3 = r - 3.0;
  ri = 1.0/r;
  ri2 = ri*ri;
  ri4 = ri2*ri2;
  rm2i = 1.0/rm2;
  sin_th = sin(th);
  cos_th = cos(th);
  sin_thi = 1.0/sin_th;
  ar   = -ri4 * rm2*rm2 + 2.0 * vr*vr * ri * rm2i + ri * (rm3 * ri - 1.0) * (vth*vth + vphi*vphi);
  ath  = -2.0 * vr * vth * ri * (rm3 * rm2i - 1.0);
  aphi = -2.0 * vr * vphi * ri * (rm3 * rm2i - 1.0);
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
  Real r, th, phi;
  Real vr, vth, vphi;
  Real ar, ath, aphi;
  Real prdot, pthdot, pphidot;
  Real mask;
  
  for (int k=pmb->ks; k<=pmb->ke; k++) {
    for (int j=pmb->js; j<=pmb->je; j++) {
      for (int i=pmb->is; i<=pmb->ie; i++) {

        // density
        rho = prim(IDN, k, j, i);
        r = pmb->pcoord->x1v(i);
        th = pmb->pcoord->x2v(j);
        phi = pmb->pcoord->x3v(k);
        vr = prim(IVX, k, j, i);
        vth = prim(IVY, k, j, i);
        vphi = prim(IVZ, k, j, i);
        mask = pmb->pscalars->r(0, k, j, i);
        TR13Accel(r, th, phi, vr, vth, vphi, ar, ath, aphi);
        
        // momentum change       
        prdot   = mask * rho * ar * dt;
        pthdot  = mask * rho * ath * dt;
        pphidot = mask * rho * aphi * dt;
        
        // momentum and energy source terms
        cons(IM1, k, j, i) += prdot;
        cons(IM2, k, j, i) += pthdot;
        cons(IM3, k, j, i) += pphidot;
        cons(IEN, k, j, i) += prdot * vr + pthdot * vth + pphidot * vphi
                              + (prdot*prdot + pthdot*pthdot + pphidot*pphidot) / (2.0 * rho);

      }
    }
  }

};

//----------------------------------------------------------------------------------------
//! \fn void inject(...)
//! \brief inject: Injection
void inject(
  MeshBlock *pmb, 
  const Real time, 
  const Real dt,
  const AthenaArray<Real> &prim, 
  const AthenaArray<Real> &prim_scalar,
  const AthenaArray<Real> &bcc, 
  AthenaArray<Real> &cons,
  AthenaArray<Real> &cons_scalar
) {

  int ii, jj, kk;
  
  for (int k=pmb->ks; k<=pmb->ke; k++) {
    for (int j=pmb->js; j<=pmb->je; j++) {
      for (int i=pmb->is; i<=pmb->ie; i++) {
        
        kk = k-pmb->ks;
        jj = j-pmb->js;
        ii = i-pmb->is;
        
        // continue if outside injection region
        if (pmb->ruser_meshblock_data[5](kk, jj, ii) == 0.0) continue;

        // injection
        for (int n=0; n<5; n++) {
          cons(n, k, j, i) = pmb->ruser_meshblock_data[n](kk, jj, ii);
        }
        cons_scalar(0, k, j, i) = pmb->ruser_meshblock_data[0](kk, jj, ii);

      }
    }
  }

};

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
  gravAccel(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);
  inject(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);
}

void InflowBoundary(
  MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &bb,
  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh
) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il-ngh; i<=il-1; ++i) {
        prim(IDN,k,j,i) = prim(IDN,k,j,il);
        prim(IPR,k,j,i) = prim(IPR,k,j,il);
        prim(IVX,k,j,i) = std::min(prim(IVX,k,j,il), 0.0);
        prim(IVY,k,j,i) = prim(IVY,k,j,il);
        prim(IVZ,k,j,i) = prim(IVZ,k,j,il);
      }
    }
  }
}

void OutflowBoundary(
  MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &bb, 
  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh
) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=iu+1; i<=iu+ngh; ++i) {
        prim(IDN,k,j,i) = prim(IDN,k,j,iu);
        prim(IPR,k,j,i) = prim(IPR,k,j,iu);
        prim(IVX,k,j,i) = std::max(prim(IVX,k,j,iu), 0.0);
        prim(IVY,k,j,i) = prim(IVY,k,j,iu);
        prim(IVZ,k,j,i) = prim(IVZ,k,j,iu);
      }
    }
  }
}

Real myTimeStep(MeshBlock *pmb) { 
  return myglobals->dtmax; 
}

void Mesh::InitUserMeshData(ParameterInput *pin) {

  // AllocateUserHistoryOutput(2);
  // EnrollUserHistoryOutput(0, calc_dt_cs, "dt_cs", UserHistoryOperation::min);
  // EnrollUserHistoryOutput(1, calc_dt_vel, "dt_vel", UserHistoryOperation::min);
  
  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, InflowBoundary);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x1, OutflowBoundary);
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
  Real r, th, phi;    
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
        r = pmb->pcoord->x1v(i);
        th = pmb->pcoord->x2v(j);
        phi = pmb->pcoord->x3v(k);
        sph2Cart(r, th, phi, x, y, z);
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
          
          // convert cartesian pn to gr
          cart2Sph(x, y, z, momx, momy, momz, momr, momth, momphi);

          // set new array values
          pmb->ruser_meshblock_data[0](kk, jj, ii) = rho;
          pmb->ruser_meshblock_data[1](kk, jj, ii) = momr;
          pmb->ruser_meshblock_data[2](kk, jj, ii) = momth;
          pmb->ruser_meshblock_data[3](kk, jj, ii) = momphi;
          pmb->ruser_meshblock_data[4](kk, jj, ii) = Etot;
          pmb->ruser_meshblock_data[5](kk, jj, ii) = static_cast<Real>(rp <= rinj);
        }
      }
    }
  }
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {

  // allocate user mesh block data
  AllocateRealUserMeshBlockDataField(6);
  ruser_meshblock_data[0].NewAthenaArray(block_size.nx3, block_size.nx2, block_size.nx1); // rho
  ruser_meshblock_data[1].NewAthenaArray(block_size.nx3, block_size.nx2, block_size.nx1); // mom1
  ruser_meshblock_data[2].NewAthenaArray(block_size.nx3, block_size.nx2, block_size.nx1); // mom2
  ruser_meshblock_data[3].NewAthenaArray(block_size.nx3, block_size.nx2, block_size.nx1); // mom3
  ruser_meshblock_data[4].NewAthenaArray(block_size.nx3, block_size.nx2, block_size.nx1); // Etot
  ruser_meshblock_data[5].NewAthenaArray(block_size.nx3, block_size.nx2, block_size.nx1); // mask

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
  readArr4D(file, "cons", cons, nx, ny, nz);
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
  Real r, datmo, patmo;
  
  // read input file
  Real rho_min = pin->GetReal("hydro", "rho_min");
  Real rho_pow = pin->GetReal("hydro", "rho_pow");
  Real pgas_min = pin->GetReal("hydro", "pgas_min");
  Real pgas_pow = pin->GetReal("hydro", "pgas_pow");
  Real gam = pin->GetReal("hydro", "gamma");

  // set globals
  myglobals->dtmax = pin->GetOrAddReal("time", "dt_max", FLT_MAX);
  
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {

        r = pcoord->x1v(i);
        datmo = 1.e2 * rho_min * pow(r, rho_pow);
        patmo = 1.e2 * pgas_min * pow(r, pgas_pow);

        // set the initial primative variables
        phydro->w(IDN, k, j, i) = datmo;
        phydro->w(IVX, k, j, i) = 0.0;
        phydro->w(IVY, k, j, i) = 0.0;
        phydro->w(IVZ, k, j, i) = 0.0;
        phydro->w(IPR, k, j, i) = patmo;
        pscalars->s(0, k, j, i) = 0.0;
      }
    }
  }

  // convert primative to conservative variables
  peos->PrimitiveToConserved(
    phydro->w, pfield->bcc, phydro->u, pcoord, 
    is-NGHOST, ie+NGHOST, js-NGHOST, je+NGHOST, ks-NGHOST, ke+NGHOST
  );   
  peos->PassiveScalarPrimitiveToConserved(
    pscalars->r, phydro->u, pscalars->s, pcoord,
    is-NGHOST, ie+NGHOST, js-NGHOST, je+NGHOST, ks-NGHOST, ke+NGHOST
  );

  return;
}
