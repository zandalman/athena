//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file eos_table_class.cpp
//! \brief Implements class EosTable for an EOS lookup table
//========================================================================================

// C headers

// C++ headers
#include <cmath>   // sqrt()
#include <fstream>
#include <iostream> // ifstream
#include <sstream>
#include <stdexcept> // std::invalid_argument
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "../inputs/ascii_table_reader.hpp"
#include "../inputs/hdf5_reader.hpp"
#include "../parameter_input.hpp"
#include "interp_table.hpp"

// Order of datafields for HDF5 EOS tables
const char *var_names1[] = {"lpres(leps,lrho)"};
const char *var_names2[] = {"legas(lpor,lrho)", "gam1(lpor,lrho)", "q(lpor,lrho)"};

//----------------------------------------------------------------------------------------
//! \fn void ReadBinaryTable(std::string fn, EosTable *peos_table, ParameterInput *pin)
//! \brief Read data from HDF5 EOS table and initialize interpolated table.

void ReadHDF5Table(std::string fn, EosTable *peos_table, ParameterInput *pin) {
#ifndef HDF5OUTPUT
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in EosTable::EosTable, ReadHDF5Table" << std::endl
        << "HDF5 EOS table specified, but HDF5 flag is not enabled."  << std::endl;
    ATHENA_ERROR(msg);
  }
#endif
  std::string dens_lim_field = pin->GetOrAddString("hydro", "EOS_dens_lim_field", "rholim");
  std::string eps_lim_field = pin->GetOrAddString("hydro", "EOS_eps_lim_field", "epslim");
  std::string por_lim_field = pin->GetOrAddString("hydro", "EOS_por_lim_field", "porlim");
  HDF5TableLoader(fn.c_str(), &peos_table->table1, 1, var_names1, eps_lim_field.c_str(), dens_lim_field.c_str());
  HDF5TableLoader(fn.c_str(), &peos_table->table2, 3, var_names2, por_lim_field.c_str(), dens_lim_field.c_str());
  peos_table->table1.GetSize(peos_table->nVar1, peos_table->nEps, peos_table->nRho);
  peos_table->table1.GetX2lim(peos_table->logEpsMin, peos_table->logEpsMax);
  peos_table->table1.GetX1lim(peos_table->logRhoMin, peos_table->logRhoMax);
  peos_table->table2.GetSize(peos_table->nVar2, peos_table->nPor, peos_table->nRho);
  peos_table->table2.GetX2lim(peos_table->logPorMin, peos_table->logPorMax);
  peos_table->table2.GetX1lim(peos_table->logRhoMin, peos_table->logRhoMax);
}

// ctor
EosTable::EosTable(ParameterInput *pin) :
    table1(), table2(), logRhoMin(), logRhoMax(),
    rhoUnit(pin->GetOrAddReal("hydro", "eos_rho_unit", 1.0)),
    egasUnit(pin->GetOrAddReal("hydro", "eos_egas_unit", 1.0)) {
  std::string eos_fn;
  eos_fn = pin->GetString("hydro", "eos_file_name");
  ReadHDF5Table(eos_fn, this, pin);
}
