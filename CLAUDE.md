# Athena++ fork: nuclear reaction network

Goal: add a nuclear network to Athena++ by adapting the existing chemistry
module (src/chemistry/).

Design notes and decisions: docs/nuclear-network-plan.md. Read this before
starting work, and update it when a design decision is made.

## Reference docs (read when relevant; don't load wholesale)
- docs/athena-wiki/: Athena++ wiki
- docs/nuclear-network-plan.md: our design decisions

## Architecture (baseline chemistry: H2 network, CVODE, adiabatic EOS)
- Selection: `configure.py` (~L708-766). `--chemistry=X` sets `CHEMNETWORK_HEADER`
  (pulled in by `src/scalars/scalars.hpp`) and compiles `src/chemistry/network/X.cpp`.
  NSPECIES is hard-coded per network name. `--chem_ode_solver=cvode` links `cvode.cpp`
  (or `forward_euler.cpp`); both define the same `ODEWrapper` class.
- Interface: the network is a class named `ChemNetwork` that derives from `NetworkWrapper`
  (`src/chemistry/network/network.hpp`). It must implement `InitializeNextStep(k,j,i)`,
  `RHS(t,y,E,ydot)` and `Edot(t,y,E)`; `Jacobian` is optional (`user_jac=true`). It also needs
  a static `species_names`. CVODE calls `NetworkWrapper::WrapRHS/WrapJacobian`
  (`network_wrapper.cpp`). `PassiveScalars` owns `chemnet` and `odew` by value.
- Storage: species are the first NSPECIES passive scalars, with `s = rho*r`. The ODE
  variable is `y = s/rho`. H2 treats y as n_s/n_H, which relies on the default "ism"
  units giving rho_code == n_H. After the solve, values are clipped at `scalar_floor`;
  nothing renormalizes them.
- Call site: `TimeIntegratorTaskList::IntegrateChemistry` (`task_list/time_integrator.cpp`).
  It runs on the last stage only, after INT_SCLR and before SRC_TERM/CONS2PRIM
  (first-order operator split). `ODEWrapper::Integrate` (`src/chemistry/cvode.cpp`) loops
  over cells: InitializeNextStep -> CVodeReInit -> CVode over the full dt. Setup is in
  `ODEWrapper::Initialize`, called from `Mesh::Initialize`.
- Energy: when NON_BAROTROPIC_EOS, internal energy (`u(IEN)` minus kinetic and magnetic
  energy) is ODE slot NSPECIES and is integrated with `Edot`. Afterwards `u(IEN)` is rebuilt
  directly. H2 gets T from its own Cv (`Thermo::CvCold`), not from the Athena EOS.
- Units: RHS/Edot take and return code units and compute in cgs, using `pmy_mesh->punit`
  (`src/units/units.cpp`). RHS multiplies by `code_time_cgs`; Edot converts with
  `code_energydensity_cgs` and `code_time_cgs`.

## Environment
  module load anaconda3/2020.11
  module load intel/2021.1.2
  module load intel-mpi/intel/2021.1.1
  module load hdf5/intel-2021.1/intel-mpi/1.10.6
