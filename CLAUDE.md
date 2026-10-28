# Athena++ fork: nuclear reaction network

Goal: add a nuclear network to Athena++ by adapting the existing chemistry
module (src/chemistry/).

Design notes and decisions: docs/nuclear-network-plan.md. Read this before
starting work, and update it when a design decision is made.

**Handoff (2026-10-28):** the project passed from the PI to a student, who is new to this
code. Read docs/nuclear-network-handoff.md first: it covers environment setup on stellar,
how to reproduce the validated one-zone tests, the pitfalls, and next steps. Teach before
building. Explain the existing work and the reasons behind it, and have the student
reproduce the tests before adding features. Give shell commands one per line (no
backslash continuations).

## Reference docs (read when relevant; don't load wholesale)
- docs/athena-wiki/: Athena++ wiki
- docs/nuclear-network-plan.md: our design decisions
- docs/nuclear-network-handoff.md: setup, status, pitfalls, next steps

## Nuclear network (status)
- `--chemistry=nuclear_<small|aprox13>` uses `src/chemistry/network/nuclear.{hpp,cpp}`
  around pynucastro-generated headers in `src/chemistry/network/nuclear/<name>/`. The
  networks are defined in `tools/nuclear/networks.py`. Fixed T, isothermal EOS only.
- Validated against a Python reference of the same network with the one-zone test
  `--prob=nuc_onezone` (`tools/nuclear/check_onezone.py`).

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
`source modules.sh` (repo root) loads the following and activates conda env `nucnet`
(pynucastro, scipy, matplotlib). Do it before building *and* before running (the Intel
runtime libs are needed). Compile on stellar-vis1 or a login node; icpx isn't usable on
compute nodes.
  module load anaconda3/2024.6
  module load intel/2021.1.2
  module load intel-mpi/intel/2021.1.1
  module load hdf5/intel-2021.1/intel-mpi/1.10.6
  module load gcc-toolset/13   # libstdc++ for C++20; build with --cxx=icpx -std=c++20
