# Nuclear network: handoff notes (for Claude)

Written 2026-10-28, at the end of the first working session. The project is being handed
from the PI (account za9132) to a student, who uses the same cluster (Princeton
**stellar**) but their own account. Read this, then `docs/nuclear-network-plan.md`
(design decisions and results), before doing anything.

## How to work with the student
- The student is new to this code. **Start by explaining before building**: walk them
  through what exists and why, and check their understanding. Don't jump to the next
  feature.
  Suggested first session:
  1. The big picture: Athena++ chemistry module -> CVODE -> network wrapper ->
     pynucastro-generated rates (see "How the pieces fit" below and the Architecture
     section of `CLAUDE.md`).
  2. Have the student set up the environment and reproduce both one-zone tests
     themselves (`small` and `aprox13`). Explain what each check number means.
  3. Only then move on to "Next steps".
- Explain the physics choices too, since they're as important as the code: why REACLIB,
  what the (a,p)(p,g) approximation is, why the C/O proton channels need the equilibrium
  approximation, and why fixed T comes first.
- Give shell commands **one per line, with no backslash continuations**. Pasted
  multi-line commands repeatedly broke in the first session (arguments ended up on a new
  line, and the shell ran them as a separate command).
- In the first session Claude had no shell tool, so the user ran every command and pasted
  the output. If you do have one, follow the cluster rules in the global instructions
  (vis node, Slurm, etc.).

## Environment setup for a new account
1. **Log in to `stellar-vis1`** (the vis node). Compile there or on a login node. `icpx` is
   not usable on compute nodes: the Intel oneAPI module files are missing there.
2. **Repo**: the work is on branch `nuclear` of the PI's fork. Make sure the PI has
   committed and pushed (at handoff, much of it was uncommitted: see "Files" below). The
   student clones it. Set `ATHENA_DIR` to the directory that *contains* `athena++_milo`
   (the run commands use `$ATHENA_DIR/athena++_milo/...`).
3. **Modules**: `source modules.sh` (repo root). It does `module purge` and loads
   anaconda3/2024.6, intel/2021.1.2, intel-mpi, hdf5 and gcc-toolset/13, then
   `conda activate nucnet`.
   - gcc-toolset/13 is needed even though we compile with `icpx`: icpx uses gcc's
     libstdc++, and the system gcc 8.5 is too old for C++20.
   - The modules must also be loaded at **run time**. Otherwise `athena` fails with
     `libimf.so: cannot open shared object file`. Put `source modules.sh` in Slurm scripts.
4. **Conda env `nucnet`** (the student must create their own; `modules.sh` assumes the
   name). It needs pynucastro, numpy, scipy and matplotlib, e.g.
   `conda create -n nucnet -c conda-forge python=3.11 pynucastro scipy matplotlib`.
   Check that it has `make_CO_burning_approx`:
   `python -c "import pynucastro as p; print(p.__version__, hasattr(p.PythonNetwork, 'make_CO_burning_approx'))"`.
   Per RC guidance, keep large conda pkgs/caches on /scratch, not /home.
5. **CVODE (SUNDIALS)**: configure uses `--cvode_path=$CVODE_DIR`. The PI's install
   location wasn't recorded. Ask the PI for the path (and read access), or have the student
   build SUNDIALS themselves (CVODE with the serial NVector and dense linear solver; the
   code uses `SUNContext`, so SUNDIALS >= 6). Then export `CVODE_DIR`.
6. **Run directory**: `/scratch/gpfs/$USER/nuc_onezone` (not /home).

## Reproducing the validated tests
From the repo root on stellar-vis1, after `source modules.sh`:
```
python tools/nuclear/gen_net.py small src/chemistry/network/nuclear/small --headers-only
python tools/nuclear/gen_net.py aprox13 src/chemistry/network/nuclear/aprox13 --headers-only
python configure.py --prob=nuc_onezone --chemistry=nuclear_aprox13 --chem_ode_solver=cvode --cvode_path=$CVODE_DIR --eos=isothermal --cxx=icpx
make clean && make -j8
```
(Regenerate the headers only if `networks.py` changes or they're missing. For the small
network, configure with `--chemistry=nuclear_small` and rebuild.)

Run on the vis node directly (each run takes seconds), or inside a job with **`srun -n 1`**:
```
mkdir -p /scratch/gpfs/$USER/nuc_onezone && cd /scratch/gpfs/$USER/nuc_onezone
cp $ATHENA_DIR/athena++_milo/inputs/chemistry/athinput.nuc_onezone .
rm -f aprox13.hst; $ATHENA_DIR/athena++_milo/bin/athena -i athinput.nuc_onezone chemistry/reltol=1e-8 chemistry/temperature=4e9 job/problem_id=aprox13
python $ATHENA_DIR/athena++_milo/tools/nuclear/check_onezone.py aprox13.hst --net aprox13 --rho 1e7 --T 4e9 --X C12=0.5 O16=0.5 --plot aprox13.png
```
Expected: max|dX| <= ~4e-8 per species, |sum(X)-1| ~ 1e-15, and no "holds N runs"
warning. Final X: Si28 0.38, S32 0.27, Ni56 0.20, Ar36 0.074, Ca40 0.065. For `small`
(T=3e9, `--net small`), max|dX| ~ 7.5e-7 at reltol=1e-8.

## Pitfalls hit in the first session
- **hst files append**: Athena++ appends to an existing `.hst` file. `rm` it before
  rerunning. `check_onezone.py` keeps only the last monotonic run and warns otherwise.
- **`srun` without `-n 1`** inside a multi-task allocation starts one athena per task,
  and they all write the same hst file. At handoff, the PI's
  `/scratch/gpfs/za9132/nuc_onezone/aprox13.hst` was corrupted this way ("31156 runs").
  The clean rerun hadn't been confirmed yet.
- **History precision**: the default `data_format` (%12.5e) hides differences below 1e-6.
  `athinput.nuc_onezone` sets `%24.16e`.
- **matplotlib TeX**: a user matplotlibrc with `text.usetex` fails on compute nodes (no TeX
  fonts). `check_onezone.py` turns it off.
- **pynucastro duplicates**: two rates with the same reactants and products raise
  `RateDuplicationError` (that's why the ModifiedRate approach for the C/O p channels was
  dropped).
- **icpx -ipo link warning** "loop not distributed ... LLVM gold plugin" is harmless.

## How the pieces fit
- `tools/nuclear/networks.py`: **single definition** of each network (`small`, `aprox13`),
  as pynucastro builders that take the network class. Both `gen_net.py` (writes C++
  `SimpleCxxNetwork` headers) and `check_onezone.py` (Python reference + scipy BDF) use it,
  so the C++ network and its reference can't drift apart.
- `src/chemistry/network/nuclear/<name>/`: generated, header-only C++20 (rates, screening,
  Jacobian). `pynucastro-info.txt` there records the pynucastro version.
- `src/chemistry/network/nuclear.{hpp,cpp}`: the Athena++ `ChemNetwork` wrapper for all
  nuclear networks. Only `nuclear.cpp` includes the generated headers. It converts mass
  fractions X (Athena++'s `s/rho`) to molar Y=X/A for pynucastro and back
  (Xdot = A*Ydot, J^X_ij = A_i/A_j J^Y_ij). It's at fixed T (`<chemistry> temperature`)
  and aborts unless `--eos=isothermal`.
- `configure.py`: `--chemistry=nuclear_<name>` selects the wrapper, sets NSPECIES (dict
  `nuclear_nspecies`) and adds `-I` for the generated headers. All compilers use
  `-std=c++20` (was c++11). Also fixed a missing space before CVODE's `-I`.
- `src/pgen/nuc_onezone.cpp` + `inputs/chemistry/athinput.nuc_onezone`: static uniform gas,
  cgs code units (`unit_system=custom`). The user timestep dt = max(dt_init, dt_fac*t)
  gives log-spaced cycles, and a tiny `iso_sound_speed` keeps hydro from limiting dt.

## Files created/changed in the first session (commit these)
- New: `docs/nuclear-network-plan.md` (sections added), this file,
  `src/chemistry/network/nuclear.{hpp,cpp}`, `src/chemistry/network/nuclear/{small,aprox13}/`
  (generated; commit them so the build doesn't require pynucastro),
  `src/pgen/nuc_onezone.cpp`, `inputs/chemistry/athinput.nuc_onezone`,
  `tools/nuclear/{networks,gen_net,check_onezone}.py`, `modules.sh`.
- Changed: `configure.py`, `CLAUDE.md`, `config.sh` (`--cxx=icpx`; still builds the H2 test).
- Obsolete; delete if still present: `src/chemistry/network/nuclear_small{,.hpp,.cpp}`,
  `tools/nuclear/gen_net_small.py`, `tools/nuclear/net_small/` (scratch build of the
  standalone driver).

## Next steps (in the order the PI agreed on)
1. Confirm a clean `aprox13.hst` (one run, no warning).
2. **Benchmark vs Timmes' aprox13** (public Fortran, cococubed.com): fixed-T one-zone
   burns at rho=1e7 and T = 3, 4, 5e9 from 50/50 C/O. Expect ~10-30% differences in
   timescales and final Si-group/Ni56 fractions (REACLIB vs mostly-CF88 rates, different
   C/O branching). Download on a login/vis node. The PI hadn't yet approved fetching it.
3. **Energy feedback**: evolve internal energy with X; T from an EOS (ideal ions +
   radiation first, then degenerate electrons, e.g. Helmholtz); `Edot` from eps_nuc;
   non-isothermal Jacobian incl. dT terms (pynucastro can generate T derivatives). This is
   the largest open piece (see the plan doc's reuse/change table).
4. Later: NSE above ~5-6e9 K, renormalizing sum(X)=1 after advection, CVODE failure
   handling (`CheckFlag` throws), performance (`WrapJacobian` allocates an AthenaArray per
   call; rates are evaluated separately in RHS and Jacobian).
