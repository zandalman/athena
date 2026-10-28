# Nuclear network plan

How the existing chemistry module works is summarized in the Architecture section of
CLAUDE.md.

## Reuse vs. change for the nuclear network

Target: mass fractions, cgs rates, much stiffer reactions.

| Piece | Verdict | Notes |
|---|---|---|
| Configure plumbing (`--chemistry`, `CHEMNETWORK_HEADER`, `--chem_ode_solver`, `--cvode_path`) | Reuse, small edit | Add an entry with the network's NSPECIES in `configure.py` (~L718) |
| `NetworkWrapper` interface (`InitializeNextStep`/`RHS`/`Edot`/`Jacobian`) | Reuse | New `ChemNetwork` in `src/chemistry/network/<net>.{hpp,cpp}` |
| Storage: passive scalars, `y = s/rho` | Reuse (better fit) | `s/rho` is already a mass fraction X_i. Drop the n_H interpretation and convert to Y_i = X_i/A_i inside RHS if needed |
| Floor + writeback (`cvode.cpp` after `CVode`) | Change | The floor breaks sum(X)=1, so renormalize after the solve. Advection doesn't conserve the sum either |
| Units contract (code units in/out, cgs inside) | Reuse the pattern | Convert density explicitly (`w(IDN)*code_density_cgs`). Rates are in s^-1, so multiply by `code_time_cgs`. Edot: eps_dot [erg/g/s] * rho_cgs / `code_energydensity_cgs` * `code_time_cgs`. Use a custom unit system |
| Energy feedback (internal-energy slot, rebuild `u(IEN)`) | Reuse as is | Only the physics in Edot changes |
| T from E inside the network (H2 uses `Thermo::CvCold`) | Change (largest) | Needs a degenerate/radiation EOS that depends on abar and zbar, consistent with the hydro EOS. `src/eos/general/*` signatures take only (rho, e), not composition |
| Task placement (`INT_CHM`, last stage, operator split) | Reuse initially | First-order split. Possibly limit dt via a user timestep, or use Strang splitting later |
| CVODE parameters (`ODEWrapper::Initialize`) | Change | `maxorder` 2->5, `reltol` 1e-2->~1e-6, raise `maxsteps`, set mass-fraction `abstol` (~1e-12) and a scaled `abstol_E`. An analytic Jacobian (`user_jac=true`) is effectively required |
| Linear solver (dense) | Keep for small networks | Fine for aprox13/21-sized networks. For ~100+ species, switch to sparse (SUNSparseMatrix + KLU) |
| `NetworkWrapper::WrapJacobian` | Change (performance) | Allocates an AthenaArray on every call. Fill the SUNMatrix directly |
| Per-cell loop, `CVodeReInit`, step-size `h` reuse | Reuse | |
| `CheckFlag` (throws on any CVode failure) | Change, probably | Retry or sub-step instead of aborting |
| `forward_euler.cpp` | Not usable | Too stiff for nuclear; only a structural reference |
| Outputs/restart (`species_names`, `h`) | Reuse | Provide `species_names` |

## Decided: reaction rate source
- Rates come from JINA REACLIB (Cyburt+ 2010). Network code (RHS + analytic Jacobian) is
  generated with pynucastro, including approximate (a,p)(p,g) alpha-chain rates, reverse
  rates/partition functions, and screening.
- Benchmark against Timmes aprox13 in one-zone burns (the standard network in the WD TDE
  literature).
- Rejected: STARLIB (maybe later for uncertainty studies), CF88 (outdated), NACRE II (A <~ 16 only).
- Record which fits are used for the most uncertain key rates (12C+12C, 12C(a,g)16O).
- Plan: start with a tiny network (He4, C12, O16, Ne20, Mg24: triple-alpha, 12C(a,g),
  12C+12C, a few alpha captures, screening) to exercise the Athena++ pipeline, then move to a
  13-isotope approximate chain and compare with aprox13.
- Code generator: pynucastro `SimpleCxxNetwork`. It is header-only and needs only the standard
  library (`amrex_bridge.H` reimplements Real/Array1D/Array2D/constexpr_for; no AMReX install).
  It provides `actual_rhs.H` (dY/dt, weak-rate neutrino losses, Jacobian), REACLIB/approximate/
  derived rates, partition functions, and Chugunov+ 2007 screening (`-DSCREENING`). It has no
  integrator, NSE solver, or thermal neutrino losses.
- It requires C++20. Decision: build all of Athena++ with `-std=c++20` (`configure.py` now
  uses it for every compiler; it was `-std=c++11`). Fallback if that causes trouble: one
  C++20 translation unit for the network behind a C++11-compatible interface.

## Small test network (status)
- Generator: `tools/nuclear/gen_net.py small OUTDIR` (REACLIB `linking_nuclei` on
  He4/C12/O16/Ne20/Mg24, 12 rates incl. REACLIB reverse fits). Its test driver compiles with
  `icpx -std=c++20 -O2 -DSCREENING -I. *.cpp` on a login/vis node; `icpx` isn't usable on
  compute nodes (the module's oneAPI files are missing there).
- Checked at rho=1e7, T=3e9: sum(A_i Ydot_i)=0 and the Jacobian's A-weighted column sums are
  0 to round-off. eps_nuc = N_A sum(B_i Ydot_i) = 1.418e24 erg/g/s agrees with the code.
  Stiff: |J(Ne20,Ne20)| ~ 1.5e7 s^-1.
- Known gaps: C12+C12 has only the alpha branch (the p/n branches are dropped because Na23/
  Mg23 aren't in the set, so the total rate is low by up to ~2x; fix with a pynucastro ModifiedRate later). No
  O16+O16, no (a,p)(p,g). Reverse rates are REACLIB fits with pf=1, not DerivedRate; that
  matters near NSE.

## One-zone test at fixed temperature
- Networks `--chemistry=nuclear_<name>` (`small`, `aprox13`) share one wrapper,
  `src/chemistry/network/nuclear.{hpp,cpp}`. Each network is defined once in
  `tools/nuclear/networks.py`, which `gen_net.py` (C++ output) and `check_onezone.py`
  (Python reference) both use. The generated headers live in
  `src/chemistry/network/nuclear/<name>/` (`gen_net.py <name> <dir> --headers-only`), and
  configure adds `-I` for them and sets NSPECIES. Only `nuclear.cpp` includes them, which
  keeps the pynucastro global names out of the rest of Athena++. `species_names` is
  copied from the generated `spec_names`.
- ODE variable X_i: Xdot_i = A_i Ydot_i, J^X_ij = (A_i/A_j) J^Y_ij, times `code_time_cgs`.
  T is fixed by `<chemistry> temperature`. It requires `--eos=isothermal` (the
  constructor aborts otherwise). Screening is always on.
- Problem `nuc_onezone` (`src/pgen/nuc_onezone.cpp`, `inputs/chemistry/athinput.nuc_onezone`):
  static uniform gas in cgs code units (`unit_system=custom`). The user timestep is
  dt = max(dt_init, dt_fac*t) (log-spaced), with a tiny `iso_sound_speed` so hydro never
  limits dt. hst output every cycle; X_i = species column / mass.
- Reference: `tools/nuclear/check_onezone.py` integrates the same REACLIB network
  (pynucastro PythonNetwork + Chugunov 2007 screening) with scipy BDF and compares.
- Result (rho=1e7, T=3e9, X_C12=X_O16=0.5, t=10 s, ~600 cycles; ends at X_Mg24~0.71,
  X_O16~0.29): |sum(X)-1| ~ 3e-15. max|dX| vs reference is 3.1e-5 at reltol=1e-6 and
  7.5e-7 at reltol=1e-8 (~40x smaller). The error is set by the solver tolerance and adds
  up over the per-cycle CVODE restarts. reltol=1e-8 costs ~1.45x the CPU time.
  The hst needs `data_format=%24.16e`; the default 6 digits hides these differences.

## aprox13-like network (`nuclear_aprox13`)
- Nuclei: He4 through Ni56 (13). It links every REACLIB rate among them, p, and the
  intermediates Na23, Al27, P31 (C/O channels) and Cl35, K39, Sc43, V47, Mn51, Co55
  ((a,p)(p,g)).
- `make_CO_burning_approx("C"/"CO"/"O")` (pynucastro's aprox-family helper) puts Na23, Al27
  and P31 in equilibrium. It combines each heavy-ion p channel with (p,a) into the alpha
  channel and with (p,g) into a direct S+S -> F rate, and builds the Ne20, Mg24 and Si28
  (a,p)(p,g) links. `make_ap_pg_approx` handles S32 through Fe52. Then the intermediates and
  p are removed. Reverse rates are kept, including the heavy-ion ones (aprox13 drops them).
  C12+Ne20 rates are kept.
- Rejected: redirecting the p/n channels with ModifiedRate. It gives the same
  reactants/products as the native alpha channel, which pynucastro rejects as
  `RateDuplicationError`. The n channels (products outside the network) are omitted, as in
  aprox13.
- Step (1) passed (rho=1e7, T=4e9, X_C12=X_O16=0.5, t=10 s, reltol=1e-8): max|dX| <= 4e-8
  for all 13 species and |sum(X)-1| ~ 1e-15. Final composition: Si28 0.38, S32 0.27,
  Ni56 0.20, Ar36 0.074, Ca40 0.065, Fe52 0.014. That run was ~10x the CPU time of
  `small`. Run one-zone tests with `srun -n 1`; with more tasks every rank writes the
  same hst file.
- Validation plan: (1) Athena++ vs the Python reference of the same network (exactness);
  (2) vs Timmes aprox13 one-zone at fixed T (physics; rates differ, CF88 vs REACLIB).

## Open decisions (undecided)
- NSE treatment above ~5-6e9 K (needed for deep disruptions; alpha chain becomes inaccurate/stiff).
- EOS coupling: which EOS to use, and how composition reaches both the network and the hydro EOS.
- Dense vs. sparse linear solver (depends on network size).
- Splitting scheme: keep the first-order split, or use Strang splitting / a burning timestep limiter.
- How to enforce sum(X)=1: renormalize after burning, after advection, or both.
