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

## Open decisions (undecided)
- EOS coupling: which EOS to use, and how composition reaches both the network and the hydro EOS.
- Dense vs. sparse linear solver (depends on network size).
- Splitting scheme: keep the first-order split, or use Strang splitting / a burning timestep limiter.
- How to enforce sum(X)=1: renormalize after burning, after advection, or both.
