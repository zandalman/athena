"""Compare an Athena++ one-zone burn (nuc_onezone.hst) with a reference solution.

The reference integrates the same network (networks.py, as a pynucastro PythonNetwork
with Chugunov 2007 screening, matching the C++ network) with scipy's BDF solver at the
same fixed density and temperature.

usage: python check_onezone.py HSTFILE --net aprox13 --rho 1e7 --T 3e9
                               --X C12=0.5 O16=0.5 [--plot out.png]
"""

import argparse
import importlib.util
import os
import re
import sys
import tempfile

import numpy as np
from scipy.integrate import solve_ivp

import pynucastro as pyna
from pynucastro.screening import chugunov_2007

from networks import NETWORKS

N_HYDRO_COLS = 9   # time, dt, mass, 3 momenta, 3 KE (isothermal, no B)


def load_hst(fname):
    """Return times, species names (from the header) and mass fractions."""
    with open(fname) as f:
        header = [line for line in f if line.startswith("#")][-1]
    cols = re.findall(r"\[\d+\]=(\S+)", header)
    species = cols[N_HYDRO_COLS:]
    data = np.loadtxt(fname, comments="#")
    # Athena++ appends to an existing hst file; keep only the last run
    restarts = np.nonzero(np.diff(data[:, 0]) <= 0)[0]
    if restarts.size:
        print(f"warning: {fname} holds {restarts.size + 1} runs; using the last one")
        data = data[restarts[-1] + 1:]
    t = data[:, 0]
    mass = data[:, 2]
    X = data[:, N_HYDRO_COLS:N_HYDRO_COLS + len(species)] / mass[:, None]
    return t, species, X


def build_reference_net(name, odir):
    net = NETWORKS[name](pyna.PythonNetwork)
    fname = os.path.join(odir, f"net_{name}_ref.py")
    net.write_network(fname)
    spec = importlib.util.spec_from_file_location(f"net_{name}_ref", fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return net, mod


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hst")
    parser.add_argument("--net", choices=sorted(NETWORKS), required=True)
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--T", type=float, required=True)
    parser.add_argument("--X", nargs="+", required=True, help="initial X, e.g. C12=0.5")
    parser.add_argument("--plot", default=None)
    args = parser.parse_args()

    t_ath, species, X_ath = load_hst(args.hst)

    with tempfile.TemporaryDirectory() as odir:
        net, mod = build_reference_net(args.net, odir)

    # map the reference network's nuclei onto the Athena++ species order
    A = np.array([n.A for n in net.unique_nuclei])
    names = [n.short_spec_name.lower() for n in net.unique_nuclei]
    if sorted(names) != sorted(s.lower() for s in species):
        sys.exit(f"species mismatch: hst has {species}, reference has {names}")
    order = [names.index(s.lower()) for s in species]

    X0 = np.zeros(len(species))
    for item in args.X:
        name, val = item.split("=")
        X0[species.index(name)] = float(val)
    Y0 = np.zeros(len(names))
    for i, j in enumerate(order):
        Y0[j] = X0[i] / A[j]

    def f(t, Y):
        return mod.rhs(t, Y, args.rho, args.T, screen_func=chugunov_2007)

    def jac(t, Y):
        return mod.jacobian(t, Y, args.rho, args.T, screen_func=chugunov_2007)

    t_eval = t_ath[t_ath > 0]
    sol = solve_ivp(f, (0., t_eval[-1]), Y0, method="BDF", jac=jac,
                    t_eval=t_eval, rtol=1e-10, atol=1e-16)
    if not sol.success:
        sys.exit(f"reference integration failed: {sol.message}")
    X_ref = (sol.y.T * A)[:, order]
    X_cmp = X_ath[t_ath > 0]

    print(f"{'species':>8s} {'X_final(athena)':>16s} {'X_final(ref)':>14s} "
          f"{'max|dX|':>10s}")
    for i, s in enumerate(species):
        print(f"{s:>8s} {X_cmp[-1, i]:16.8e} {X_ref[-1, i]:14.8e} "
              f"{np.max(np.abs(X_cmp[:, i] - X_ref[:, i])):10.2e}")
    print(f"max |sum(X) - 1| (athena) = {np.max(np.abs(X_ath.sum(axis=1) - 1)):.2e}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        # a user matplotlibrc may enable TeX, whose fonts are missing on compute nodes
        matplotlib.rcParams["text.usetex"] = False
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i, s in enumerate(species):
            line, = ax.loglog(t_eval, np.maximum(X_cmp[:, i], 1e-30), label=s)
            ax.loglog(t_eval, np.maximum(X_ref[:, i], 1e-30), ls="--",
                      color=line.get_color())
        ax.set_xlabel("t [s]")
        ax.set_ylabel("X")
        ax.set_ylim(1e-12, 2)
        ax.set_title(f"{args.net}: rho={args.rho:g}, T={args.T:g} "
                     "(solid: Athena++, dashed: ref)")
        ax.legend(fontsize="small", ncol=2)
        fig.savefig(args.plot, dpi=150)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
