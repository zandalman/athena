"""Generate a nuclear network (see networks.py) as pynucastro SimpleCxxNetwork C++20 code.

usage: python gen_net.py NAME OUTDIR [--headers-only]

For Athena++ (--chemistry=nuclear_NAME), write the headers into
src/chemistry/network/nuclear/NAME/ with --headers-only, which removes the standalone
test driver (main.cpp, GNUmakefile).
"""

import argparse
import os

import pynucastro as pyna

from networks import NETWORKS

DRIVER_FILES = ["main.cpp", "GNUmakefile"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", choices=sorted(NETWORKS))
    parser.add_argument("outdir")
    parser.add_argument("--headers-only", action="store_true")
    args = parser.parse_args()

    odir = os.path.abspath(args.outdir)
    os.makedirs(odir, exist_ok=True)

    net = NETWORKS[args.name](pyna.SimpleCxxNetwork)
    print(net.network_overview())
    print(f"{len(net.unique_nuclei)} nuclei, {len(net.get_rates())} rates")

    # write_network writes into the current directory
    os.chdir(odir)
    net.write_network()
    if args.headers_only:
        for f in DRIVER_FILES:
            if os.path.exists(f):
                os.remove(f)
    print(f"wrote network to {odir}")


if __name__ == "__main__":
    main()
