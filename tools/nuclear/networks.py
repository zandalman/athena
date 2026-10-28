"""Nuclear network definitions built with pynucastro from JINA REACLIB.

Each builder takes a pynucastro network class (SimpleCxxNetwork for Athena++,
PythonNetwork for the reference solution) and returns a network instance, so the
C++ network and its reference are built identically.
"""

import pynucastro as pyna

ALPHA_CHAIN = ["he4", "c12", "o16", "ne20", "mg24", "si28", "s32",
               "ar36", "ca40", "ti44", "cr48", "fe52", "ni56"]

# intermediates of the C12+C12, C12+O16, O16+O16 proton channels; make_CO_burning_approx
# puts them in equilibrium (this also builds the Ne20, Mg24, Si28 (a,p)(p,g) links)
CO_INTERMEDIATES = {"C": "na23", "CO": "al27", "O": "p31"}

# intermediates of the remaining A(a,p)X(p,g)B links, S32 -> Ni56
AP_PG_INTERMEDIATES = ["cl35", "k39", "sc43", "v47", "mn51", "co55"]


def small(net_class):
    """He4, C12, O16, Ne20, Mg24: every REACLIB rate linking them."""
    rl = pyna.ReacLibLibrary()
    lib = rl.linking_nuclei(["he4", "c12", "o16", "ne20", "mg24"])
    return net_class(libraries=[lib])


def aprox13(net_class):
    """13-isotope alpha chain He4..Ni56 with aprox-family approximations (aprox13-like).

    Every REACLIB rate linking the alpha chain, p and the intermediates is included. Then:
    - C+C, C+O, O+O: make_CO_burning_approx (proton channel with the intermediate in
      equilibrium);
    - S32..Fe52 alpha captures: make_ap_pg_approx;
    - the intermediates and p are removed.
    Neutron channels (c12(c12,n)mg23, o16(c12,n)si27, o16(o16,n)s31) are not included,
    since their products are outside the network (as in aprox13).
    """
    rl = pyna.ReacLibLibrary()
    intermediates = list(CO_INTERMEDIATES.values()) + AP_PG_INTERMEDIATES
    lib = rl.linking_nuclei(["p"] + ALPHA_CHAIN + intermediates)

    net = net_class(libraries=[lib])
    for root in CO_INTERMEDIATES:
        net.make_CO_burning_approx(root_nuclei=root)
    net.make_ap_pg_approx(intermediate_nuclei=AP_PG_INTERMEDIATES)
    net.remove_nuclei(intermediates)
    if pyna.Nucleus("p") in net.unique_nuclei:
        net.remove_nuclei(["p"])
    return net


NETWORKS = {"small": small, "aprox13": aprox13}
