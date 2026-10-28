# H2 chemistry test: chem_uniform, CVODE, adiabatic EOS
# configure and build; run from the repo root after: source modules.sh
python configure.py --prob=chem_uniform --cxx=icpx --chemistry=H2 --chem_ode_solver=cvode --cvode_path=$CVODE_DIR --eos=adiabatic
make clean
make -j8
