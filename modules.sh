# environment for building and running; use: source modules.sh
module purge
module load anaconda3/2024.6
module load intel/2021.1.2
module load intel-mpi/intel/2021.1.1
module load hdf5/intel-2021.1/intel-mpi/1.10.6
module load gcc-toolset/13  # newer libstdc++ for icpx -std=c++20 (system gcc 8.5 is too old)
conda activate nucnet
