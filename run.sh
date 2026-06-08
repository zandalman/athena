cp $ATHENA_DIR/athena++_milo/inputs/tde/athinput.tde .
cp $ATHENA_DIR/athena++_milo/bin/athena .
rm tde.*
rm run.log
srun -N 1 -n 96 athena -i athinput.tde
