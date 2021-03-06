#!/bin/bash
echo "============================================================="
echo "additional setup to run MPI under notebooks"
echo "============================================================="

export OMPI_MCA_rmaps_base_oversubscribe=1

jupyter serverextension enable --py ipyparallel

CFG_FILE="$HOME/.ipython/profile_mpi/ipcluster_config.py"
if [[ -f "$CFG_FILE" ]]; then
  echo "using existing ipcluster configuration: $CFG_FILE"
else
  echo "creating ipcluster configuration: $CFG_FILE"
  ipython profile create --parallel --profile=mpi
  echo "c.IPClusterEngines.engine_launcher_class = 'MPIEngineSetLauncher'" >> $CFG_FILE
  echo "c.IPClusterStart.controller_launcher_class = 'MPIControllerLauncher'" >> $CFG_FILE
fi

ipcluster start -n 4 --profile=mpi &
