#!/bin/bash

# shell script to submit analysis job to the batch system

#source /cvmfs/sw.hsf.org/key4hep/setup.sh
LOCAL_DIR=${1}
SCRIPT=${2}
INPUTFILE=${3}
OUTPUTFILE=${4}
NEVENTS=${5}

echo "LOCAL_DIR dir: ${LOCAL_DIR}"

cd ${LOCAL_DIR}
#source setup.sh
#source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2024-03-10
source setup.sh
export PYTHONPATH=${LOCAL_DIR}:$PYTHONPATH
#export LD_LIBRARY_PATH=$LOCAL_DIR/install/lib:$LD_LIBRARY_PATH
#export ROOT_INCLUDE_PATH=$LOCAL_DIR/install/include/FCCAnalyses:$ROOT_INCLUDE_PATH
#export EOS_MGM_URL="root://eospublic.cern.ch"

cd -
mkdir job
cd job

#echo $PYTHONPATH
echo $LD_LIBRARY_PATH

echo "copying file here ... "
python /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py ${INPUTFILE} in.root
cp ${SCRIPT} script.py

echo $PYTHONPATH

BASEOUTDIR=$(dirname -- "$OUTPUTFILE")
mkdir -p ${BASEOUTDIR}
echo "output dir:  ${BASEOUTDIR}"
echo "output file:  ${OUTPUTFILE}"

echo "running script ... "
which fccanalysis
echo ${SCRIPT}
fccanalysis run ${SCRIPT} --output out.root --files-list ${PWD}/in.root --nevents ${NEVENTS}
#fccanalysis run script.py --output out.root --files-list in.root --nevents ${NEVENTS}
echo "job done ... "
python /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py out.root ${OUTPUTFILE}
echo "copying output ... "
                                                                                                                                            45,1          Bot

