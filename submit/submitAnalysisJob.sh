#!/bin/bash

# shell script to submit analysis job to the batch system

# source key4ehp
source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2024-04-12

# get input parameters 
FROM_I=${1} # Wwhich index to start with regarding the input root files
NUM_FILES=${2} # number of files to process
FILE_PATTERN=${3}
OUTPUT_DIR=${4} # output directory
OUTPUT_FILE=${5} # output file

# get input files
root_files=($(ls ${FILE_PATTERN} 2>/dev/null | sort | tail -n +$((FROM_I + 1)) | head -n ${NUM_FILES}))

# Check if there are any matching files
if [ ${#root_files[@]} -eq 0 ]; then
    echo "No root files found matching the pattern."
    exit 1
fi


# Copy each file using the Python script
for file in "${root_files[@]}"; do
    echo "Copying file ${file} to ${destination_dir}..."
    python3 /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py "${file}" "${destination_dir}/$(basename ${file})"
done

# merge 1000 root files to one input file
hadd merged_input.root ins.root
python create_jet_based_tree.py merged_input.root out.root

# make directory if it does not exist:
if [ ! -d ${OUTPUT_DIR} ]; then
  mkdir -p ${OUTPUT_DIR}
fi

# copy file to output dir
python /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py out.root ${OUTPUTFILE}
echo "copying output ... "














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
python create_fullsim_tagger_data.py ${INPUT_FILE_PATTERN} ${OUTPUT_DIR} ${OUTPUT_FILE} ${START_IND} ${END_IND}
echo "job done ... "
python /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py out.root ${OUTPUTFILE}
echo "copying output ... "

