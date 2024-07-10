#!/bin/bash

# shell script to submit analysis job to the batch system

start_time=$(date +%s)

# source key4ehp
source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2024-04-12

# get input parameters 
FROM_I=${1} # Wwhich index to start with regarding the input root files
NUM_FILES=${2} # number of files to process
FILE_PATTERN=${3}
echo "file pattern ${FILE_PATTERN}"
OUTPUT_DIR=${4} # output directory
echo "output dir ${OUTPUT_DIR}"
OUTPUT_FILE=${5} # output file
echo "output file ${OUTPUT_FILE}"
echo "output dir ${OUTPUT_DIR}"

# get input files
root_files=($(ls ${FILE_PATTERN} 2>/dev/null | sort | tail -n +$((FROM_I + 1)) | head -n ${NUM_FILES})) # works fine even if NUM_FILES is larger than the number of files available 

# Check if there are any matching files
if [ ${#root_files[@]} -eq 0 ]; then
    echo "No root files found matching the pattern."
    exit 1
fi


# Copy each file using the Python script
for file in "${root_files[@]}"; do
    python3 /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py "${file}" "./$(basename ${file})"
done

# merge 1000 root files to one input file
hadd merged_input.root ./*.root
preproc_time=$(date +%s)
middle_time=$((preproc_time - start_time))
echo "time before running script: $middle_time seconds" # 13 sec -> x100 = 21 min
python /afs/cern.ch/work/s/saaumill/public/FullSimTagger/src/create_jet_based_tree.py merged_input.root out.root
echo "job done ... "
job_endtime=$(date +%s)
job_time=$((job_endtime - preproc_time)) # 54 sec for 1000 files -> index 0 to 10. So for 0 to 1000 it should be 1.5h 
echo "time to run job: $job_time seconds" # 
# make directory if it does not exist:
if [ ! -d ${OUTPUT_DIR} ]; then
  mkdir -p ${OUTPUT_DIR}
fi

# copy file to output dir
python /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py out.root ${OUTPUT_FILE}
echo "Ran script successfully!"
end_time=$(date +%s)
execution_time=$((end_time - start_time)) # rouhgly 2h for index 0 to 1000
echo "Execution time: $execution_time seconds"

