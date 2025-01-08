#!/bin/bash

# shell script to submit analysis job to the batch system
echo "Running analysis job ..."
start_time=$(date +%s)

# source key4ehp
source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2024-04-12

# get input parameters 
FROM_I=${1} # Wwhich index to start with regarding the input root files
NUM_FILES=${2} # number of files to process
FILE_PATTERN=${3}
OUTPUT_DIR=${4} # output directory
OUTPUT_FILE=${5} # output file

# get input files
root_files=($(ls ${FILE_PATTERN} 2>/dev/null | sort | tail -n +$((FROM_I + 1)) | head -n ${NUM_FILES})) # works fine even if NUM_FILES is larger than the number of files available 
echo "loading root files ..."

# Check if there are any matching files
if [ ${#root_files[@]} -eq 0 ]; then
    echo "No root files found matching the pattern."
    exit 1
fi

# make directory
mkdir job

# Copy each file using the Python script
for file in "${root_files[@]}"; do
    # Get the basename of the file
    base_name=$(basename "${file}")
    # Copy the file using the Python script
    python3 /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py "${file}" "./job/${base_name}"
done

# Extract the pattern (e.g., Hbb, Huu, etc.)
pattern=$(echo "${FILE_PATTERN}" | grep -oE '/H[[:alnum:]]+/')
pattern=${pattern:1:-1} # Remove the leading and trailing slashes
#echo "file pattern: $pattern"

# merge 1000 root files to one input file
hadd "./job/merged_input_${pattern}.root" ./job/*.root
preproc_time=$(date +%s)
middle_time=$((preproc_time - start_time))
echo "time before running script: $middle_time seconds" # 13 sec -> x100 = 21 min
python /afs/cern.ch/work/s/saaumill/public/FullSimTagger/src_debug_tracks/create_jet_based_tree.py "./job/merged_input_${pattern}.root" ./job/out.root
echo "job done ... "
job_endtime=$(date +%s)
job_time=$((job_endtime - preproc_time)) # 54 sec for 1000 files -> index 0 to 10. So for 0 to 1000 it should be 1.5h 
echo "time to run job: $job_time seconds" # 
# make directory if it does not exist:
if [ ! -d ${OUTPUT_DIR} ]; then
  mkdir -p ${OUTPUT_DIR}
fi

# copy file to output dir
python /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py ./job/out.root ${OUTPUT_FILE}
echo "Ran script successfully!"
end_time=$(date +%s)
execution_time=$((end_time - start_time)) # rouhgly 2h for index 0 to 1000
echo "Execution time: $execution_time seconds"

