def generate_analysis_sub():
    # Define file patterns and other fixed values
    file_patterns = ["Huu"]
    file_dirs = {"Huu": "00016808", "Hcc": "00016810"} 
    num_files = 50
    start_indices = range(0, 10000, num_files)
    base_command = "/eos/experiment/fcc/prod/fcc/ee/test_spring2024/240gev/{pattern}/CLD_o2_v05/rec/*/*/{pattern}_rec_*.root"
    #base_command = "/eos/experiment/fcc/prod/fcc/ee/test_spring2024/240gev/{pattern}/CLD_o2_v05/rec/{file_dir}/*/{pattern}_rec_*.root" # new data with fixed PV, but does not work

    #output_base = "/eos/experiment/fcc/ee/datasets/CLD_fullsim_tagging_debug_tracks/"
    output_base = "/eos/experiment/fcc/ee/datasets/CLD_fullsim_tagging_debug_tracks/UsingMCPV/"
    #output_base = "/eos/experiment/fcc/ee/datasets/CLD_fullsim_tagging_debug_tracks/with_fixesPV/"
    
    # Prepare the header of the file
    header = """# run commands for analysis,

# here goes your shell script
executable    = submitAnalysisJob.sh
#requirements = (OpSysAndVer =?= "CentOS7")
# here you specify where to put .log, .out and .err files
output                = /afs/cern.ch/work/s/saaumill/public/condor/std-condor/job.$(ClusterId).$(ProcId).out
error                 = /afs/cern.ch/work/s/saaumill/public/condor/std-condor/job.$(ClusterId).$(ProcId).err
log                   = /afs/cern.ch/work/s/saaumill/public/condor/std-condor/job.$(ClusterId).$(ClusterId).log

+AccountingGroup = "group_u_FCC.local_gen"
+JobFlavour    = "microcentury"
"""

    # Prepare the content with arguments
    content = ""
    for pattern in file_patterns:
        job_counter = 0
        for start_index in start_indices:
            input_pattern = base_command.format(pattern=pattern, file_dir=file_dirs[pattern])
            output_file = f"{output_base}{pattern}_{job_counter}.root"
            arguments = f"{start_index} {num_files} \'{input_pattern}\' {output_base} {output_file}"
            content += f"arguments=\"{arguments}\"\nqueue\n"
            job_counter += 1

    # Write to the analysis.sub file
    with open("analysis.sub", "w") as file:
        file.write(header)
        file.write(content)

# Run the function to generate the file
generate_analysis_sub()
