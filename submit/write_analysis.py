def generate_analysis_sub():
    # Define file patterns and other fixed values
    file_patterns = ["Hbb", "Huu", "Hdd", "Hcc", "Hss", "Hgg", "Htautau"]
    num_files = 50
    start_indices = range(0, 10000, num_files)
    base_command = "/eos/experiment/fcc/prod/fcc/ee/test_spring2024/240gev/{pattern}/CLD_o2_v05/rec/*/*/{pattern}_rec_*.root"
    #output_base = "/afs/cern.ch/work/s/saaumill/public/Hxx-inputNN-largedata_from_batch/"
    output_base = "/eos/experiment/fcc/ee/datasets/CLD_fullsim_tagging_input_artif-track-clusster-matching/"
    
    # Prepare the header of the file
    header = """# run commands for analysis,

# here goes your shell script
executable    = submitAnalysisJob.sh
#requirements = (OpSysAndVer =?= "CentOS7")
# here you specify where to put .log, .out and .err files
output                = /afs/cern.ch/work/s/saaumill/public/std-condor/job.$(ClusterId).$(ProcId).out
error                 = /afs/cern.ch/work/s/saaumill/public/std-condor/job.$(ClusterId).$(ProcId).err
log                   = /afs/cern.ch/work/s/saaumill/public/std-condor/job.$(ClusterId).$(ClusterId).log

+AccountingGroup = "group_u_FCC.local_gen"
+JobFlavour    = "longlunch"
"""

    # Prepare the content with arguments
    content = ""
    for pattern in file_patterns:
        job_counter = 0
        for start_index in start_indices:
            input_pattern = base_command.format(pattern=pattern)
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
