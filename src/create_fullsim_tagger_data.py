import os
import subprocess
import sys
import glob

def run_create_jet_based_tree(file_pattern, output_folder, output_root_filename):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all root files matching the pattern
    root_files = glob.glob(file_pattern)
    
    if not root_files:
        print("No root files found matching the pattern.")
        return
    # Create a log file to store the output of the subprocess
    log_file_path = os.path.join(output_folder, "process_log.txt")
    with open(log_file_path, "w") as log_file:

        # Process each root file
        output_files = []
        print(f"Processing {len(root_files)} root files...")
        for i, root_file in enumerate(root_files):
            output_file_path = os.path.join(output_folder, f"tagger_input_data_fullsim_{os.path.basename(root_file)}")
            # Run the create_jet_based_tree.py script
            subprocess.run(
                ['python', 'create_jet_based_tree.py', root_file, output_file_path],
                stdout=log_file,
                stderr=log_file
            )
            output_files.append(output_file_path)
            if (i + 1) % 20 == 0:
                print(f"Done with {i + 1} events")
    
    # Combine all output root files into one big root file
    combined_output_path = os.path.join(output_folder, output_root_filename)
    hadd_command = ['hadd', combined_output_path] + output_files
    subprocess.run(hadd_command)
    
    print(f"Combined root file created at: {combined_output_path}")
    print(f"Log file created at: {log_file_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <file_pattern> <output_folder> <output_root_filename>")
        # python create_fullsim_tagger_data.py "/eos/experiment/fcc/prod/fcc/ee/test_spring2024/240gev/H*/CLD_o2_v05/rec/*/*/H*_rec_*.root" /afs/cern.ch/work/s/saaumill/public/fullsim_smalldataset/ /afs/cern.ch/work/s/saaumill/public/fullsim_smalldata.root
        # python create_fullsim_tagger_data.py "/eos/experiment/fcc/prod/fcc/ee/test_spring2024/240gev/H*/CLD_o2_v05/rec/*/*/H*_rec_*.root" /afs/cern.ch/work/s/saaumill/public/Hxx-inputNN-smalldata_new /afs/cern.ch/work/s/saaumill/public/final-fullsim-inputNN/Hxx-inputNN-smalldata_new.root
        sys.exit(1)
    
    file_pattern = sys.argv[1]
    output_folder = sys.argv[2]
    output_root_filename = sys.argv[3]
    
    run_create_jet_based_tree(file_pattern, output_folder, output_root_filename)
