import os
import subprocess
import sys
import glob
import time

def run_create_jet_based_tree(file_pattern, output_folder, output_root_filename, from_i=0, to_j=-1):
    # Ensure output folder exists

    os.makedirs(output_folder, exist_ok=True)
    
    # Find all root files matching the pattern
    root_files = glob.glob(file_pattern)
    root_files.sort()  # Ensure consistent ordering
    
    if not root_files:
        print("No root files found matching the pattern.")
        return
    
    # Validate index range
    if from_i < 0 or to_j > len(root_files): #or from_i >= to_j:
        print("Invalid index range.")
        return
    
    # Select the subset of files to process
    root_files_subset = root_files[from_i:to_j]
    
    # Create a log file to store the output of the subprocess
    log_file_path = os.path.join(output_folder, f"process_log_{from_i}_{to_j}.txt")
    with open(log_file_path, "w") as log_file:

        # Process each root file
        output_files = []
        print(f"Processing {len(root_files_subset)} root files from index {from_i} to {to_j}...")
        for i, root_file in enumerate(root_files_subset):
            output_file_path = os.path.join(output_folder, f"tagger_input_data_fullsim_{os.path.basename(root_file)}")
            # Run the create_jet_based_tree.py script
            subprocess.run(
                ['python', 'create_jet_based_tree.py', root_file, output_file_path],
                stdout=log_file,
                stderr=log_file
            )
            output_files.append(output_file_path)
            if (i + 1) % 1000 == 0:
                print(f"Done with {i + 1} events")
    
    # Combine all output root files into one big root file
    combined_output_filename = f"{os.path.splitext(output_root_filename)[0]}_{from_i}_{to_j}.root"
    combined_output_path = os.path.join(output_folder, combined_output_filename)
    try:
        hadd_command = ['hadd', combined_output_path] + output_files
        subprocess.run(hadd_command)
        print(f"Combined root file created at: {combined_output_path}")
        print(f"Log file created at: {log_file_path}")
    except OSError as e:
        print(f"Error combining root files: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 6:
        print("Usage: python script.py <file_pattern> <output_folder> <output_root_filename> <from_i> <to_j>")
        # python create_fullsim_tagger_data.py "/eos/experiment/fcc/prod/fcc/ee/test_spring2024/240gev/H*/CLD_o2_v05/rec/*/*/H*_rec_*.root" /afs/cern.ch/work/s/saaumill/public/Hxx-inputNN-smalldata_new /afs/cern.ch/work/s/saaumill/public/final-fullsim-inputNN/Hxx-inputNN-smalldata_new.root 0 1000
        sys.exit(1)
    
    file_pattern = sys.argv[1]
    output_folder = sys.argv[2]
    output_root_filename = sys.argv[3]
    # Set default values for from_i and to_j
    from_i = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    to_j = int(sys.argv[5]) if len(sys.argv) > 5 else -1
    
    a = time.time()
    run_create_jet_based_tree(file_pattern, output_folder, output_root_filename, from_i, to_j)
    b = time.time()
    print(f"Time taken: {(b - a)/60:.2f} minutes")
