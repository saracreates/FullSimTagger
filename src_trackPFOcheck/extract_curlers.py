import ROOT

# List of event numbers to extract
events_to_extract = [
    5, 5, 8, 8, 19, 22, 25, 27, 31, 32, 33, 33, 34, 34, 34, 37,
    39, 43, 45, 45, 46, 48, 50, 52, 53, 53, 55, 56, 61, 64, 66,
    67, 68, 73, 75, 81, 85, 92, 92, 94, 94, 95, 96
]

# Remove duplicates to avoid unnecessary loops
unique_events_to_extract = list(set(events_to_extract))

# Open the input ROOT file
input_file = ROOT.TFile.Open("/afs/cern.ch/work/m/mgarciam/public/Hbb_rec_16562_10.root", "READ")

# Check if file is open
if input_file.IsZombie():
    print("Error: Could not open input file.")
    exit()

# Get the tree from the file
input_tree = input_file.Get("events")  # Replace 'tree' with the actual tree name

# Check if tree is retrieved successfully
if not input_tree or not isinstance(input_tree, ROOT.TTree):
    print("Error: Tree not found in input file.")
    input_file.Close()
    exit()

# Create a new ROOT file to store the extracted events
output_file = ROOT.TFile.Open("extracted_events.root", "RECREATE")
output_tree = input_tree.CloneTree(0)  # Create an empty clone of the tree structure

# Loop over all entries and copy the specified events
for i in range(input_tree.GetEntries()):
    input_tree.GetEntry(i)
    if i+1 in unique_events_to_extract:
        output_tree.Fill()

# Write the new tree to the output file
output_tree.Write()
output_file.Close()
input_file.Close()

print("Extraction completed.")
