import sys
import math

import ROOT
from array import array

from ROOT import TFile, TTree
import numpy as np
from podio import root_io
import edm4hep
import re

from tree_tools import initialize, clear_dic, store_jet

## debug is used to work with only 2 events and add some prints
debug = False

def extract_hxx(filename):
    # Use regular expression to find "H" followed by either "tautau" or exactly two lowercase letters (Huu, Hdd, etc.)
    match = re.search(r'H(?:tautau|[a-z]{2})', filename)
    if match:
        return match.group(0) 
    return None

## Input args are the file to read from and the file to write on
input_file = sys.argv[1]
output_file = sys.argv[2]


H_to_xx = extract_hxx(input_file)

CLIC = "False"
reader = root_io.Reader(input_file)
out_root = TFile(output_file, "RECREATE")
t = TTree("tree", "pf tree lar")
event_number, n_hit, n_part, dic, t = initialize(t)

event_number[0] = 0
print("Processing ", len(reader.get("events")), " events...")
for i, event in enumerate(reader.get("events")):

    if debug:
        if i > 10:
            break
    # clear all the vectors
    dic = clear_dic(dic)

    #print("")
    #print(" ----- new event: {} ----------".format(event_number[0]))
    #print("")

    dic, event_number, t = store_jet(
        event,
        debug,
        dic,
        event_number, 
        t, 
        H_to_xx
    )


t.SetDirectory(out_root)
t.Write()
