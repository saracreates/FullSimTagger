# Instructions

This folder is meant to provide a submit routine for running the `create_jet_based_tree.py` from `src` over many many events by using condor. Check out there [webpage](https://batchdocs.web.cern.ch/local/quick.html) for more information. 

Submitting the jobs looks like this: `condor_submit analysis.sub` that calls `submitAnalysisJob.sh` for different files.
`write_analysis.py` generates the `submitAnalysisJob.sh` file.

The rest of the files is from testing & debugging etc ... 

In the end, you'll end up with many root files that are the **input** for FullSim flavor tagging for CLD with the ParticleNet. 