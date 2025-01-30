# Jet-Flavor tagging on full simulation

For the first time at FCC-ee, we perform jet-flavor tagging in full simulation. This repository hosts the code for these first studies on full simulation at CLD. Therefore, we

1. Extract the desired variables (such as kinematics, PID, and track parameters) from CLD full simulation
2. Compare these full simulation variables to fast simulation variables at CLD
3. Train the Particle Transformer on this data (using weaver-core, not shown here). We also perform studies on fast simulation of CLD and IDEA. 
4. Evaluate the performance. (Check out [this note](https://repository.cern/records/4pcr6-r0d06) and [this code](https://github.com/saracreates/TaggingResults))

The work is published on the [CERN Document Server](https://repository.cern/records/4pcr6-r0d06) and used for the FCC feasibility study.

Part of this code is implemented in key4hep, checkout [this repo](https://github.com/saracreates/JetTagging).

## Overview of this repository

- the `src` folders contain the code to extract all desired variables from full simulation. There are different variations such as 
  - fixing the pandora algorithm by using tracks and not PFOs to describe charged particles (`src_tc-match`) 
  - using tracks only and neglecting neutral particles (`src_tracks`). 
  - `src_trackPFOcheck` does a study on lost charged particles is PFOs. 
  - `src_debug_tracks` performs a study on the 5 helix track parameters and its covariance parameters. 
  Each of these folders has `create_jet_based_tree.py` and `tree_tools.py` where the core of the work lies. 

- We use condor to submit the jobs for creating desired root files. The submit files (`analysis.sub`) for the different cases are in respective `submit` folders and automatically created by the `write_analysis.py` files. The `analysis.sub` files execute the `submitAnalysisJob.sh` script. 

## Example usage
 - clone this repository
 - source key4hep via `source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2024-04-12`
 - go into the desired `src` folder and run 

```python create_jet_based_tree.py /eos/experiment/fcc/prod/fcc/ee/test_spring2024/240gev/Hbb/CLD_o2_v05/rec/00016562/000/Hbb_rec_16562_111.root test_output.root```

## Citation

If you find this code helpful and use it in your research, please cite:

```
@manual{aumiller_2024_4pcr6-r0d06,
  title        = {Jet Flavor Tagging Performance at FCC-ee},
  author       = {Aumiller, Sara and
                  Garcia, Dolores and
                  Selvaggi, Michele},
  month        = nov,
  year         = 2024,
  doi          = {10.17181/4pcr6-r0d06},
  url          = {https://doi.org/10.17181/4pcr6-r0d06}
}
```
