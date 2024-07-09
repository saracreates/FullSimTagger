#!/usr/bin/env python
import os, sys, subprocess
import glob
import argparse
import time

# ____________________________________________________________________________________________________________
def absoluteFilePaths(directory):
    files = []
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            files.append(os.path.abspath(os.path.join(dirpath, f)))
    return files


# _____________________________________________________________________________________________________________
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--indir",
        help="input directory containing edm trees (most likely on eos)",
        dest="indir",
        default="/eos/experiment/fcc/ee/generation/DelphesStandalone/Edm4Hep/pre_winter2023_tests_v2/p8_ee_ZZ_ecm240",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        help="output directory (default will have same name as input dir)",
        dest="outdir",
        default="/eos/experiment/fcc/ee/generation/DelphesStandalone/flat_trees/hss_analysis/p8_ee_ZZ_ecm240",
    )

    parser.add_argument(
        "-s",
        "--script",
        help="python analysis script",
        dest="script",
        default="examples/FCCee/weaver/analysis_inference.py",
    )
    parser.add_argument("-n", "--nev", help="max number of events (-1 runs on all events)", default=-1)
    parser.add_argument("--njobs", help="max number of jobs", default=-1)

    parser.add_argument("--dry", help="check local submission", action="store_true")

    parser.add_argument(
        "--queue",
        help="queue for condor",
        choices=[
            "espresso",
            "microcentury",
            "longlunch",
            "workday",
            "tomorrow",
            "testmatch",
            "nextweek",
        ],
        default="workday",
    )

    args = parser.parse_args()

    indir = os.path.abspath(args.indir)
    outdir = os.path.abspath(args.outdir)
    analysis = os.path.abspath(args.script)
    queue = args.queue
    nev = args.nev
    njobs = int(args.njobs)

    if not "LOCAL_DIR" in os.environ:
        sys.exit("ERROR: need to setup fccanalyses environment ...")

    # find list of already produced files:
    list_of_outfiles = []
    for name in glob.glob("{}/*.root".format(outdir)):

        ## consider only files > 100 kB
        if os.path.getsize(name) > 1e5:
            list_of_outfiles.append(name)
            list_of_infiles = []
    for name in glob.glob("{}/*.root".format(indir)):
        list_of_infiles.append(name)

    print(list_of_infiles)
    script = "submitAnalysisJob.sh"
    jobCount = 0

    cmdfile = """# here goes your shell script
executable    = {}
#requirements = (OpSysAndVer =?= "CentOS7")
# here you specify where to put .log, .out and .err files
output                = std/condor.$(ClusterId).$(ProcId).out
error                 = std/condor.$(ClusterId).$(ProcId).err
log                   = std/condor.$(ClusterId).log

+AccountingGroup = "group_u_FCC.local_gen"
+JobFlavour    = "{}"
""".format(
        script, queue
    )
    print("")

    rm_empty_cmd_file = "rm "

    if njobs == -1:
        njobs = len(list_of_infiles)
    number_of_jobs = min(njobs, len(list_of_infiles))

    for job in range(number_of_jobs):

        basename = "events_" + str(job)
        outputFile = outdir + "/" + basename + ".root"
        inputFile = list_of_infiles[job]
        # print outdir, basename, outputFile

        if not outputFile in list_of_outfiles:
            # print("{} : missing output file ".format(outputFile))
            jobCount += 1

            argts = "{} {} {} {} {}".format(os.environ["LOCAL_DIR"], analysis, inputFile, outputFile, nev)

            cmdfile += 'arguments="{}"\n'.format(argts)
            cmdfile += "queue\n"

            cmd = "rm -rf job; ./{} {}".format(script, argts)

            rm_empty_cmd_file += "{} ".format(outputFile)

            if jobCount == 1:
                print("")
                print(cmd)

    print("")
    print("submitting {} files ".format(jobCount))
    print("")
    print("remove empty files ... ")
    os.system(rm_empty_cmd_file)

    with open("condor_analysis.sub", "w") as f:
        f.write(cmdfile)

    ### submitting jobs
    if jobCount > 0:
        if not args.dry:
            print("")
            print("[Submitting jobs] ... ")
            os.system("condor_submit condor_analysis.sub")


# _______________________________________________________________________________________
if __name__ == "__main__":
    main()
            
