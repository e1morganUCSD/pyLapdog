"""
initial testing of new model structure, completes processing of models, but does not analyze results yet
"""

import sys

sys.path.insert(0, "/home/AD/e1morgan/PycharmProjects/pyLapdog/")

import contrastmodel.params.paramsDef as par
import contrastmodel.functions.stimuli as stimclass
import contrastmodel.functions.models as modelclass
import contrastmodel.functions.subjects as subj

#mainDir = "C:\\Users\\Eric\\Documents\\PyLapdog_Output\\initialtest\\"
mainDir = "/home/AD/e1morgan/Documents/e1morgan_data/pyLapdog_output/two_lapdog_test/"

print("Generating params:")
params = par.FilterParams(mainDir, verbosity=3)

# #reduce the parameters for now, for quicker testing of later stuff
# params.filt_orientations = params.filt_orientations[0:3]
# params.filt_stdev_pixels = params.filt_stdev_pixels[0:3]
params.load_filtermasks()

print("Generating models:")
# modellist = [["lapdog2", [3], [[2, 1]]]]
modellist = [["lapdog", [1], [[2]]],
             ["lapdog2", [1, 5], [[2, 1]]]]

models = {}
for model in range(len(modellist)):
    models[model] = modelclass.LapdogModel(modellist[model][0], modellist[model][1], modellist[model][2])

print("done generating models")

print("Generating stims:")
stimlist = ["Whites",
            "Rings"]
            # "Whites (DD)",
            # "Zigzag"]

stims = {}

for stimname in stimlist:
    stims[stimname] = stimclass.Stim(stimname, params)

results = {}        # used to hold final processed output
results_dirs = {}   # holds directories for each model output
diffs = {}          # used to hold patch differences for each stim/model

print("Processing stims:")
for stimname in stimlist:
    print("--Processing " + stimname)

    results[stimname] = {}
    results_dirs[stimname] = {}
    diffs[stimname] = {}

    for modelnum in range(len(models)):
        print("----Processing model {} of {}".format(modelnum + 1, len(models)))
        results_temp, results_dirs_temp = models[modelnum].process_stim(stims[stimname])
        results[stimname].update(results_temp)
        results_dirs[stimname].update(results_dirs_temp)

    diffs[stimname].update(stims[stimname].find_diffs(results[stimname], results_dirs[stimname]))

subj_data = subj.get_subject_data()
subj.plot_diffs(diffs, subj_data, stimlist, params.mainDir)
