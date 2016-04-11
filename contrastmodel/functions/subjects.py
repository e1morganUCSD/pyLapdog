"""
Provides functions and data related to comparing model results with human subject data
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.cm as mpltcm
import matplotlib.colors as colors


def get_subject_data():
    """

    :return: dictionary of human subject values, using stimulus names as keys
    :rtype: dict[str, list[float32]]
    """
    # first 33 rows are high-side responses for each stimulus, second 33 rows are low-side responses for each stimulus
    # each row consists of responses for each of 14 subjects
    responses = [[77.1667, 92.1250, 100.3750, 74.1111, 84.5000, 79.7000, 71.7500,
                  88.5556, 82.6250, 78.8889, 90.1429, 97.7000, 76.2500, 96.7500],
                 [81.5000, 95.7778, 98.3333, 67.0000, 85.7778, 83.8889, 74.4444, 92.8889,
                 87.6667, 79.5556, 92.4286, 92.1111, 84.1111, 98.1250],
                 [79.0000, 102.2500, 107.5000, 81.3333, 94.4444, 95.5556, 83.6250,
                 90.0000, 94.8000, 77.3750, 92.0000, 99.2222, 82.8889, 100.7778],
                 [94.3333, 101.7778, 104.1111, 74.3000, 92.4444, 102.3333, 81.0000,
                 85.6667, 96.5000, 86.4444, 96.4286, 99.4444, 90.1250, 111.4444],
                 [103.2857, 105.7778, 114.7778, 80.8750, 104.4444, 105.6667, 76.0000,
                 88.7500, 102.7000, 94.3000, 96.2857, 90.3750, 87.7778, 104.0000],
                 [74.6667, 95.3750, 99.2500, 64.7500, 87.7000, 86.8750, 77.6250,
                 88.3750, 86.3333, 78.6000, 90.0000, 94.8889, 83.4444, 99.2222],
                 [81.5000, 94.1111, 94.0000, 84.4444, 94.8889, 102.8889, 85.8750,
                 86.7000, 93.0000, 85.4444, 96.5000, 91.1111, 88.8750, 93.8750],
                 [69.8000, 96.5000, 103.4444, 82.1111, 90.2222, 83.3333, 80.7778,
                 94.1111, 85.7500, 73.2222, 89.0000, 96.1250, 85.4444, 95.0000],
                 [79.1429, 97.1111, 97.7778, 74.6250, 92.0000, 94.3750, 86.0000,
                 90.1111, 93.3750, 82.2222, 96.5714, 100.6250, 92.1111, 95.4444],
                 [81.0000, 94.6667, 87.6250, 71.7000, 86.2222, 90.8889, 79.0000,
                 86.6250, 88.8000, 76.0000, 89.1429, 89.1250, 83.1111, 90.6667],
                 [74.8333, 98.2500, 97.0000, 71.4444, 89.6667, 93.5556, 87.8889,
                 94.5556, 93.1111, 83.0000, 91.7500, 92.2500, 79.5556, 93.5556],
                 [151.0000, 168.0000, 166.8889, 154.1250, 163.5714, 151.1111, 165.0000,
                 146.5556, 148.1250, 144.8750, 154.0000, 167.5000, 161.1111, 153.1111],
                 [159.8571, 169.8889, 173.2500, 157.5000, 166.3333, 154.4444, 163.2500,
                 148.1111, 151.7778, 152.0000, 154.8571, 173.4444, 162.0000, 152.6667],
                 [156.8333, 172.7778, 174.4444, 156.8750, 167.7500, 159.2500, 166.2222,
                 154.6250, 152.0000, 152.1111, 158.5714, 178.8889, 162.3000, 156.1429],
                 [160.3333, 174.5556, 173.5000, 154.0000, 172.5000, 160.8889, 158.3000,
                 152.0000, 153.1111, 153.2500, 165.1429, 175.0000, 161.5556, 156.7778],
                 [165.1667, 171.3750, 170.7778, 148.4000, 168.2222, 158.3333, 160.8889,
                 151.7500, 148.3333, 155.7500, 153.0000, 170.2000, 152.4444, 149.4444],
                 [163.0000, 171.7778, 176.6667, 156.3333, 172.8750, 159.0000, 164.8750,
                 155.0000, 155.4444, 147.5556, 158.8571, 176.8000, 162.1111, 151.1111],
                 [173.5000, 167.0000, 176.0000, 164.8750, 165.7778, 161.7778, 172.0000,
                 158.7500, 159.8889, 155.4444, 165.7500, 186.3750, 162.3333, 158.1111],
                 [154.4000, 167.9000, 165.4444, 142.3333, 164.3333, 146.6250, 165.8000,
                 152.1111, 151.1111, 148.0000, 150.6250, 160.6667, 160.0000, 150.6667],
                 [161.5714, 172.2500, 170.6000, 157.3750, 166.2500, 157.2500, 166.6250,
                 159.7500, 154.5556, 152.4444, 157.7143, 179.5000, 159.3000, 156.0000],
                 [168.1667, 173.7500, 174.0000, 160.2222, 169.8750, 155.2500, 170.4444,
                 154.8889, 152.3750, 149.1111, 163.0000, 179.2222, 162.6667, 157.3333],
                 [156.2857, 170.2500, 174.4444, 150.8000, 167.1111, 155.8889, 163.0000,
                 154.3333, 154.3333, 148.8750, 161.4286, 182.1111, 159.8750, 153.8889],
                 [44.0000, 46.5556, 40.6667, 33.6667, 44.3333, 38.8889, 32.3333,
                 43.1111, 45.1250, 42.4444, 46.5714, 42.3333, 48.3333, 49.1111],
                 [39.1429, 45.8889, 39.6250, 38.6667, 45.0000, 40.8889, 32.6250,
                 43.7500, 45.0000, 34.7778, 49.6250, 40.1111, 47.1250, 47.7778],
                 [41.0000, 47.3333, 42.7500, 34.4444, 45.6667, 42.8889, 29.4444,
                 44.2500, 43.5000, 40.1111, 48.2500, 44.2222, 50.4444, 48.2222],
                 [46.5000, 53.3750, 44.2000, 40.4444, 51.8000, 44.1000, 37.1250,
                 45.1250, 50.3333, 43.6250, 52.0000, 47.8889, 50.1250, 48.6667],
                 [49.0000, 56.8000, 51.0000, 41.4444, 56.4444, 50.6667, 34.8889,
                 48.4000, 51.7500, 46.2222, 53.7143, 50.8750, 48.6667, 52.1000],
                 [40.0000, 44.8889, 39.8889, 34.5000, 44.6667, 36.9000, 29.1111,
                 44.1250, 41.7778, 38.0000, 47.5000, 39.8750, 44.2222, 45.5556],
                 [46.1667, 48.8889, 45.0000, 40.1250, 52.5714, 48.7778, 36.5556,
                 45.0000, 48.5000, 44.7778, 53.8571, 46.6667, 51.6250, 48.0000],
                 [38.3333, 44.7000, 45.4444, 31.7778, 48.1111, 39.7000, 33.5000,
                 44.5000, 47.3333, 38.3333, 49.8750, 40.7143, 51.6000, 47.4444],
                 [38.4286, 43.1250, 44.3750, 35.6250, 47.3333, 40.0000, 31.8889,
                 44.0000, 44.1111, 42.6667, 47.0000, 46.3333, 44.4444, 48.2500],
                 [40.4286, 48.8889, 41.0000, 38.2222, 47.1111, 39.5000, 31.7778,
                 42.5556, 49.2500, 42.8750, 49.2857, 45.8750, 45.5556, 48.5000],
                 [40.4000, 44.4444, 42.2500, 37.5556, 45.0000, 39.4444, 32.3000,
                 43.6250, 45.5556, 40.0000, 46.1429, 43.8889, 47.3333, 49.6667],
                 [102.0000, 107.2000, 110.8750, 93.8889, 107.2222, 106.7778, 113.3750,
                 94.6250, 104.6667, 93.0000, 103.2857, 112.6667, 91.1111, 107.4444],
                 [104.0000, 105.8750, 111.8750, 88.3333, 108.2222, 104.2500, 105.4000,
                 94.2000, 99.2500, 90.2500, 103.0000, 103.4444, 100.6667, 107.6250],
                 [85.1667, 103.0000, 107.4444, 87.8750, 103.4444, 91.3333, 97.1111,
                 89.4444, 95.5556, 78.5000, 99.0000, 100.2222, 92.4444, 111.4444],
                 [77.1667, 99.6667, 112.0000, 90.7778, 105.1111, 88.7778, 97.1111,
                 89.7778, 95.1111, 78.8750, 105.4286, 120.2222, 92.9000, 105.8889],
                 [70.6667, 97.8889, 108.5000, 85.1250, 102.2222, 89.4444, 90.3750,
                 84.3750, 89.2222, 80.0000, 101.1429, 107.3333, 90.3750, 100.5556],
                 [97.8333, 101.5556, 102.8750, 93.2222, 106.0000, 103.3333, 97.6667,
                 85.0000, 95.0000, 89.5556, 98.0000, 112.4444, 93.6667, 105.7000],
                 [99.3333, 100.3333, 104.3750, 92.3750, 107.6250, 98.3000, 107.8889,
                 95.8750, 99.5556, 87.6667, 97.7500, 124.4444, 107.2222, 107.0000],
                 [96.3333, 103.1111, 106.5000, 85.0000, 100.7778, 102.4444, 88.2500,
                 85.7500, 97.0000, 87.8889, 107.8571, 101.0000, 91.2222, 103.5000],
                 [90.6667, 102.8750, 111.5000, 85.7500, 106.6667, 96.2222, 106.7500,
                 93.4444, 100.5556, 85.8889, 99.4286, 117.2222, 100.3333, 109.1250],
                 [89.6667, 103.6667, 116.1250, 89.3333, 111.8750, 95.4444, 115.8750,
                 92.7500, 104.3333, 87.2000, 102.8750, 123.1250, 102.2222, 121.1111],
                 [98.3333, 96.1000, 116.7500, 81.2222, 104.5556, 94.5556, 98.3333,
                 91.8889, 99.1250, 84.3333, 103.8571, 105.6667, 93.4444, 107.5556],
                 [160.5000, 165.2222, 172.9000, 162.4444, 160.0000, 154.6667, 166.8889,
                 156.0000, 152.4444, 146.8889, 155.6667, 173.5000, 154.7500, 153.6000],
                 [159.5714, 170.0000, 174.0000, 144.1250, 162.3000, 154.5000, 168.8889,
                 151.6667, 154.5000, 154.6667, 158.7143, 172.7778, 156.7500, 155.2500],
                 [157.8333, 164.7778, 173.7778, 153.8889, 163.4444, 157.5556, 166.2500,
                 153.5000, 151.8750, 153.5556, 157.2857, 179.1111, 160.2222, 155.3750],
                 [149.1667, 161.1250, 168.6667, 145.6667, 162.0000, 151.2500, 158.8750,
                 147.1111, 148.8889, 147.6667, 154.6250, 163.8750, 151.5556, 148.5556],
                 [146.8333, 162.0000, 165.1000, 144.5556, 162.0000, 151.4286, 156.6250,
                 146.0000, 149.3333, 154.2222, 152.1429, 157.4444, 150.7500, 147.9000],
                 [157.1429, 167.7500, 169.8571, 155.8750, 166.6667, 156.4444, 168.2222,
                 159.2857, 153.3333, 152.1111, 152.8571, 171.1111, 156.6667, 154.4444],
                 [155.4286, 162.0000, 175.1111, 149.8000, 168.8889, 160.2222, 174.7778,
                 157.5556, 153.8889, 153.3333, 152.4286, 166.6250, 154.4444, 154.5556],
                 [159.5000, 167.1250, 167.0000, 146.0000, 158.6667, 151.8889, 164.2222,
                 156.1111, 154.1250, 154.8889, 160.1667, 169.1250, 151.7778, 155.2222],
                 [146.5000, 163.7778, 172.4444, 146.7500, 158.6667, 152.8889, 167.6667,
                 154.3333, 148.6667, 152.1111, 156.4286, 169.3333, 158.7778, 152.1000],
                 [140.1667, 161.4444, 168.4000, 145.6667, 159.8000, 153.3000, 160.1250,
                 151.2000, 147.2500, 143.7500, 152.7143, 161.3333, 159.4000, 150.6667],
                 [144.7143, 166.6667, 170.2222, 149.5000, 165.8750, 155.2222, 166.7000,
                 153.2222, 152.8889, 152.0000, 157.4286, 169.6250, 161.1250, 147.5000],
                 [32.8333, 42.6250, 42.5556, 30.8889, 44.6000, 34.7778, 33.8000,
                 41.4444, 43.2500, 40.4444, 44.0000, 42.2500, 43.3750, 44.8750],
                 [38.0000, 42.0000, 38.2222, 34.6250, 42.3333, 37.8889, 34.0000,
                 42.6667, 42.1000, 41.0000, 44.5714, 39.5556, 48.0000, 45.8889],
                 [38.1667, 42.6667, 37.0000, 33.0000, 42.5000, 36.2222, 30.5000,
                 44.8889, 40.7778, 41.0000, 47.3750, 41.5000, 43.0000, 45.6667],
                 [36.0000, 41.0000, 39.0000, 32.7778, 40.3000, 35.1250, 35.0000,
                 41.8889, 40.2500, 39.3000, 40.4286, 42.3750, 41.1250, 46.4444],
                 [34.6667, 43.5556, 37.3333, 34.7500, 39.3333, 37.8000, 34.1111,
                 43.7778, 41.7778, 41.3333, 44.5714, 43.8750, 44.7778, 47.7778],
                 [32.7143, 40.6667, 40.6667, 29.3750, 41.3333, 34.4444, 26.7778,
                 42.1111, 42.3333, 39.6667, 43.1250, 40.0000, 45.0000, 44.5556],
                 [31.0000, 38.6667, 34.8750, 31.7778, 43.5556, 37.4444, 28.8889,
                 40.7778, 40.0000, 41.0000, 42.1429, 42.6250, 38.7778, 43.6250],
                 [36.6667, 41.8000, 43.4444, 36.2500, 42.0000, 40.4000, 31.3750,
                 43.8889, 42.0000, 40.8750, 47.1429, 38.7500, 45.2500, 45.0000],
                 [34.0000, 40.5556, 39.1111, 29.7000, 41.6250, 33.5556, 30.0000,
                 41.8889, 41.0000, 40.4444, 44.0000, 38.3750, 44.4444, 44.3750],
                 [30.8571, 39.3333, 34.1250, 27.0000, 38.4000, 31.4444, 30.6250,
                 43.0000, 39.4000, 39.3750, 39.4286, 36.0000, 44.6250, 43.0000],
                 [33.3333, 39.6667, 37.4444, 31.8750, 40.5556, 34.6250, 28.8000,
                 43.2500, 43.2500, 40.7000, 43.1429, 35.5556, 45.2222, 45.7778]]

    stimlabels = ['Whites', 'Howe var B', 'Howe', 'Howe var D', 'SBC', 'Anderson', 'Rings', 'Radial', 'Zigzag',
                  'Jacob 1', 'Jacob 2', 'Whites (DI)', 'Howe var B (DI)', 'Howe (DI)', 'Howe var D (DI)',
                  'SBC (DI)', 'Anderson (DI)', 'Rings (DI)', 'Radial (DI)', 'Zigzag (DI)', 'Jacob 1 (DI)',
                  'Jacob 2 (DI)', 'Whites (DD)', 'Howe var B (DD)', 'Howe (DD)', 'Howe var D (DD)', 'SBC (DD)',
                  'Anderson (DD)', 'Rings (DD)', 'Radial (DD)', 'Zigzag (DD)', 'Jacob 1 (DD)', 'Jacob 2 (DD)']

    # find diff values for each stimulus and subject
    output = {}
    for rowindex in range(33):
        hi_row = responses[rowindex]
        lo_row = responses[rowindex + 33]
        for value in range(len(hi_row)):
            hi_row[value] = hi_row[value] - lo_row[value]
        output[stimlabels[rowindex]] = hi_row

    return output


def plot_diffs(diffs, human_data, stimlist, outdir):
    """

    :param outdir: output directory for plot files
    :type outdir: str
    :param stimlist: list of stimuli names processed by the model
    :type stimlist: list[str]
    :param diffs: dictionary of model results per stim
    :type diffs: dict[str, dict[str, float32]]
    :param human_data: dictionary of human responses to stimuli
    :type human_data: dict[str, list[float32]
    :return:
    :rtype:
    """

    model_names = []        # to store model names in order
    model_diffs = {}
    subject_names = []      # to store human subject 'names' in order
    subject_diffs = {}

    # get the names of all the models, using stimulus #1 as an example
    for key in diffs[stimlist[0]]:
        model_names.append(key)
        model_diffs[key] = []

    # get subject 'names'
    for num in range(len(human_data[stimlist[0]])):
        key = "subject #{}".format(num + 1)
        subject_names.append(key)
        subject_diffs[num] = []

    # get max values while iterating through the list for later normalization
    model_max_diff = 0
    subject_max_diff = 0

    for stimname in stimlist:
        modelvals = diffs[stimname]
        subject_vals = human_data[stimname]
        for key in model_names:
            model_diffs[key].append(modelvals[key])      # add the current stimulus' diff value to the end of the
            # model's list
            if abs(modelvals[key]) > model_max_diff:
                model_max_diff = abs(modelvals[key])

        for keynum in range(len(subject_names)):
            subject_diffs[keynum].append(subject_vals[keynum])
            if abs(subject_vals[keynum]) > subject_max_diff:
                subject_max_diff = abs(subject_vals[keynum])

    # normalize diffs
    for modeldiff in model_diffs:
        for index in range(len(model_diffs[modeldiff])):
            if model_max_diff != 0:
                model_diffs[modeldiff][index] /= model_max_diff

    for subjectdiff in subject_diffs:
        for index in range(len(subject_diffs[subjectdiff])):
            if subject_max_diff != 0:
                subject_diffs[subjectdiff][index] /= subject_max_diff

    # create output directory for plots
    new_outdir = outdir + "comparisons/"
    if not os.path.exists(new_outdir):
        os.makedirs(new_outdir)

    # create a plot for each subject vs model
    index = range(len(stimlist))
    bar_width = 0.35

    for subject_num in range(len(subject_names)):
        for modelname in model_names:
            filename = "{}_results_{}.png".format(modelname, subject_num)

            fig, ax = plt.subplots()

            rects1 = plt.bar(index, model_diffs[modelname], bar_width, label=modelname, color='b')
            second_bar_indexes = list(x + bar_width for x in index)
            rects2 = plt.bar(second_bar_indexes, subject_diffs[subject_num], bar_width,
                             label="Subject #{}".format(subject_num + 1), color='y')

            plt.xlabel("Stimuli")
            plt.ylabel("Response Difference")
            plt.title("{} - results vs. Subject #{}".format(modelname, subject_num + 1))
            plt.xticks([x + bar_width for x in index], stimlist, rotation=45, horizontalalignment='right')

            plt.legend()
            # plt.tight_layout()

            # the following will work if Qt is the backend for matplotlib.  TODO: make this universal
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()

            plt.savefig(new_outdir + filename)
            plt.close()

    # calculate model success rates and fit values
    model_correct_count = np.zeros((len(subject_names), len(model_names)))
    model_correct_vals = np.zeros((len(subject_names), len(model_names)))
    for subject_num in range(len(subject_names)):
        for modelnum in range(len(model_names)):
            modelname = model_names[modelnum]
            correctcount = 0
            correctval = 0

            for index in range(len(model_diffs[modelname])):
                if math.copysign(1, model_diffs[modelname][index]) == \
                        math.copysign(1, subject_diffs[subject_num][index]):  # if the human and model vals have
                    #                                                           the same sign
                    correctcount += 1
                    correctval += abs(model_diffs[modelname][index] - subject_diffs[subject_num][index])

            model_correct_count[subject_num, modelnum] = correctcount
            model_correct_vals[subject_num, modelnum] = correctval

    # plot model success rates and values
    filename = "Model_correct_count_totals.png"

    fig = plt.figure(figsize=(12.0, 12.0))  # figure size in inches
    ax = fig.add_subplot(1, 1, 1)

    indexes = range(len(model_names))
    num_colors = len(subject_names)
    cm = plt.get_cmap('rainbow')
    cNorm = colors.Normalize(vmin=0, vmax=num_colors-1)
    scalarMap = mpltcm.ScalarMappable(norm=cNorm, cmap=cm)
    ax.set_color_cycle([scalarMap.to_rgba(i) for i in range(num_colors)])
    # color_iterator = [scalarMap. for i in range(num_colors)]
    # colorlist = cm(color_iterator)
    # ax.set_prop_cycle(color=colorlist)

    # colorlist = list(cm(1.*i/len(subject_names) for i in range(len(subject_names))))
    # color = iter(cm.rainbow(np.linspace(0, 1, len(subject_names))))
    for subject_num in range(len(subject_names)):
        # c = next(colorlist)
        plt.plot(indexes, model_correct_count[subject_num, :], label="Subject #{}".format(subject_num + 1))

    plt.xticks(indexes, model_names, rotation=45, horizontalalignment='right')
    plt.legend()
    plt.tight_layout()

    plt.savefig(new_outdir + filename)
    plt.close()

    # save individual plots for each subject
    for subject_num in range(len(subject_names)):
        filename = "Model_correct_count_totals_subject{}.png".format(subject_num)
        plt.subplot(2, 1, 1)
        plt.plot(indexes, model_correct_count[subject_num, :])
        plt.xticks(indexes, model_names, rotation=45, horizontalalignment='right')

        plt.subplot(2, 1, 2)
        plt.plot(indexes, model_correct_vals[subject_num, :])
        plt.xticks(indexes, model_names, rotation=45, horizontalalignment='right')

        plt.tight_layout()

        # the following will work if Qt is the backend for matplotlib.  TODO: make this universal
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()

        plt.savefig(new_outdir + filename)
        plt.close()







