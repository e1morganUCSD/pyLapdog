"""
contains class definition for stimuli
"""

import numpy as np
import math
import contrastmodel.params.paramsDef as par
import matplotlib.pyplot as plt
import scipy.signal as ss
import contrastmodel.functions.imaging as imaging
import contrastmodel.functions.gpufunc as gpuf
# from numpy.fft import fft2, ifft2
import os
from progressbar import ProgressBar, Bar, Percentage


class Stim(object):
    """
    Stimulus processed by model - also holds filter responses to each stimulus
    """
    def __init__(self, stimtype, params):
        """

        :param str stimtype: type of stimulus generated
        :param par.FilterParams params: filter parameters and filters
        """
        # strip off variant name if present
        if stimtype[-2:] == "DD" or stimtype[-2:] == "DI":
            variant = stimtype[-2:]
            stimtype = stimtype[:-3]
        else:
            variant = ""

        self.variant = variant
        self.stimtype = stimtype
        self.params = params

        # create "friendly" name of stimulus for use in figures, folders, etc
        self.friendlyname = stimtype
        if variant != "":
            self.friendlyname = self.friendlyname + " (" + variant + ")"

        # create output directory stub for stimulus
        self.outDir = self.friendlyname + "/"

        print("Generating stimulus " + self.friendlyname + "...")

        # generate stimulus image
        if stimtype == "Whites":
            self.img, self.regions, self.cutX, self.cutY, self.bg_region, \
                self.low_region, self.high_region = _make_whites()
            if variant == "DD":
                self.img[self.img == 0.0] = -1.0
                self.img[self.img == 0.5] = 0.0
                self.img[self.img == -1.0] = 0.5
                self.img = _fix_variant_bg(self.img, 128, 128, 0, 0)
            elif variant == "DI":
                self.img[self.img == 0.5] = -1.0
                self.img[self.img == 1.0] = 0.5
                self.img[self.img == -1.0] = 1.0
                self.img = _fix_variant_bg(self.img, 128, 128, 0, 0)

        elif stimtype == "Howe var B":
            self.img, self.regions, self.cutX, self.cutY, self.bg_region, \
                self.low_region, self.high_region = _make_howe_b()
            if variant == "DD":
                self.img[self.img == 0.0] = -1.0
                self.img[self.img == 0.5] = 0.0
                self.img[self.img == -1.0] = 0.5
                self.img = _fix_variant_bg(self.img, 128, 128, 0, 0)
            elif variant == "DI":
                self.img[self.img == 0.5] = -1.0
                self.img[self.img == 1.0] = 0.5
                self.img[self.img == -1.0] = 1.0
                self.img = _fix_variant_bg(self.img, 128, 128, 0, 0)

        elif stimtype == "Howe":
            self.img, self.regions, self.cutX, self.cutY, self.bg_region, \
                self.low_region, self.high_region = _make_howe()
            if variant == "DD":
                self.img[self.img == 0.0] = -1.0
                self.img[self.img == 0.5] = 0.0
                self.img[self.img == -1.0] = 0.5
                self.img = _fix_variant_bg(self.img, 128, 128, 0, 0)
            elif variant == "DI":
                self.img[self.img == 0.5] = -1.0
                self.img[self.img == 1.0] = 0.5
                self.img[self.img == -1.0] = 1.0
                self.img = _fix_variant_bg(self.img, 128, 128, 0, 0)

        elif stimtype == "Howe var D":
            self.img, self.regions, self.cutX, self.cutY, self.bg_region, \
                self.low_region, self.high_region = _make_howe_d()
            if variant == "DD":
                self.img[self.img == 0.0] = -1.0
                self.img[self.img == 0.5] = 0.0
                self.img[self.img == -1.0] = 0.5
                self.img = _fix_variant_bg(self.img, 128, 128, 0, 0)
            elif variant == "DI":
                self.img[self.img == 0.5] = -1.0
                self.img[self.img == 1.0] = 0.5
                self.img[self.img == -1.0] = 1.0
                self.img = _fix_variant_bg(self.img, 128, 128, 0, 0)

        elif stimtype == "SBC":
            self.img, self.regions, self.cutX, self.cutY, self.bg_region, \
                self.low_region, self.high_region = _make_sbc()
            if variant == "DD":
                self.img[self.img == 0.0] = -1.0
                self.img[self.img == 0.5] = 0.0
                self.img[self.img == -1.0] = 0.5
                self.img = _fix_variant_bg(self.img, 128, 128, 0, 0)
            elif variant == "DI":
                self.img[self.img == 0.5] = -1.0
                self.img[self.img == 1.0] = 0.5
                self.img[self.img == -1.0] = 1.0
                self.img = _fix_variant_bg(self.img, 128, 128, 0, 0)

        elif stimtype == "Anderson":
            self.img, self.regions, self.cutX, self.cutY, self.bg_region, \
                self.low_region, self.high_region = _make_anderson()
            if variant == "DD":
                self.img[self.img == 0.0] = -1.0
                self.img[self.img == 0.5] = 0.0
                self.img[self.img == -1.0] = 0.5
                self.img = _fix_variant_bg(self.img, 128, 128, 0, 0)
            elif variant == "DI":
                self.img[self.img == 0.5] = -1.0
                self.img[self.img == 1.0] = 0.5
                self.img[self.img == -1.0] = 1.0
                self.img = _fix_variant_bg(self.img, 128, 128, 0, 0)

        elif stimtype == "Rings":
            self.img, self.regions, self.cutX, self.cutY, self.bg_region, \
                self.low_region, self.high_region = _make_rings()
            if variant == "DD":
                self.img[self.img == 0.0] = -1.0
                self.img[self.img == 0.5] = 0.0
                self.img[self.img == -1.0] = 0.5
                self.img = _fix_variant_bg(self.img, 128, 128, 0, 0)
            elif variant == "DI":
                self.img[self.img == 0.5] = -1.0
                self.img[self.img == 1.0] = 0.5
                self.img[self.img == -1.0] = 1.0
                self.img = _fix_variant_bg(self.img, 128, 128, 0, 0)

        elif stimtype == "Radial":
            self.img, self.regions, self.cutX, self.cutY, self.bg_region, \
                self.low_region, self.high_region = _make_radial()
            if variant == "DD":
                self.img[self.img == 0.0] = -1.0
                self.img[self.img == 0.5] = 0.0
                self.img[self.img == -1.0] = 0.5
                self.img = _fix_variant_bg(self.img, 256, 256, 256, 256)
            elif variant == "DI":
                self.img[self.img == 0.5] = -1.0
                self.img[self.img == 1.0] = 0.5
                self.img[self.img == -1.0] = 1.0
                self.img = _fix_variant_bg(self.img, 256, 256, 256, 256)

        elif stimtype == "Zigzag":
            self.img, self.regions, self.cutX, self.cutY, self.bg_region, \
                self.low_region, self.high_region = _make_zigzag()
            if variant == "DD":
                self.img[self.img == 0.0] = -1.0
                self.img[self.img == 0.5] = 0.0
                self.img[self.img == -1.0] = 0.5
                self.img = _fix_variant_bg(self.img, 128, 128, 128, 128)
            elif variant == "DI":
                self.img[self.img == 0.5] = -1.0
                self.img[self.img == 1.0] = 0.5
                self.img[self.img == -1.0] = 1.0
                self.img = _fix_variant_bg(self.img, 128, 128, 128, 128)

        elif stimtype == "Jacob 1":
            self.img, self.regions, self.cutX, self.cutY, self.bg_region, \
                self.low_region, self.high_region = _make_jacob_1()
            if variant == "DD":
                self.img[self.img == 0.0] = -1.0
                self.img[self.img == 0.5] = 0.0
                self.img[self.img == -1.0] = 0.5
                self.img = _fix_variant_bg(self.img, 128, 128, 128, 128)
            elif variant == "DI":
                self.img[self.img == 0.5] = -1.0
                self.img[self.img == 1.0] = 0.5
                self.img[self.img == -1.0] = 1.0
                self.img = _fix_variant_bg(self.img, 128, 128, 128, 128)

        elif stimtype == "Jacob 2":
            self.img, self.regions, self.cutX, self.cutY, self.bg_region, \
                self.low_region, self.high_region = _make_jacob_2()
            if variant == "DD":
                self.img[self.img == 0.0] = -1.0
                self.img[self.img == 0.5] = 0.0
                self.img[self.img == -1.0] = 0.5
                self.img = _fix_variant_bg(self.img, 128, 128, 128, 128)
            elif variant == "DI":
                self.img[self.img == 0.5] = -1.0
                self.img[self.img == 1.0] = 0.5
                self.img[self.img == -1.0] = 1.0
                self.img = _fix_variant_bg(self.img, 128, 128, 128, 128)

        # convert img to single precision to save memory
        self.img = self.img.astype('float32')

        # print stimulus img
        _filename = self.friendlyname + ".png"
        _title = self.friendlyname
        _outdir = self.params.mainDir + self.outDir
        # make sure output dir exists
        if not os.path.exists(_outdir):
            os.makedirs(_outdir)
        imaging.generate_image(self.img, _title, _filename, _outdir,
                               cmap="gray")

        # get responses of filters to stimulus
        print("Generating responses of filters to " + self.friendlyname)
        self.filtresponses, self.ap_filtresponses = \
            _gen_filter_response(self.params, self.img)

        # (optionally) print filter responses for stimulus
        if params.verbosity > 1:
            print("Printing filter images...")
            self._print_filter_responses()

        print("Done Generating " + self.friendlyname)

    def _print_filter_responses(self):
        """
        Prints filter responses to stimulus to image file

        
        """
        outdir = self.params.mainDir + self.outDir

        for o in range(len(self.params.filt_orientations)):
            for f in range(len(self.params.filt_stdev_pixels)):
                _filename = self.params.filttype + "_filterresponse-{}-" \
                    "{}.png".format(self.params.filt_orientations[o],
                                    self.params.filt_stdev_pixels[f])
                _title = "Initial {} filter response: orientation {}, " \
                         "frequency " \
                         "(pixels) {}".format(self.params.filttype,
                                              self.params.filt_orientations[o],
                                              self.params.filt_stdev_pixels[f],)
                imaging.generate_image(self.filtresponses[o][f], _title,
                                       _filename, outdir)

                _filename = self.params.filttype + "_filterresponse_ap-{}-" \
                    "{}.png".format(self.params.filt_orientations[o],
                                    self.params.filt_stdev_pixels[f])
                _title = "Initial AP {} filter response: orientation {}, " \
                         "frequency " \
                         "(pixels) {}".format(self.params.filttype,
                                              self.params.filt_orientations[o],
                                              self.params.filt_stdev_pixels[f])
                imaging.generate_image(self.ap_filtresponses[o][f], _title,
                                       _filename, outdir)

    def find_diffs(self, results, results_dirs):
        """

        :param results: results of model applications to stimulus
        :type results: dict[str, numpy.core.multiarray.ndarray]
        :param results_dirs: model file directories for output of diff plot
        :type results_dirs: dict[str, str]
        :return: Dictionary of patch difference means for each model, each as a list [difference, mean_hi, mean_low,
            mean_background]
        :rtype: dict[str, float32]
        """
        img_mean = np.mean(self.img)

        diffs = {}
        for key in results:
            bg_region_mean = np.mean(results[key][self.regions == self.bg_region]) - img_mean
            hi_region_mean = np.mean(results[key][self.regions == self.high_region]) - img_mean
            lo_region_mean = np.mean(results[key][self.regions == self.low_region]) - img_mean

            diffs[key] = hi_region_mean - lo_region_mean

            # plot difference for this model
            outDir = results_dirs[key]
            filename = self.friendlyname + "-" + key + "-regionmeans.png"

            fig = plt.figure()
            ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
            ax.bar([-0.125, 1.0 - 0.125], [bg_region_mean, hi_region_mean, lo_region_mean], 0.25)
            ax.set_xticks([0, 1, 2])  # ticks for bg, hi, lo
            ax.set_xlim([-0.5, 2.5])
            ax.set_xticklabels(['BG', 'HI', 'LO'])
            plt.title(self.friendlyname + " - " + key)
            plt.savefig(outDir + filename)
            plt.close()

        return diffs


def _make_whites():
    """

    :return: img, regions, cutx, cuty, bg_region, low_region, high_region
    :rtype: (numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray,
             int, int, int, int, int)
    """

    # draw a gray background
    img = np.ones((1024, 1024)) * 0.5

    # make a white background for the stimulus itself
    img[271:271 + 480, 255:255 + 512] = 1.0
    patch_h = 96
    patch_w = 32

    lg_position = 1
    rg_position = 3

    for n in range(0, 8):
        # draw alternating black and white stripes
        img[271:271 + 480, 255 + (32 * n * 2):
            255 + (32 * n * 2) + 32] = 0.0

    # draw the left gray patch
    img[271 + patch_h * lg_position:271 + patch_h * (
        lg_position + 1), 255 + patch_w * 5 + 1:255 + patch_w *
        5 + 32] = 0.5

    # draw the right gray patch
    img[271 + patch_h * rg_position:271 + patch_h * (
        rg_position + 1), 255 + patch_w * 10 + 1:255 + patch_w *
        10 + 32] = 0.5

    # find and label regions set to gray in the image
    value = 0.5
    regions = _label_regions(img, value)

    # coordinates of cut lines through test patches
    cuty = 470
    cutx = 430

    # human illusion direction of regions
    bg_region = 1
    low_region = 2
    high_region = 3

    return img, regions, cutx, cuty, bg_region, low_region, high_region


def _make_howe():
    """

    :return: img, regions, cutx, cuty, bg_region, low_region, high_region
    :rtype: (numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray,
             int, int, int, int, int)
    """
    # draw a gray background
    img = np.ones((1024, 1024)) * 0.5

    # make a white background for the stimulus itself
    img[271:271 + 480, 255:255 + 512] = 1.0
    patch_h = 96
    patch_w = 32

    lg_position = 1
    rg_position = 3

    for n in range(0, 8):
        # draw 1st, 3rd, 5th row of alternating black and white stripes
        img[271:271 + patch_h, 255 + (patch_w * n * 2):
            255 + (patch_w * n * 2) + patch_w] = 0.0  # first row
        img[271 + patch_h * 2:271 + patch_h * 3, 255 + (patch_w * n * 2):
            255 + (patch_w * n * 2) + patch_w] = 0.0  # third row
        img[271 + patch_h * 4:271 + patch_h * 5, 255 + (patch_w * n * 2):
            255 + (patch_w * n * 2) + patch_w] = 0.0  # fifth row

    # draw 2nd row all black except 5th patch
    for n in range(0, 16):
        img[271 + patch_h:271 + patch_h * 2,
            255 + patch_w * n:255 + patch_w * n + patch_w] = 0.0

    # draw the 6th patch all white for all 5 rows
    img[271:271 + patch_h * 5,
        255 + patch_w * 5:255 + patch_w * 5 + patch_w] = 1.0

    # draw 11th patch all black for 5 rows
    img[271:271 + patch_h * 5,
        255 + patch_w * 10:255 + patch_w * 10 + patch_w] = 0.0

    # draw the left gray patch
    img[271 + patch_h * lg_position:271 + patch_h * (lg_position + 1),
        255 + patch_w * 5:255 + patch_w * 5 + patch_w] = 0.5

    # draw the right gray patch
    img[271 + patch_h * rg_position:271 + patch_h * (rg_position + 1),
        255 + patch_w * 10:255 + patch_w * 10 + patch_w] = 0.5

    # find and label regions set to gray in the image
    value = 0.5
    regions = _label_regions(img, value)

    # coordinates of cut lines through test patches
    cuty = 470
    cutx = 430

    # human illusion direction of regions
    bg_region = 1
    low_region = 2
    high_region = 3

    return img, regions, cutx, cuty, bg_region, low_region, high_region


def _make_howe_b():
    """

    :return: img, regions, cutx, cuty, bg_region, low_region, high_region
    :rtype: (numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray,
             int, int, int, int, int)
    """
    # draw a gray background
    img = np.ones((1024, 1024)) * 0.5

    # make a white background for the stimulus itself
    img[271:271 + 480, 255:255 + 512] = 1.0
    patch_h = 96
    patch_w = 32

    lg_position = 1
    rg_position = 3

    for n in range(0, 8):
        # draw 1st, 3rd, 5th row of alternating black and white stripes
        img[271:271 + patch_h, 255 + (patch_w * n * 2):
            255 + (patch_w * n * 2) + patch_w] = 0.0  # first row
        img[271 + patch_h * 2:271 + patch_h * 3, 255 + (patch_w * n * 2):
            255 + (patch_w * n * 2) + patch_w] = 0.0  # third row
        img[271 + patch_h * 4:271 + patch_h * 5, 255 + (patch_w * n * 2):
            255 + (patch_w * n * 2) + patch_w] = 0.0  # fifth row

    # draw 2nd row all black except 5th patch
    for n in range(0, 16):
        img[271 + patch_h+patch_h / 3:271+patch_h * 2 - patch_h / 3,
            255 + patch_w * n:255 + patch_w * n + patch_w] = 0.0

    # draw top and bottom of 2nd row with black and white stripes
    for n in range(0, 8):
        img[271:271 + patch_h + patch_h / 3, 255 + patch_w * n * 2:
            255 + patch_w * n * 2 + patch_w] = 0  # top
        img[271 + patch_h + patch_h * 2 / 3:271 + patch_h * 2,
            255 + patch_w * n * 2:255 + patch_w * n * 2 + patch_w] = 0  # btm

    # draw top and bottom of 4th row with black and white stripes
    for n in range(0, 8):
        img[271 + patch_h * 3: 271 + patch_h * 3 + patch_h / 3,
            255 + patch_w * n * 2: 255 + patch_w * n * 2 + patch_w] = 0  # top
        img[271 + patch_h * 3 + patch_h * 2 / 3:271 + patch_h * 4,
            255 + patch_w * n * 2:255 + patch_w * n * 2 + patch_w] = 0  # btm

    # draw the 6th patch all white for all 5 rows
    img[271:271 + patch_h * 5,
        255 + patch_w * 5:255 + patch_w * 5 + patch_w] = 1

    # draw 11th patch all black for 5 rows
    img[271:271 + patch_h * 5,
        255 + patch_w * 10:255 + patch_w * 10 + patch_w] = 0

    # draw the left gray patch
    img[271 + patch_h * lg_position:271 + patch_h * (lg_position + 1),
        255 + patch_w * 5:255 + patch_w * 5 + 32] = 0.5

    # draw the right gray patch
    img[271 + patch_h * rg_position:271 + patch_h * (
        rg_position + 1), 255 + patch_w * 10:255 + patch_w *
        10 + 32] = 0.5

    # find and label regions set to gray in the image
    value = 0.5
    regions = _label_regions(img, value)

    # coordinates of cut lines through test patches
    cuty = 470
    cutx = 430

    # human illusion direction of regions
    bg_region = 1
    low_region = 2
    high_region = 3

    return img, regions, cutx, cuty, bg_region, low_region, high_region


def _make_howe_d():
    """

    :return: img, regions, cutx, cuty, bg_region, low_region, high_region
    :rtype: (numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray,
             int, int, int, int, int)
    """
    # draw a gray background
    img = np.ones((1024, 1024)) * 0.5

    # make a white background for the stimulus itself
    img[271:271 + 480, 255:255 + 512] = 1.0
    patch_h = 96
    patch_w = 32

    lg_position = 1
    rg_position = 3

    for n in range(0, 8):
        # draw 1st, 3rd, 5th row of alternating black and white stripes
        img[271:271 + patch_h, 255 + (patch_w * n * 2):
            255 + (patch_w * n * 2) + patch_w] = 0.0  # first row
        img[271 + patch_h * 2:271 + patch_h * 3, 255 + (patch_w * n * 2):
            255 + (patch_w * n * 2) + patch_w] = 0.0  # third row
        img[271 + patch_h * 4:271 + patch_h * 5, 255 + (patch_w * n * 2):
            255 + (patch_w * n * 2) + patch_w] = 0.0  # fifth row

    # draw 2nd row all black except 5th patch
    for n in range(0, 16):
        img[271 + patch_h:271 + patch_h * 2,
            255 + patch_w * n:255 + patch_w * n + patch_w] = 0.0

    # draw the 6th patch all white for all 5 rows
    img[271:271 + patch_h * 5,
        255 + patch_w * 5:255 + patch_w * 5 + patch_w] = 1.0

    # draw 11th patch all black for 5 rows
    img[271:271 + patch_h * 5,
        255 + patch_w * 10:255 + patch_w * 10 + patch_w] = 0.0

    # draw top and bottom of 2nd row all black
    img[271 + patch_h * 2 / 3:271 + patch_h, 255:255 + 512] = 0  # top
    img[271 + patch_h * 2:271 + patch_h * 2 + patch_h * 1 / 3,
        255:255 + 512] = 0  # bottom

    # draw top and bottom of 4th row all white
    img[271 + patch_h * 2 + patch_h * 2 / 3:271 + patch_h * 2 + patch_h,
        255:255 + 512] = 1.0  # top
    img[271 + patch_h * 2 + patch_h * 2:271 + patch_h * 2 + patch_h * 2 +
        patch_h * 1 / 3, 255:255 + 512] = 1.0  # bottom

    # draw the left gray patch
    img[271 + patch_h * lg_position:271 + patch_h * (lg_position + 1),
        255 + patch_w * 5:255 + patch_w * 5 + patch_w] = 0.5

    # draw the right gray patch
    img[271 + patch_h * rg_position:271 + patch_h * (rg_position + 1),
        255 + patch_w * 10:255 + patch_w * 10 + patch_w] = 0.5

    # find and label regions set to gray in the image
    value = 0.5
    regions = _label_regions(img, value)

    # coordinates of cut lines through test patches
    cuty = 470
    cutx = 430

    # human illusion direction of regions
    bg_region = 1
    low_region = 3
    high_region = 2

    return img, regions, cutx, cuty, bg_region, low_region, high_region


def _make_sbc():
    """

    :return: img, regions, cutx, cuty, bg_region, low_region, high_region
    :rtype: (numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray,
             int, int, int, int, int)
    """
    # draw a gray background
    img = np.ones((1024, 1024)) * 0.5

    # make a white background for half of the stimulus
    img[271:271 + 480, 255:255 + 512] = 1.0

    # make a black background for the other half of the stimulus
    img[271:271 + 240, 255:255 + 512] = 0.0

    patch_h = 96
    patch_w = 32

    lg_position = 1
    rg_position = 3

    # draw the left gray patch
    img[271 + patch_h * lg_position:271 + patch_h * (
        lg_position + 1), 255 + patch_w * 5:255 + patch_w *
        5 + 32] = 0.5

    # draw the right gray patch
    img[271 + patch_h * rg_position:271 + patch_h * (
        rg_position + 1), 255 + patch_w * 10:255 + patch_w *
        10 + 32] = 0.5

    # find and label regions set to gray in the image
    value = 0.5
    regions = _label_regions(img, value)

    # coordinates of cut lines through test patches
    cuty = 470
    cutx = 430

    # human illusion direction of regions
    bg_region = 1
    low_region = 3
    high_region = 2

    return img, regions, cutx, cuty, bg_region, low_region, high_region


def _make_anderson():
    """

    :return: img, regions, cutx, cuty, bg_region, low_region, high_region
    :rtype: (numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray,
             int, int, int, int, int)
    """

    # draw a gray background
    img = np.ones((1024, 1024)) * 0.5

    # make a white background for the stimulus itself
    img[271:271 + 480, 255:255 + 512] = 1.0
    patch_h = 96
    patch_w = 32

    lg_position = 1
    rg_position = 3

    for n in range(0, 8):
        # draw 1st, 3rd, 5th row of alternating black and white stripes
        img[271:271 + patch_h, 255 + (patch_w * n * 2):
            255 + (patch_w * n * 2) + patch_w] = 0.0  # first row
        img[271 + patch_h * 2:271 + patch_h * 3, 255 + (patch_w * n * 2):
            255 + (patch_w * n * 2) + patch_w] = 0.0  # third row
        img[271 + patch_h * 4:271 + patch_h * 5, 255 + (patch_w * n * 2):
            255 + (patch_w * n * 2) + patch_w] = 0.0  # fifth row

    # draw 2nd row all black except 5th patch
    for n in range(0, 16):
        img[271 + patch_h:271 + patch_h * 2,
            255 + patch_w * n:255 + patch_w * n + patch_w] = 0.0

    # draw the 6th patch all white for all 5 rows
    img[271:271 + patch_h * 5,
        255 + patch_w * 5:255 + patch_w * 5 + patch_w] = 1.0

    # draw 11th patch all black for 5 rows
    img[271:271 + patch_h * 5,
        255 + patch_w * 10:255 + patch_w * 10 + patch_w] = 0.0

    # draw the left gray patch
    img[271 + patch_h * lg_position:271 + patch_h * (lg_position + 1),
        255 + patch_w * 5:255 + patch_w * 5 + patch_w] = 0.5

    # draw the right gray patch
    img[271 + patch_h * rg_position:271 + patch_h * (rg_position + 1),
        255 + patch_w * 10:255 + patch_w * 10 + patch_w] = 0.5

    # find and label regions set to gray in the image
    value = 0.5
    regions = _label_regions(img, value)

    # coordinates of cut lines through test patches
    cuty = 470
    cutx = 430

    # human illusion direction of regions
    bg_region = 1
    low_region = 2
    high_region = 3

    return img, regions, cutx, cuty, bg_region, low_region, high_region


def _make_rings():
    """

    :return: img, regions, cutx, cuty, bg_region, low_region, high_region
    :rtype: (numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray,
             int, int, int, int, int)
    """

    # set up the concentric circles
    m = 512
    n = 512
    m_center = 255
    n_center = 255
    bar_width = 32  # 1 degree
    start_r = 0

    # start with a background of a "non-color" value so it can easily be
    # filtered out later
    model = np.ones((m, n)) * -2.0

    # make a list of r's
    r = range(start_r, min(m - m_center, n - n_center) + 1, bar_width)

    # start with the outermost bar and work in
    last_circ = 1.0
    for i in range(len(r), 0, -1):
        hps = r[i-1]  # radius
        sqhps = hps * hps

        # make circle centered on model
        for p_y in range(-hps, hps + 1):
            for p_x in range(-hps, hps + 1):
                if (p_y - 0.5)**2 + (p_x - 0.5)**2 < sqhps:
                    if i == 6:
                        model[m_center + p_y, n_center + p_x] = 0.5
                    else:
                        model[m_center + p_y, n_center + p_x] = last_circ

        # swap last_bar between black and white
        if last_circ == 1.0:
            last_circ = 0.0
        else:
            last_circ = 1.0

    # create negative of stimulus
    neg_model = 1 - model

    # fix backgrounds for models
    model[model == -2.0] = 0.5
    neg_model[neg_model == 3.0] = 0.5

    # create a background then add the stimuli
    img = np.ones((1024, 1024)) * 0.5
    img[256:256 + 512, 0:512] = neg_model
    img[256:256 + 512, 512:] = model

    # find and label regions set to gray in the image
    value = 0.5
    regions = _label_regions(img, value)

    # coordinates of cut lines through test patches
    cuty = 512
    cutx = 256

    # human illusion direction of regions
    bg_region = 1
    low_region = 3
    high_region = 2

    return img, regions, cutx, cuty, bg_region, low_region, high_region


def _make_radial():
    """

    :return: img, regions, cutx, cuty, bg_region, low_region, high_region
    :rtype: (numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray,
             int, int, int, int, int)
    """

    # set up the params
    d = 512
    r = d / 2.0

    model = np.zeros((d, d))

    midpt = r
    rad = midpt
    radsq = midpt**2

    num_wedge = 26.0
    bw_wedge = 360 / (num_wedge / 2.0)
    single_wedge = 360 / num_wedge

    # patches should be 2 deg by 1 deg
    patch_width = 32
    patch_height = 64

    # (1/num_wedge)*2*pi*r*mid_point = patch_width
    mid_point = (patch_width * num_wedge) / (2 * math.pi * r)
    height_rad_fract = patch_height / r

    in_rad_sq = ((mid_point - height_rad_fract / 2) * rad)**2
    out_rad_sq = ((mid_point + height_rad_fract / 2) * rad)**2

    for i in range(d):
        for j in range(d):
            rvalsq = (i - midpt)**2 + (j - midpt)**2
            if rvalsq <= radsq:
                angle = math.atan2(i - midpt, j - midpt) * 180 / math.pi + 180
                if (angle % bw_wedge) < single_wedge:
                    model[i, j] = 1.0

                if in_rad_sq < rvalsq < out_rad_sq:
                    if 270 + single_wedge/2 >= angle >= 270 - single_wedge/2 \
                           or 90 - single_wedge/2 <= angle <= 90 + \
                           single_wedge/2:
                        model[i, j] = 0.5

            else:
                model[i, j] = 0.5

    # center model on full-size image
    img = np.ones((1024, 1024)) * 0.5
    img[256:256 + 512, 256:256 + 512] = model

    # find and label regions set to gray in the image
    value = 0.5
    regions = _label_regions(img, value)  # TODO: NOTE: in the original MATLAB
    # code, a bug caused region 3 to be named region 4 - watch for this

    # coordinates of cut lines through test patches
    cuty = 380
    cutx = 512

    # human illusion direction of regions
    bg_region = 1
    low_region = 3
    high_region = 2

    return img, regions, cutx, cuty, bg_region, low_region, high_region


def _make_zigzag():
    """

    :return: img, regions, cutx, cuty, bg_region, low_region, high_region
    :rtype: (numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray,
             int, int, int, int, int)
    """

    # zigzag image with the gray patches shifted 1/2
    y = 544  # length of image, y has to equal to x
    x = 544  # width of image
    width = 32  # width of the bars
    length = 128  # length of the bars

    square = np.ones((y, x))

    # calculate the starting point, start from bottom leftest corner
    strpt_v = y - width
    strpt_h = 0

    # priming these values to prevent compiler warnings later
    new_y = strpt_v
    new_x = x

    for j in range(101):
        # draw middle horizontal bars first
        # check for border conditions
        if strpt_v <= 1 and strpt_h + length >= x:
            square[0:strpt_v + width, strpt_h:x] = 0
            new_y = strpt_v
            new_x = x
            break
        else:
            if strpt_v <= 1 and strpt_h + length <= x:
                square[0:strpt_v + width, strpt_h:strpt_h + length] = 0
                new_y = strpt_v
                new_x = strpt_h + length - width
                break
            else:
                if strpt_v >= 1 and strpt_h + length >= x:
                    square[strpt_v:strpt_v + width, strpt_h:x] = 0
                    new_y = strpt_v
                    new_x = x
                    break

        square[strpt_v:strpt_v + width, strpt_h: strpt_h + length] = 0
        strpt_v -= 2 * width
        strpt_h += 2 * width

        # starting point for drawing upper bars
        newu_strpt_v = strpt_v - length + width
        newu_strpt_h = strpt_h - width

        # starting point for drawing lower bars
        newl_strpt_v = strpt_v
        newl_strpt_h = strpt_h + length - width

        # draw upper Zs
        for i in range(101):

            # draw upper vertical bars
            if i % 2 == 0:
                if newu_strpt_v >= 1 >= newu_strpt_h:
                    square[newu_strpt_v:newu_strpt_v + length,
                           0:newu_strpt_h + width] = 0
                    break
                else:
                    if newu_strpt_h >= 1 >= newu_strpt_v:
                        square[0:newu_strpt_v + length,
                               newu_strpt_h:newu_strpt_h + width] = 0
                        break
                    else:
                        if newu_strpt_v <= 1 and newu_strpt_h <= 1:
                            square[0:newu_strpt_v + length,
                                   0:newu_strpt_h + width] = 0
                            break
                        else:
                            square[newu_strpt_v:newu_strpt_v + length,
                                   newu_strpt_h:newu_strpt_h + width] = 0
                            newu_strpt_v = newu_strpt_v
                            newu_strpt_h = newu_strpt_h - length + width

            # draw upper horizontal bars
            if i % 2 > 0:
                if newu_strpt_h <= 1:
                    square[newu_strpt_v:newu_strpt_v + width,
                           0:newu_strpt_h + length - width] = 0
                    break
                else:
                    square[newu_strpt_v:newu_strpt_v + width,
                           newu_strpt_h:newu_strpt_h + length] = 0
                    newu_strpt_v = newu_strpt_v - length + width
                    newu_strpt_h -= width

        # draw lower Zs
        for k in range(101):
            # draw lower vertical bars
            if k % 2 == 0:
                if newl_strpt_v + length >= y and newl_strpt_h + width <= x:
                    square[newl_strpt_v:y,
                           newl_strpt_h + 0:newl_strpt_h + width] = 0
                    break
                else:
                    if newl_strpt_h + width >= x and \
                       newl_strpt_v + length <= y:
                        square[newl_strpt_v:newl_strpt_v + length,
                               newl_strpt_h:x] = 0
                        break
                    else:
                        if newl_strpt_h + width > x and \
                           newl_strpt_v + length > y:
                            square[newl_strpt_v:y, newl_strpt_h:x] = 0
                            break
                        else:
                            square[newl_strpt_v:newl_strpt_v + length,
                                   newl_strpt_h:newl_strpt_h + width] = 0
                            newl_strpt_v = newl_strpt_v + length - width
                            newl_strpt_h += width

            # draw lower horizontal bars
            if k % 2 > 0:
                if newl_strpt_h + length >= x:
                    square[newl_strpt_v:newl_strpt_v + width,
                           newl_strpt_h:x] = 0
                    break
                else:
                    square[newl_strpt_v:newl_strpt_v + width,
                           newl_strpt_h:newl_strpt_h + length] = 0
                    newl_strpt_v = newl_strpt_v
                    newl_strpt_h = newl_strpt_h + length - width

    # draw gray patches
    square[160 + 64:160 + 64 + 32 * 3, 160:160 + 32] = 0.5
    square[256 + 64:256 + 64 + 32 * 3, 320:320 + 32] = 0.5

    square[482 + 32:544, 448:544] = 0

    # crop image to make it symmetric
    model = np.copy(square[new_y:y, 0:new_x])

    # center model on full-size image
    location_index = ((1024 - model.shape[0]) / 2,
                      (1024 - model.shape[1]) / 2)
    img = np.ones((1024, 1024)) * 0.5
    img[location_index[0]:location_index[0] + model.shape[0],
        location_index[1]:location_index[1] + model.shape[1]] = model

    # find and label regions set to gray in the image
    value = 0.5
    regions = _label_regions(img, value)

    # coordinates of cut lines through test patches
    cuty = 496
    cutx = 704

    # human illusion direction of regions
    bg_region = 1
    low_region = 3
    high_region = 2

    return img, regions, cutx, cuty, bg_region, low_region, high_region


def _make_jacob_1():
    """

    :return: img, regions, cutx, cuty, bg_region, low_region, high_region
    :rtype: (numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray,
             int, int, int, int, int)
    """

    # make a white block
    block = np.ones((480, 576))

    # on the rows 1, 3 and 5 make horizontal blocks:
    # top row
    for i in range(1, 4):
        for j in range(1, 7):
            if (i + j) / 2.0 == math.floor((i + j) / 2.0):
                block[(i - 1) * 32: i * 32, (j - 1) * 96: j * 96] = 0

    # middle row
    for i in range(1, 4):
        for j in range(1, 7):
            if (i + j) / 2.0 == math.floor((i + j) / 2.0):
                block[2 * 96 + (i - 1) * 32: 2 * 96 + i * 32,
                      (j - 1) * 96: j * 96] = 0

    # bottom row
    for i in range(1, 4):
        for j in range(1, 7):
            if (i + j) / 2.0 == math.floor((i + j) / 2.0):
                block[4 * 96 + (i - 1) * 32: 4 * 96 + i * 32,
                      (j - 1) * 96: j * 96] = 0

    # on rows 2 and 4 make vertical blocks:
    # top row
    for i in range(1, 19):
        if i / 2.0 == math.floor(i / 2.0):
            block[96: 96 + 96, (i - 1) * 32: i * 32] = 0

    # middle row
    for i in range(1, 19):
        if i / 2.0 == math.floor(i / 2.0):
            block[3 * 96: 3 * 96 + 96, (i - 1) * 32: i * 32] = 0

    # make the grey patches

    block[96: 96 + 96, 32 * 5: 32 * 6] = 0.5
    block[96 * 3: 96 * 3 + 96, 32 * 12: 32 * 13] = 0.5

    # center model on full-size image
    location_index = ((1024 - block.shape[0]) / 2,
                      (1024 - block.shape[1]) / 2)
    img = np.ones((1024, 1024)) * 0.5
    img[location_index[0]:location_index[0] + block.shape[0],
        location_index[1]:location_index[1] + block.shape[1]] = block

    # find and label regions set to gray in the image
    value = 0.5
    regions = _label_regions(img, value)

    # coordinates of cut lines through test patches
    cuty = 470
    cutx = 430

    # human illusion direction of regions
    bg_region = 1
    low_region = 2
    high_region = 3

    return img, regions, cutx, cuty, bg_region, low_region, high_region


def _make_jacob_2():
    """

    :return: img, regions, cutx, cuty, bg_region, low_region, high_region
    :rtype: (numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray,
             int, int, int, int, int)
    """

    # make a white block
    block = np.ones((480, 576))

    # on the rows 1, 3 and 5 make horizontal blocks:
    # top row
    for i in range(1, 4):
        for j in range(1, 7):
            if (i + j) / 2.0 != math.floor((i + j) / 2.0):
                block[(i - 1) * 32: i * 32, (j - 1) * 96: j * 96] = 0

    # middle row
    for i in range(1, 4):
        for j in range(1, 7):
            if (i + j) / 2.0 != math.floor((i + j) / 2.0):
                block[2 * 96 + (i - 1) * 32: 2 * 96 + i * 32,
                      (j - 1) * 96: j * 96] = 0

    # bottom row
    for i in range(1, 4):
        for j in range(1, 7):
            if (i + j) / 2.0 != math.floor((i + j) / 2.0):
                block[4 * 96 + (i - 1) * 32: 4 * 96 + i * 32,
                      (j - 1) * 96: j * 96] = 0

    # on rows 2 and 4 make vertical blocks:
    # top row
    for i in range(1, 19):
        if i / 2.0 == math.floor(i / 2.0):
            block[6: 96 + 96, (i - 1) * 32: i * 32] = 0

    # middle row
    for i in range(1, 19):
        if i / 2.0 == math.floor(i / 2.0):
            block[3 * 96: 3 * 96 + 96, (i - 1) * 32: i * 32] = 0

    # make the grey patches

    block[96: 96 + 96, 32 * 5: 32 * 6] = 0.5
    block[96 * 3: 96 * 3 + 96, 32 * 12: 32 * 13] = 0.5

    # center model on full-size image
    location_index = ((1024 - block.shape[0]) / 2,
                      (1024 - block.shape[1]) / 2)
    img = np.ones((1024, 1024)) * 0.5
    img[location_index[0]:location_index[0] + block.shape[0],
        location_index[1]:location_index[1] + block.shape[1]] = block

    # find and label regions set to gray in the image
    value = 0.5
    regions = _label_regions(img, value)

    # coordinates of cut lines through test patches
    cuty = 470
    cutx = 430

    # human illusion direction of regions
    bg_region = 1
    low_region = 2
    high_region = 3

    return img, regions, cutx, cuty, bg_region, low_region, high_region


def _fix_variant_bg(img, top_rows, bottom_rows, left_cols, right_cols):
    """
    Since the variant stimuli change the default gray background to another
    color, this changes the background back so that the background does not
    significantly change the model output

    :param numpy.core.multiarray.ndarray img: stimulus image to be re-bordered
    :param int top_rows: number of rows at the top of the image to change
    :param int bottom_rows: number of rows at the bottom to change
    :param int left_cols: number of columns to the left of the image to change
    :param int right_cols: number of columns to the right to change
    :return: revised stimulus image
    :rtype: numpy.core.multiarray.ndarray
    """
    if top_rows > 0:
        img[0:top_rows, :] = 0.5
    if bottom_rows > 0:
        img[-bottom_rows:, :] = 0.5
    if left_cols > 0:
        img[:, 0:left_cols] = 0.5
    if right_cols > 0:
        img[:, -right_cols:] = 0.5
    return img


def _label_regions(img, value):
    """
    creates mask of regions where the image has given value, with regions
    individually numbered

    :param numpy.core.multiarray.ndarray img: image to be regioned
    :param float value: value to be found
    :return: map of regions
    :rtype: numpy.core.multiarray.ndarray
    """
    # start at region number 2 since the logical mask consists of zeros and
    # ones
    region_number = 2
    mask_size = img.shape

    mask = np.copy(img)
    mask[mask != value] = 0
    mask[mask == value] = 1

    # loop through the pixels looking for the value
    for i in range(mask_size[0]):
        for j in range(mask_size[1]):
            # if this region has not been set yet, grow the region and set it
            if mask[i, j] == 1:
                mask = _grow_region(i, j, mask, region_number)
                region_number += 1

    # reduce the region numbers by one
    mask[mask == 0] = 1
    mask -= 1

    return mask


def _grow_region(i_start, j_start, mask, reg_num):
    """
    sets all pixels in a contiguous area to reg_num value

    :param int i_start: i coordinate of region starting pixel
    :param int j_start: j coordinate of region starting pixel
    :param numpy.core.multiarray.ndarray mask: region mask to be updated
    :param float reg_num: value to set the region number to
    :return: region mask with contiguous region marked with the same value
    :rtype: numpy.core.multiarray.ndarray
    """
    # get mask size
    mask_size = mask.shape

    # value to set region finds to
    set_val = -1.0

    # set the current pixel
    mask[i_start, j_start] = set_val

    # loop through the mask looking for any pixel attached to the passed value
    for i in range(mask_size[0]):
        for j in range(mask_size[1]):
            if mask[i, j] != 0:
                # check to the right
                if j < (mask_size[1] - 1) and mask[i, j + 1] == set_val:
                    mask[i, j] = set_val
                # check above
                elif i > 1 and mask[i - 1, j] == set_val:
                    mask[i, j] = set_val
                # check left
                elif j > 1 and mask[i, j - 1] == set_val:
                    mask[i, j] = set_val
                # check below
                elif i < (mask_size[0] - 1) and mask[i + 1, j] == set_val:
                    mask[i, j] = set_val

        # loop through backwards in case anything was missed
        for j in range(mask_size[1] - 1, -1, -1):
            if mask[i, j] != 0:
                # check to the right
                if j < (mask_size[1] - 1) and mask[i, j + 1] == set_val:
                    mask[i, j] = set_val
                # check above
                elif i > 1 and mask[i - 1, j] == set_val:
                    mask[i, j] = set_val
                # check left
                elif j > 1 and mask[i, j - 1] == set_val:
                    mask[i, j] = set_val
                # check below
                elif i < (mask_size[0] - 1) and mask[i + 1, j] == set_val:
                    mask[i, j] = set_val

    # set the found pixels to the region number
    mask[mask == set_val] = reg_num

    return mask


def _gen_filter_response(params, img):
    """
    gets response of each filter to the stimulus

    :param par.FilterParams params: filter parameters and images
    :param numpy.core.multiarray.ndarray img: stimulus array
    :return: dictionaries of filter responses and AP filter responses
    :rtype: (dict[int, dict[int, numpy.core.multiarray.ndarray]],
        dict[int, dict[int, numpy.core.multiarray.ndarray]]
    """

    print("Getting responses of filters to stimulus...")

    # create a progress bar, since this will likely take a while
    pbar_maxval = len(params.filt_orientations) * len(params.filt_stdev_pixels)
    pbar = ProgressBar(widgets=[Percentage(), Bar()],
                       maxval=pbar_maxval).start()
    pbar_count = 1

    filtresponse = {}
    ap_filtresponse = {}
    for o in range(len(params.filt_orientations)):
        filtresponse[o] = {}
        ap_filtresponse[o] = {}
        for f in range(len(params.filt_stdev_pixels)):
            filt = params.filts[o][f]
            ap_filt = params.ap_filts[o][f]

            # since the filters use -1 to represent dark sensitivity and +1
            # to represent light sensitivity, their stimulus convolution
            # results in values ranging from -1 to 1, but this needs to be
            # re-ranged to between 0 and 1 to reflect that the responses are
            #  levels of response from the filter to the stimulus
            filtresponse[o][f] = (gpuf.normalized_conv(img, filt, 0.5) + 1.0) / 2.0
            ap_filtresponse[o][f] = (gpuf.normalized_conv(img, ap_filt, 0.5) + 1.0) / 2.0

            pbar.update(pbar_count)
            pbar_count += 1

    pbar.finish()

    return filtresponse, ap_filtresponse


def _normalized_conv_old(img, filt):
    """
    Performs FFT-based normalization on filter and image, with normalization so that a value of 1 in the response
    represents maximum possible response from filter to stimulus.

    :param numpy.core.multiarray.ndarray img: stimulus image to be convolved
    :param numpy.core.multiarray.ndarray filt: filter to convolve with
    :return: result of convolution
    :rtype: numpy.core.multiarray.ndarray
    """
    # get normalization info (convolving the filter with a matrix of 1s gives
    # us the max value of the convolution with that filter - dividing the end
    # convolution result by these max values gives us a normalized response)
    normimg = np.ones(filt.shape)
    filt = np.fliplr(np.flipud(filt))
    normtemp = ss.fftconvolve(normimg, filt, mode="same")

    return ss.fftconvolve(img, filt, mode="same") / normtemp


def _normalized_conv_nogpu(img, filt, padval):
    """
    Performs FFT-based normalization on filter and image, with normalization so that a value of 1 in the response
    represents maximum possible response from filter to stimulus.

    :param numpy.core.multiarray.ndarray img: stimulus image to be convolved
    :param numpy.core.multiarray.ndarray filt: filter to convolve with
    :param float padval: value with which to pad the img before convolution
    :return: result of convolution
    :rtype: numpy.core.multiarray.ndarray
    """
    # pad the images
    s_filt = filt.shape
    s_img = img.shape

    # appropriate padding depends on context
    pad_img = np.ones((s_img[0] + s_filt[0], s_img[1] + s_filt[1])) * padval

    pad_img[0: s_img[0], 0: s_img[1]] = img

    pad_filt = np.zeros((s_img[0] + s_filt[0], s_img[1] + s_filt[1]))

    pad_filt[0: s_filt[0], 0: s_filt[1]] = filt

    # get normalization info (convolving the filter with a matrix of 1s gives
    # us the max value of the convolution with that filter - dividing the end
    # convolution result by these max values gives us a normalized response)
    normimg = np.ones(pad_filt.shape)
    normtemp = (np.fft.ifft2(np.fft.fft2(np.absolute(pad_filt)) * np.fft.fft2(normimg))).real

    # Paul's slightly corrected version
    temp_out = (np.fft.ifft2(np.fft.fft2(pad_img) * np.fft.fft2(pad_filt))).real
    temp_out = temp_out / normtemp

    # extract the appropriate portion of the filtered image
    filtered = temp_out[(s_filt[0] / 2): -(s_filt[0] / 2), s_filt[1] / 2: -(s_filt[1] / 2)]

    return filtered




if __name__ == '__main__':
    import matplotlib.pyplot as plt

    filtparams = par.FilterParams(maindir="../")

    # for friendlyname in ["Whites", "Howe var B"]:
    for stimname in ["Whites", "Howe var B", "Howe var D", "Howe", "SBC",
                     "Anderson", "Rings", "Radial", "Zigzag", "Jacob 1",
                     "Jacob 2"]:

        temp = Stim(stimname, filtparams)

        tempfilename = temp.stimtype + temp.variant + ".png"
        outputdir = "../../experiments/output/"
        fig = plt.imshow(temp.img, interpolation="none", cmap="gray")
        plt.colorbar()
        plt.suptitle(tempfilename)
        plt.savefig(outputdir + tempfilename)
        plt.close()
