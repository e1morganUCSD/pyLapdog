"""
This modules contains the class definition for the parameters class used to
hold filter and model parameters
"""

import math
import contrastmodel.functions.filtergen as fg
import contrastmodel.functions.imaging as imaging
import os


class FilterParams:
    """
    holds various parameters for the model to use
    """

    def __init__(self, maindir="", filttype="odog", verbosity=1):
        """

        :param str maindir: main output directory, defaults to
            "~/pyLapdog/output"
        :param str filttype: type of filter - "odog" for ODOG filters, "gabor"
            for Gabor filters
        :param int verbosity: level of output generation - 1 only outputs
            final results, 2 outputs midlevel processing info, 3 outputs
            everything
        :rtype: FilterParams
        """

        if maindir == "":
            # output directory (ending in slash)
            self.mainDir = os.path.expanduser("~pyLapdog/output/")
        else:
            self.mainDir = maindir
            if "~" in self.mainDir:
                self.mainDir = os.path.expanduser(self.mainDir)

        self.filttype = filttype

        # ---constants---
        # number of degrees of visual angle per pixel
        self.const_DEG_PER_PIXEL = 0.03125

        # number of pixels per degree of visual angle (automatically calculated
        # from DEG_PER_PIXEL)
        self.const_PIXELS_PER_DEG = 1 / self.const_DEG_PER_PIXEL

        # ---filter parameters---
        # whether or not to use normalized filters
        self.filt_norm = 0

        # orientations of the filters (6 steps of 30 degrees by default)
        self.filt_orientations = range(0, 179, 30)

        # spatial frequencies of the filters
        self.filt_freqs = range(0, 6, 1)

        # whether or not to use filter weighting
        self.filt_w_use = 1

        # array of weight values per frequency
        self.filt_w_val = []

        # slope of weighting function
        self.filt_w_valslope = 0.1

        # extent of filter in rows
        self.filt_rows = 1024

        # extent of filter in columns
        self.filt_cols = 1024

        # save verbosity level
        self.verbosity = verbosity

        if filttype == "odog":
            # conversion factor
            self.const_SPACE_CONST_TO_STD = 1.0 / math.sqrt(2.0)

            # conversion factor
            self.const_SPACE_CONST_TO_WIDTH = 2.0 * math.sqrt(math.log(2.0))

            # additive constant for preventing divide by zero
            self.const_ADD_CONST = 10**(-12)

            # standard deviation of different frequency filters in pixels
            self.filt_stdev_pixels = []

            # holds file and directory friendly name of
            # filter normalization model
            # TODO: move this to model class?
            self.filt_label = []

            # following values are used in gen_odog
            self.filt_stretchWidth = 1.0        # width of DOG, == 1 in ODOG,
            # > 1 makes
            # filter more Gabor-like.
            self.filt_negwidth = 1.0    # std ratio of neg surround to center,
            # == 1 in ODOG
            self.filt_neglen = 2.0      # std ratio of neg surround to center,
            # == 2 in ODOG
            self.filt_centerW = 1.0  # weight on center gaussian, >1 =
            # positive sum gaussian. == 1 in ODOG

            # ---calculate values for filt.stdev_pixels and filt.w
            # compute the standard deviations of the different Gaussian
            # in pixels
            self.space_const = [2.0**x * 1.5 for x in self.filt_freqs]  # space
            # constant of Gaussians

            self.filt_stdev_pixels = [x * self.const_SPACE_CONST_TO_STD for
                                      x in self.space_const]  # in pixels

            # matches Table 1 in BM(1999)
            self.space_const_deg = [x * self.const_DEG_PER_PIXEL
                                    for x in self.space_const]  # in deg.

            # (almost matches) points along x-axis of Fig. 10 BM(1997)
            self.cpd = [1.0 / (2.0 * x * self.const_SPACE_CONST_TO_WIDTH) for
                        x in self.space_const_deg]

            # (almost matches) points along y-axis of Fig. 10 BM(1997)
            self.filt_w_val = [x**self.filt_w_valslope for x in self.cpd]

            # generate filters
            self.filts, self.ap_filts = self._gen_filters_odog()

            if self.verbosity == 3:
                self._print_filts()

    def _gen_filters_odog(self):
        """

        :return: Dictionary of ODOG filters, indexed by [o][f], where o =
            filter orientation, f = filter frequency
        :rtype: (dict[int, dict[int, numpy.core.multiarray.ndarray]],
                 dict[int, dict[int, numpy.core.multiarray.ndarray]])
        """
        # generate output directories
        filts = {}
        ap_filts = {}

        print("Generating filters...")

        for o in range(len(self.filt_orientations)):
            filts[o] = {}
            ap_filts[o] = {}
            for f in range(len(self.filt_stdev_pixels)):
                filt = fg.gen_odog(self.filt_rows, self.filt_cols,
                                   self.filt_stdev_pixels[f],
                                   self.filt_stdev_pixels[f],
                                   self.filt_negwidth, self.filt_neglen,
                                   self.filt_orientations[o] * math.pi / 180.0,
                                   self.filt_centerW)
                ap_filt = filt * -1.0
                filts[o][f] = filt
                ap_filts[o][f] = ap_filt

        return filts, ap_filts

    def _print_filts(self):
        """
        prints filter figures

        """
        print("Generating filter files in " + self.mainDir)
        # make sure output dir exists
        if not os.path.exists(self.mainDir):
            os.makedirs(self.mainDir)

        for o in range(len(self.filt_orientations)):
            for f in range(len(self.filt_stdev_pixels)):
                title = "Initial ODOG filter: orientation {}, " \
                        "frequency (pixels) {}".format(o, f)
                filename = "ODOGfilter-{}-{}.png".format(o, f)
                imaging.generate_image(self.filts[o][f], title, filename,
                                       self.mainDir)

                title = "Initial ODOG AP filter: orientation {}, " \
                        "frequency (pixels) {}".format(o, f)
                filename = "ODOGfilter_ap-{}-{}.png".format(o, f)
                imaging.generate_image(self.ap_filts[o][f], title, filename,
                                       self.mainDir)

if __name__ == "__main__":
    if os.name == "nt":
        params = FilterParams(verbosity=3,
                              maindir="c:\\Users\\Eric\\pylapdog\\output\\")
    else:
        params = FilterParams(verbosity=3, maindir="~/pyLapdog/output/")
    print("class instance created, doing nothing else")
