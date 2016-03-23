import math


# creating some structures to hold stuff
class _ParamStruct:
    pass


class Params:
    """
    holds various parameters for the model to use
    """

    def __init__(self):
        """
        nothing to do here?
        """

    # creating various structs to organize the variables:
    const = _ParamStruct()
    filt = _ParamStruct()
    filt.w = _ParamStruct()

    # ---constants---
    # number of degrees of visual angle per pixel
    const.DEG_PER_PIXEL = 0.03125

    # conversion factor
    const.SPACE_CONST_TO_STD = 1.0 / math.sqrt(2.0)

    # conversion factor
    const.SPACE_CONST_TO_WIDTH = 2.0 * math.sqrt(math.log(2.0))

    # number of pixels per degree of visual angle (automatically calculated
    # from DEG_PER_PIXEL)
    const.PIXELS_PER_DEG = 1/const.DEG_PER_PIXEL

    # additive constant for preventing divide by zero
    const.ADD_CONST = 10**(-12)

    # ---filter parameters---
    # whether or not to use normalized filters
    filt.norm = 0

    # orientations of the filters (6 steps of 30 degrees by default)
    filt.orientations = range(0, 179, 30)

    # spatial frequencies of the filters
    filt.freqs = range(0, 6, 1)

    # standard deviation of different frequency filters in pixels
    filt.stdev_pixels = []

    # whether or not to use filter weighting
    filt.w.use = 1

    # array of weight values per frequency
    filt.w.val = []

    # slope of weighting function
    filt.w.valslope = 0.1

    # extent of filter in x
    filt.x = 1024

    # extent of filter in y
    filt.y = 1024

    # holds file and directory friendly name of filter normalization model
    # TODO: move this to model class?
    filt.label = []

    # following values are used in gen_ODOG
    filt.stretchWidth = 1.0        # width of DOG, == 1 in ODOG, > 1 makes
    # filter more Gabor-like.
    filt.negwidth = 1.0     # std ratio of negative surround to center,
    # == 1 in ODOG
    filt.neglen = 2.0       # std ratio of negative surround to center,
    # == 2 in ODOG
    filt.centerW = 1.0      # weight on center gaussian, >1 = positive sum
    # gaussian. == 1 in ODOG

    # ---calculate values for filt.stdev_pixels and filt.w
    # compute the standard deviations of the different Gaussian in pixels
    space_const = [2.0**x * 1.5 for x in filt.freqs]  # space constant of
    # Gaussians

    filt.stdev_pixels = [x * const.SPACE_CONST_TO_STD for x in space_const]
    # in pixels

    # matches Table 1 in BM(1999)
    space_const_deg = [x * const.DEG_PER_PIXEL for x in space_const]  # in deg.

    # (almost matches) points along x-axis of Fig. 10 BM(1997)
    cpd = [1.0 / (2.0 * x * const.SPACE_CONST_TO_WIDTH) for x in
           space_const_deg]

    # (almost matches) points along y-axis of Fig. 10 BM(1997)
    filt.w.val = [x**filt.w.valslope for x in cpd]


if __name__ == "__main__":
    params = Params()
    print("class instance created, doing nothing else")
