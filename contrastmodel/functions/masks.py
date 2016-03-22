import numpy as np
import math
import scipy.ndimage as ndi
from progressbar import ProgressBar, Percentage, Bar



def _compare_filters(pre_filter, post_filter):
    """
    helper function for generate_correlation_mask below

    At each point of the receptive field of a postsynaptic filter
    (post_filter), calculates level of same-phase correlation of
    receptive fields between a presynaptic_filter (pre_filter) centered at that
    point and the postsynaptic filter, using the formula:

    c'(a,b) = sum of all a_i * b_j where a_i and b_j are points that
    spatially overlap in filters a and b
    c(a,b) = c'(a,b)/sqrt(c'(a,a), c'(b,b))

    :param pre_filter: np.array
    ;param post_filter: np.array
    :return: np.array
    """

    # # create output matrices to hold results
    # filtcomparison = np.zeros(pre_filter.shape[0] + post_filter.shape[0],
    #                           pre_filter.shape[1] + post_filter.shape[1])
    #
    # antiphase_filtcomparison = np.zeros(filtcomparison.shape)

    pre_filter_selfcomp = np.sum(pre_filter * pre_filter)
    post_filter_selfcomp = np.sum(post_filter * post_filter)

    correlation_denominator = math.sqrt(pre_filter_selfcomp *
                                        post_filter_selfcomp)

    # since scipy.ndimage.convolve slides a backwards filter over an image,
    # we need to flip the pre_filter
    pre_filter = np.fliplr(pre_filter)
    pre_filter = np.flipud(pre_filter)

    return ndi.convolve(post_filter, pre_filter, mode='constant', cval=0.0)


def generate_correlation_mask(params):
    """
    Generates array of correlation values between two filters (as described
    in Troyer et al)

    :param params: contrastmodel.params.Params
    :return: np.array
    """

    # pull out useful information from params
    orientations = params.filt.orientations
    stdev_pixels = params.filt.stdev_pixels

    # create cell arrays to put things into
    filtermasks = np.zeros((len(orientations), len(stdev_pixels),
                            len(orientations), len(stdev_pixels)))
    ap_filtermasks = np.zeros((len(orientations), len(stdev_pixels),
                               len(orientations), len(stdev_pixels)))

    # create a progress bar, since this will likely take a while
    pbar_maxval = len(orientations)**2 * len(stdev_pixels)**2
    pbar = ProgressBar(widgets=[Percentage(), Bar()],
                       maxval=pbar_maxval).start()

    for o in orientations:
        for f in stdev_pixels:
            post_filter = 3  # TODO: replace 3 with ODOG generation function
