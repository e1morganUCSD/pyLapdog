import numpy as np
import math
import scipy.ndimage as ndi
import scipy.signal as ss
import filtergen as fg
from progressbar import ProgressBar, Percentage, Bar
import time
import cPickle as pickle


def _compare_filters(pre_filter, post_filter):
    """
    helper function for generate_correlation_mask below using non-FFT
    convolution

    At each point of the receptive field of a postsynaptic filter
    (post_filter), calculates level of same-phase correlation of
    receptive fields between a presynaptic_filter (pre_filter) centered at that
    point and the postsynaptic filter, using the formula:

    c'(a,b) = sum of all a_i * b_j where a_i and b_j are points that
    spatially overlap in filters a and b
    c(a,b) = c'(a,b)/sqrt(c'(a,a), c'(b,b))

    :type post_filter: numpy.ndarray(float32)
    :type pre_filter: numpy.ndarray(float32)
    :param pre_filter: numpy.ndarray
    ;param post_filter: numpy.ndarray
    :return: numpy.ndarray
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

    pre_filter = np.float32(pre_filter)
    post_filter = np.float32(post_filter)

    return ndi.convolve(post_filter, pre_filter, mode='constant', cval=0.0) \
           / correlation_denominator


def _compare_filters_fft(pre_filter, post_filter):
    """
    helper function for generate_correlation_mask below, using FFT convolution

    At each point of the receptive field of a postsynaptic filter
    (post_filter), calculates level of same-phase correlation of
    receptive fields between a presynaptic_filter (pre_filter) centered at that
    point and the postsynaptic filter, using the formula:

    c'(a,b) = sum of all a_i * b_j where a_i and b_j are points that
    spatially overlap in filters a and b
    c(a,b) = c'(a,b)/sqrt(c'(a,a), c'(b,b))

    :type post_filter: numpy.ndarray(float32)
    :type pre_filter: numpy.ndarray(float32)
    :param pre_filter: numpy.ndarray
    ;param post_filter: numpy.ndarray
    :return: numpy.ndarray
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

    pre_filter = np.float32(pre_filter)
    post_filter = np.float32(post_filter)

    return ss.fftconvolve(post_filter, pre_filter, mode="same") \
           / correlation_denominator


def generate_correlation_mask(params):
    """
    Generates array of correlation values between two filters (as described
    in Troyer et al.)
    ELEMENTWISE VERSION - USES HUGE AMOUNTS OF MEMORY (>20GB)

    :rtype: dict
    :param params: contrastmodel.params.Params
    :return: numpy.ndarray
    """

    # pull out useful information from params
    orientations = params.filt.orientations
    stdev_pixels = params.filt.stdev_pixels

    # create cell arrays to put things into
    # filtermasks = np.zeros((len(orientations), len(stdev_pixels),
    #                         len(orientations), len(stdev_pixels),
    #                         params.filt.x, params.filt.y))
    # ap_filtermasks = np.zeros((len(orientations), len(stdev_pixels),
    #                            len(orientations), len(stdev_pixels),
    #                            params.filt.x, params.filt.y))

    # create dictionary to hold filtermasks for output
    filtermasks = {}
    ap_filtermasks = {}

    # create a progress bar, since this will likely take a while
    pbar_maxval = len(orientations)**2 * len(stdev_pixels)**2
    pbar = ProgressBar(widgets=[Percentage(), Bar()],
                       maxval=pbar_maxval).start()
    pbar_count = 1

    for o in range(len(orientations)):
        # print("{} {}".format("Now calculating orientation", o))
        filtermasks[o] = {}
        ap_filtermasks[o] = {}
        for f in range(len(stdev_pixels)):
            filtermasks[o][f] = {}
            ap_filtermasks[o][f] = {}
            post_filter = fg.gen_ODOG(params.filt.y, params.filt.x,
                                      stdev_pixels[f], stdev_pixels[f],
                                      params.filt.negwidth,
                                      params.filt.neglen, orientations[o] *
                                      (math.pi/180), params.filt.centerW)

            # trim filter to reduce processing of unnecessary near-zero values
            post_filter = fg.trimfilt(post_filter, np.max(np.abs(
                post_filter)) * 0.01)

            for o2 in range(len(orientations)):
                # print("{} {}-{}-{}".format("Now calculating orientation",
                # o, f, o2))
                filtermasks[o][f][o2] = {}
                ap_filtermasks[o][f][o2] = {}
                for f2 in range(len(stdev_pixels)):
                    pre_filter = fg.gen_ODOG(params.filt.y, params.filt.x,
                                             stdev_pixels[f2], stdev_pixels[f2],
                                             params.filt.negwidth,
                                             params.filt.neglen, orientations[o2] *
                                             (math.pi/180),
                                             params.filt.centerW)

                    # trim filter to reduce processing of near-zero values
                    pre_filter = fg.trimfilt(pre_filter, np.max(np.abs(
                        pre_filter)) * 0.01)

                    filtcomparison = _compare_filters(pre_filter, post_filter)
                    filtermasks[o][f][o2][f2] = filtcomparison
                    ap_filtermasks[o][f][o2][f2] = -filtcomparison

                    pbar.update(pbar_count)
                    pbar_count += 1

    pbar.finish()

    pickle.dump(filtermasks, open("filtermasks.pkl", mode='wb'), protocol=2)
    pickle.dump(ap_filtermasks, open("ap_filtermasks.pkl", mode='wb'), protocol=2)
    # np.save("filtermasks_nonFFT.npy", filtermasks)
    # np.save("ap_filtermasks_nonFFT.npy", ap_filtermasks)

    return filtermasks, ap_filtermasks


def generate_correlation_mask_fft(params):
    """
    Generates array of correlation values between two filters (as described
    in Troyer et al.)
    FFT VERSION - USES THIS ONE

    :rtype: dict
    :param params: contrastmodel.params.Params
    :return: numpy.ndarray
    """

    # pull out useful information from params
    orientations = params.filt.orientations
    stdev_pixels = params.filt.stdev_pixels

    # create cell arrays to put things into
    # filtermasks = np.zeros((len(orientations), len(stdev_pixels),
    #                         len(orientations), len(stdev_pixels),
    #                         params.filt.x, params.filt.y))
    # ap_filtermasks = np.zeros((len(orientations), len(stdev_pixels),
    #                            len(orientations), len(stdev_pixels),
    #                            params.filt.x, params.filt.y))

    # create dictionary to hold filtermasks for output
    filtermasks = {}
    ap_filtermasks = {}

    # create a progress bar, since this will likely take a while
    pbar_maxval = len(orientations)**2 * len(stdev_pixels)**2
    pbar = ProgressBar(widgets=[Percentage(), Bar()],
                       maxval=pbar_maxval).start()
    pbar_count = 1

    for o in range(len(orientations)):
        # print("{} {}".format("Now calculating orientation", o))
        filtermasks[o] = {}
        ap_filtermasks[o] = {}
        for f in range(len(stdev_pixels)):
            filtermasks[o][f] = {}
            ap_filtermasks[o][f] = {}
            post_filter = fg.gen_ODOG(params.filt.y, params.filt.x,
                                      stdev_pixels[f], stdev_pixels[f],
                                      params.filt.negwidth,
                                      params.filt.neglen, orientations[o] *
                                      (math.pi/180), params.filt.centerW)

            # no filter trimming for FFT version
            # # trim filter to reduce processing of unnecessary near-zero values
            # post_filter = fg.trimfilt(post_filter, np.max(np.abs(
            #     post_filter)) * 0.01)

            for o2 in range(len(orientations)):
                # print("{} {}-{}-{}".format("Now calculating orientation",
                # o, f, o2))
                filtermasks[o][f][o2] = {}
                ap_filtermasks[o][f][o2] = {}
                for f2 in range(len(stdev_pixels)):
                    pre_filter = fg.gen_ODOG(params.filt.y, params.filt.x,
                                             stdev_pixels[f2], stdev_pixels[f2],
                                             params.filt.negwidth,
                                             params.filt.neglen, orientations[o2] *
                                             (math.pi/180),
                                             params.filt.centerW)

                    # no filter trimming for FFT version
                    # # trim filter to reduce processing of near-zero values
                    # pre_filter = fg.trimfilt(pre_filter, np.max(np.abs(
                    #     pre_filter)) * 0.01)

                    filtcomparison = _compare_filters_fft(pre_filter,
                                                          post_filter)

                    # trim small edge values from filtermask - these
                    # represent low-probability connections
                    filtcomparison = fg.trimfilt(filtcomparison, np.max(np.abs(
                        filtcomparison)) * 0.01)
                    filtermasks[o][f][o2][f2] = filtcomparison
                    ap_filtermasks[o][f][o2][f2] = -filtcomparison

                    pbar.update(pbar_count)
                    pbar_count += 1

    pbar.finish()

    pickle.dump(filtermasks, open("filtermasks_FFT.pkl", mode='wb'),
                protocol=2)
    pickle.dump(ap_filtermasks, open("ap_filtermasks_FFT.pkl", mode='wb'),
                protocol=2)
    # np.save("filtermasks_FFT.npy", filtermasks)
    # np.save("ap_filtermasks_FFT.npy", ap_filtermasks)

    return filtermasks, ap_filtermasks



# ---TESTING---

if __name__ == "__main__":
    import contrastmodel.params.paramsDef as par

    params = par.Params()

    # reduce the number of options so it processes more quickly
    params.filt.orientations = range(0, 89, 30)
    params.filt.stdev_pixels = [4.0, 8.0, 16.0]

    start = time.time()
    filtermasks, ap_filtermasks = generate_correlation_mask_fft(params)

    end = time.time()
    print("elapsed time:")
    print(end - start)

    test = filtermasks[1][2][2][1]
    import matplotlib.pyplot as plt
    fig = plt.imshow(test)
    plt.colorbar()
    plt.show()




