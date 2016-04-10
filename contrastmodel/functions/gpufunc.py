"""
functions designed to run on GPU
"""

from accelerate.cuda.fft import FFTPlan, fft_inplace, ifft_inplace, fft, ifft
from numba import cuda, vectorize
import numpy as np
import scipy.signal as ss


@vectorize(['complex64(complex64, complex64)'], target='cuda')
def vmult(a, b):
    """
    vectorized helper function for multiplying a * b

    :param complex a:
    :param complex b:
    :rtype complex
    """
    # return (a.real * b.real) + 0j
    return a * b


@vectorize(['complex64(complex64, complex64)'], target='cuda')
def vdiv(a, b):
    """
    vectorized helper function for dividing a / b

    :param complex a:
    :param complex b:
    :rtype complex
    """
    #return (a.real / b.real) + 0j
    return a / b


def normalized_conv(img, filt, padval):
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
    pad_img = np.ones((s_img[0] + s_filt[0], s_img[1] + s_filt[1]), dtype=np.float32) * padval

    pad_img[0: s_img[0], 0: s_img[1]] = img

    pad_filt = np.zeros((s_img[0] + s_filt[0], s_img[1] + s_filt[1]), dtype=np.float32)

    pad_filt[0: s_filt[0], 0: s_filt[1]] = filt

    # initialize the GPU
    FFTPlan(shape=pad_img.shape, itype=np.complex64, otype=np.complex64)

    # get normalization info (convolving the filter with a matrix of 1s gives
    # us the max value of the convolution with that filter - dividing the end
    # convolution result by these max values gives us a normalized response)
    normimg = np.ones(pad_filt.shape, dtype=np.complex64)
    abs_pad_filt = np.absolute(pad_filt).astype(np.complex64)
    normtemp1 = np.zeros(pad_filt.shape, dtype=np.complex64)
    normtemp2 = np.zeros(pad_filt.shape, dtype=np.complex64)
    normtemp3 = np.zeros(pad_filt.shape, dtype=np.complex64)

    # transfer data to GPU
    d_normimg = cuda.to_device(normimg)
    d_abs_pad_filt = cuda.to_device(abs_pad_filt)
    d_normtemp1 = cuda.to_device(normtemp1)
    d_normtemp2 = cuda.to_device(normtemp2)
    d_normtemp3 = cuda.to_device(normtemp3)
    fft(d_normimg, d_normtemp1)
    fft(d_abs_pad_filt, d_normtemp2)
    vmult(d_normtemp1, d_normtemp2, out=d_normtemp1)
    ifft(d_normtemp1, d_normtemp2)
    # normtemp = (cuda.fft.ifft_inplace(cuda.fft.fft_inplace(np.absolute(pad_filt)) * cuda.fft.fft_inplace(
    # normimg))).real

    d_pad_filt = cuda.to_device(pad_filt.astype(np.complex64))
    d_pad_img = cuda.to_device(pad_img.astype(np.complex64))
    fft(d_pad_filt, d_normtemp1)
    fft(d_pad_img, d_normtemp3)
    vmult(d_normtemp1, d_normtemp3, out=d_normtemp1)
    ifft(d_normtemp1, d_normtemp3)
    # temp_out = (cuda.fft.ifft_inplace(cuda.fft.fft_inplace(pad_img)) * cuda.fft.fft_inplace(pad_filt)).real
    vdiv(d_normtemp3, d_normtemp2, out=d_normtemp3)

    temp_out = d_normtemp3.copy_to_host().real

    # extract the appropriate portion of the filtered image
    filtered = temp_out[(s_filt[0] / 2): (s_filt[0] / 2) + s_img[0], (s_filt[1] / 2): (s_filt[1] / 2) + s_img[1]]

    return filtered


def our_conv(img, filt, padval):
    """
    Performs FFT-based normalization on filter and image, without normalization

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

    # initialize the GPU
    FFTPlan(shape=pad_img.shape, itype=np.complex64, otype=np.complex64)

    # create temporary arrays for holding FFT values
    normtemp1 = np.zeros(pad_img.shape, dtype=np.complex64)
    normtemp2 = np.zeros(pad_img.shape, dtype=np.complex64)

    d_pad_filt = cuda.to_device(pad_filt.astype(np.complex64))
    d_pad_img = cuda.to_device(pad_img.astype(np.complex64))
    d_normtemp1 = cuda.to_device(normtemp1)
    d_normtemp2 = cuda.to_device(normtemp2)

    fft(d_pad_filt, d_normtemp1)
    fft(d_pad_img, d_normtemp2)
    vmult(d_normtemp1, d_normtemp2, out=d_normtemp1)
    ifft(d_normtemp1, d_normtemp2)
    # temp_out = (cuda.fft.ifft_inplace(cuda.fft.fft_inplace(pad_img)) * cuda.fft.fft_inplace(pad_filt)).real
    temp_out = d_normtemp2.copy_to_host().real

    # extract the appropriate portion of the filtered image
    filtered = temp_out[(s_filt[0] / 2): (s_filt[0] / 2) + s_img[0], (s_filt[1] / 2): (s_filt[1] / 2) + s_img[1]]

    return filtered


def lapconv(img, filt, padval):
    """
    Performs FFT-based normalization on filter and image, without normalization

    :param numpy.core.multiarray.ndarray img: stimulus image to be convolved
    :param numpy.core.multiarray.ndarray filt: filter to convolve with
    :param float padval: value with which to pad the img before convolution
    :return: result of convolution
    :rtype: numpy.core.multiarray.ndarray
    """

    # get the number of nonzero entries in the filter for later dividing of the results
    filt_nnz = np.count_nonzero(filt)

    # pad the images
    s_filt = filt.shape
    s_img = img.shape

    # appropriate padding depends on context
    pad_img = np.ones((s_img[0] + s_filt[0], s_img[1] + s_filt[1])) * padval

    pad_img[0: s_img[0], 0: s_img[1]] = img

    pad_filt = np.zeros((s_img[0] + s_filt[0], s_img[1] + s_filt[1]))

    pad_filt[0: s_filt[0], 0: s_filt[1]] = filt

    # initialize the GPU
    FFTPlan(shape=pad_img.shape, itype=np.complex64, otype=np.complex64)

    # create temporary arrays for holding FFT values
    normtemp1 = np.zeros(pad_img.shape, dtype=np.complex64)
    normtemp2 = np.zeros(pad_img.shape, dtype=np.complex64)

    d_pad_filt = cuda.to_device(pad_filt.astype(np.complex64))
    d_pad_img = cuda.to_device(pad_img.astype(np.complex64))
    d_normtemp1 = cuda.to_device(normtemp1)
    d_normtemp2 = cuda.to_device(normtemp2)

    fft(d_pad_filt, d_normtemp1)
    fft(d_pad_img, d_normtemp2)
    vmult(d_normtemp1, d_normtemp2, out=d_normtemp1)
    ifft(d_normtemp1, d_normtemp2)
    # temp_out = (cuda.fft.ifft_inplace(cuda.fft.fft_inplace(pad_img)) * cuda.fft.fft_inplace(pad_filt)).real
    temp_out = d_normtemp2.copy_to_host().real

    # extract the appropriate portion of the filtered image
    filtered = temp_out[(s_filt[0] / 2): (s_filt[0] / 2) + s_img[0], (s_filt[1] / 2): (s_filt[1] / 2) + s_img[1]]

    # divide each value by the number of nonzero entries in the filter (and image?!?), so we get an average of all the
    # values
    filtered /= (filt_nnz * s_img[0] * s_img[1])

    return filtered


def stupidconv(img, filt, padval):
    """
    does convolution without using FFT because FFT is pissing me off and giving me weird answers
    :param img:
    :param filt:
    :param padval:
    :return:
    """

    # get the number of nonzero entries in the filter for later averaging of result
    filt_nnz = np.count_nonzero(filt)

    # pad the images
    s_filt = filt.shape
    s_img = img.shape

    # appropriate padding depends on context
    # pad with filt size all around img
    pad_img = np.ones((s_img[0] + (2 * s_filt[0]), s_img[1] + (2 * s_filt[1]))) * padval

    pad_img[s_filt[0]: s_img[0] + s_filt[0], s_filt[1]: s_img[1] + s_filt[1]] = img

    output = np.zeros(pad_img.shape)

    for row in range(s_filt[0] + s_img[0]):
        for col in range(s_filt[1] + s_img[1]):
            # total = 0.0
            # for filtrow in range(s_filt[0]):
            #     for filtcol in range(s_filt[1]):
            #         total = total + (filt[filtrow, filtcol] * pad_img[row + filtrow, col + filtcol])
            output[row, col] = np.sum(filt * pad_img[row:row + s_filt[0], col:col + s_filt[1]])

    output = output[s_filt[0]:s_filt[0] + s_img[0], s_filt[1]:s_filt[1] + s_img[1]]

    return output / filt_nnz


@cuda.jit('void(float32[:,:], float32[:,:], int32, int32, int32, int32, float32[:,:])')
def stupidconv_gpu_helper(pad_img, filt, img_rows, img_cols, filt_rows, filt_cols, d_output):

    for row in range(filt_rows + img_rows):
        for col in range(filt_cols + img_cols):
            total = 0.0
            for filtrow in range(filt_rows):
                for filtcol in range(filt_cols):
                    total = total + (filt[filtrow, filtcol] * pad_img[row + filtrow, col + filtcol])
            d_output[row, col] = total


def stupidconv_gpu(img, filt, padval):
    """
    does convolution without using FFT because FFT is pissing me off and giving me weird answers
    :param img:
    :param filt:
    :param padval:
    :return:
    """
    cuda.close()
    cuda.select_device(1)
    # get the number of nonzero entries in the filter for later averaging of result
    filt_nnz = np.count_nonzero(filt)

    # pad the images
    s_filt = filt.shape
    s_img = img.shape

    # appropriate padding depends on context
    # pad with filt size all around img
    pad_img = np.ones((s_img[0] + (2 * s_filt[0]), s_img[1] + (2 * s_filt[1])), dtype=np.float32) * padval

    pad_img[s_filt[0]: s_img[0] + s_filt[0], s_filt[1]: s_img[1] + s_filt[1]] = img

    output = np.zeros(pad_img.shape, dtype=np.float32)

    d_pad_img = cuda.to_device(pad_img)
    d_filt = cuda.to_device(filt)
    d_output = cuda.to_device(output)

    stupidconv_gpu_helper(d_pad_img, d_filt, s_img[0], s_img[1], s_filt[0], s_filt[1], d_output)

    output = d_output.copy_to_host()
    output = output[s_filt[0]:s_filt[0] + s_img[0], s_filt[1]:s_filt[1] + s_img[1]]

    return output / filt_nnz


def stupidconv2(img, filt, padval):
    """
    does convolution without using FFT because FFT is pissing me off and giving me weird answers
    :param img:
    :param filt:
    :param padval:
    :return:
    """

    # get the number of nonzero entries in the filter for later averaging of result
    filt_nnz = np.count_nonzero(filt)

    filt = np.fliplr(np.flipud(filt))

    output = ss.fftconvolve(img, filt, mode="same")

    return output / filt_nnz
