"""
This module contains functions used for generation of ODOG and Gabor filters
"""

import math
import numpy as np
from numba import cuda


# ---FILTER GENERATION HELPER FUNCTIONS---
def gauss(x, std):
    """
    Generates a 1D gaussian

    :param numpy.core.multiarray.ndarray x: 1D array of values to be Gaussianed
    :param float std: standard deviation of Gaussian
    :return: 1D Gaussian
    :rtype: numpy.core.multiarray.ndarray
    """

    return np.exp(-(x ** 2) / (2 * (std ** 2))) / (
            std * math.sqrt(2 * math.pi))


def _d2gauss(n1, std1, n2, std2, theta):
    """
    Generates a 2D Gaussian with size n1*n2.

    :param int n1: number of rows in the Gaussian
    :param float std1: standard deviation along the width? of the filter
    :param int n2: number of columns in the Gaussian
    :param float std2: standard deviation along the length of the filter
    :param float theta: angle that the Gaussian is rotated counterclockwise
    :return: 2D Gaussian
    :rtype: numpy.core.multiarray.ndarray
    """

    # create transformation matrix for rotation
    r = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta), math.cos(theta)]])

    temp = np.array(range(-(n1 - 1) / 2, n1 / 2))
    Xs = np.tile(temp, (n2, 1))
    temp = np.array([range(-(n2 - 1) / 2, n1 / 2)])
    temp = temp.T
    Ys = np.tile(temp, (1, n1))  # creates an len(temp) x n1 array?

    # reshape into vectors
    Xs = np.reshape(Xs, (1, n1 * n2), order='F').copy()
    Ys = np.reshape(Ys, (1, n1 * n2), order='F').copy()

    coor = np.dot(r, np.vstack((Xs, Ys)))

    # compute 1D Gaussians
    gaussX = gauss(np.array(coor[0, :]), std1)
    gaussY = gauss(np.array(coor[1, :]), std2)

    # elementwise multiplication creates a 2D Gaussian
    h = np.reshape(gaussX * gaussY, (n2, n1), order='F').copy()
    h = h / np.sum(h)

    return h


# ---FILTER GENERATION FUNCTIONS---
def gen_odog(rows, cols, row_std, col_std, sr1, sr2, theta, center_weight):
    """
    Generates a 2D Oriented Difference of Gaussian (ODOG) filter

    :param int rows: number of rows in the filter
    :param int cols: number of columns in the filter
    :param float row_std: standard deviation in the vertical direction
        before rotation
    :param float col_std: standard deviation in the horizontal direction
        before rotation
    :param float sr1: scaling factor for row_std
    :param float sr2: scaling factor for col_std
    :param float theta: angle to which the filter is rotated counterclockwise
    :param float center_weight: scaling factor for the peak of the Gaussian
    :return: 2D Difference-Of-Gaussian filter
    :rtype: numpy.core.multiarray.ndarray
    """

    return center_weight * (_d2gauss(rows, row_std, cols, col_std, theta) -
                            _d2gauss(rows, row_std * sr1, cols, col_std * sr2,
                                     theta))


# ---FILTER HELPER FUNCTIONS---
def trimfilt(filt, threshold):
    """
    flattens values between +threshold and -threshold to 0, then deletes
    all-zero rows from all sides of 2D array

    :param numpy.core.multiarray.ndarray filt: filter to be flattened
    :param float threshold: values between positive and negative threshold
        become 0
    :return: flattened filter
    :rtype: numpy.core.multiarray.ndarray
    """
    filt[(filt < threshold) & (filt > -threshold)] = 0

    # do rows first, then columns, to reduce the number of comparisons
    topindex = 0
    bottomindex = filt.shape[0] - 1
    alldone = False
    while not alldone:
        alldone = True
        if np.sum(filt[topindex, :]) == 0:
            alldone = False
            topindex += 1
        if np.sum(filt[bottomindex, :]) == 0:
            alldone = False
            bottomindex -= 1

        # if top and bottom meet, the filter is all zeros
        if topindex - bottomindex > 0:
            return filt  # we can't trim an all zeros filter

    trimmed_filt = filt[topindex:bottomindex, :].copy()

    # now we trim the sides
    leftindex = 0
    rightindex = trimmed_filt.shape[1] - 1
    alldone = False
    while not alldone:
        alldone = True
        if np.sum(trimmed_filt[:, leftindex]) == 0:
            alldone = False
            leftindex += 1
        if np.sum(trimmed_filt[:, rightindex]) == 0:
            alldone = False
            rightindex -= 1

        # if left and right meet, the filter is all zeros
        if leftindex - rightindex > 0:
            alldone = True

    return trimmed_filt[:, leftindex:rightindex].copy()


# ---TESTING---
if __name__ == "__main__":
    print("testing d2gauss - should produce 200x200 gaussian")
    test = _d2gauss(200, 8.4, 200, 8.4, 60)

    import matplotlib.pyplot as plt

    fig = plt.imshow(test)
    plt.show()

    print("now generating ODOG filter")
    test2 = gen_odog(500, 500, 8.4, 8.4, 1.0, 2.0, 60, 1.0)

    fig2 = plt.imshow(test2)
    plt.colorbar()
    plt.show()

    # Note: double checked against MATLAB code, and results were the same
    test3 = trimfilt(test2, np.amax(np.abs(test2)) * 0.01)

    fig3 = plt.imshow(test3)
    plt.colorbar()
    plt.show()
