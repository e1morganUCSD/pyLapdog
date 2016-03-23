import math
import numpy as np


# ---FILTER GENERATION HELPER FUNCTIONS---
def _gauss(x, std):
    """
    Generates a 1D gaussian
    :param x: np.array
    :param std: float
    :return: np.array
    """

    return np.exp(-(x**2) / (2 * (std**2))) / (std * math.sqrt(2 * math.pi))


def _d2gauss(n1, std1, n2, std2, theta):
    """
    Generates a 2D Gaussian with size n1*n2.  Theta is the angle that the
    filter is rotated counterclockwise, std1 and std2 are the standard
    deviations of the Gaussian functions, with std2 being the length of the
    filter along the rotated axis.
    :param n1: int
    :param std1: float
    :param n2: int
    :param std2: float
    :param theta: float
    :return: np.array
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
    Xs = np.reshape(Xs, (1, n1*n2), order='F').copy()
    Ys = np.reshape(Ys, (1, n1*n2), order='F').copy()

    coor = np.dot(r, np.vstack((Xs, Ys)))

    # compute 1D Gaussians
    gaussX = _gauss(np.array(coor[0, :]), std1)
    gaussY = _gauss(np.array(coor[1, :]), std2)

    # elementwise multiplication creates a 2D Gaussian
    h = np.reshape(gaussX * gaussY, (n2, n1), order='F').copy()
    h = h / np.sum(h)

    return h


# ---FILTER GENERATION FUNCTIONS---
def gen_ODOG(rows, cols, row_std, col_std, sr1, sr2, theta, centerWeight):
    """
    Generates a 2D Oriented Difference of Gaussian (ODOG) filter
    :param rows: int
    :param cols: int
    :param row_std: float
    :param col_std: float
    :param sr1: float
    :param sr2: float
    :param theta: float
    :param centerWeight: float
    :return: numpy.array
    """

    return centerWeight * (_d2gauss(rows, row_std, cols, col_std, theta) -
                           _d2gauss(rows, row_std * sr1, cols, col_std * sr2,
                                    theta))


# ---FILTER HELPER FUNCTIONS---
def trimfilt(filt, threshold):
    """
    flattens values between +threshold and -threshold to 0, then deletes
    all-zero rows from all sides of 2D array
    :param filt: numpy.array
    :param threshold: float
    :return: numpy.array
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
            alldone = True
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
    test2 = gen_ODOG(500, 500, 8.4, 8.4, 1.0, 2.0, 60, 1.0)

    fig2 = plt.imshow(test2)
    plt.colorbar()
    plt.show()

    # Note: double checked against MATLAB code, and results were the same
    test3 = trimfilt(test2, np.amax(np.abs(test2)) * 0.01)

    fig3 = plt.imshow(test3)
    plt.colorbar()
    plt.show()




