import math
import numpy as np


def _gauss(x, std):
    """
    Generates a 1D gaussian
    :param x: np.array
    :param std: float
    :return: np.array
    """

    return np.exp(x**2 / (2 * std**2) / (std * math.sqrt(2 * math.pi)))


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
    r = numpy.array([[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]])

    temp = np.array(range(-(n1 - 1) / 2, n1 / 2))  # TODO: verify this is
    # integer division
    Xs = np.tile(temp, (n2, 1))  # creates an n2 x len(temp) array?
    temp = np.array([range(-(n2 - 1) / 2, n1 / 2)])  # TODO: verify that
    # this results in a single-row 2D array (shows up as [[ x y z ]] rather
    # than [x y z]
    temp = temp.T
    Ys = np.tile(temp, (1, n1))  # creates an len(temp) x n1 array?

    coor = r.dot(np.vstack((Xs, Ys)))

    #TODO: finish this function

