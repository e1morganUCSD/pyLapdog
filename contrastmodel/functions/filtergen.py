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
    r = np.array([[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]])

    temp = np.array(range(-(n1 - 1) / 2, n1 / 2))
    Xs = np.tile(temp, (n2, 1))
    temp = np.array([range(-(n2 - 1) / 2, n1 / 2)])
    temp = temp.T
    Ys = np.tile(temp, (1, n1))  # creates an len(temp) x n1 array?

    # reshape into vectors
    Xs = np.reshape(Xs, (1, n1*n2))
    Ys = np.reshape(Ys, (1, n1*n2))

    coor = r.dot(np.vstack((Xs, Ys)))

    # compute 1D Gaussians
    gaussX = _gauss(np.array(coor[0, :]), std1)
    gaussY = _gauss(np.array(coor[1, :]), std2)

    # elementwise multiplication creates a 2D Gaussian
    h = np.reshape(gaussX * gaussY, (n2, n1))
    h = h / np.sum(h)

    return h

# TODO: implement dogEx


# testing
if __name__ == "__main__":
    print("testing d2gauss - should produce 200x200 gaussian")
    test = _d2gauss(200, 20, 200, 20, 60)

    import matplotlib.pyplot as plt

    fig = plt.imshow(test)
    plt.show()





