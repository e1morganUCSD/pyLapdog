"""
This module contains functions for generating image files
"""

import matplotlib.pyplot as plt
import os


def generate_image(img, title, filename, img_dir, cmap="jet", maximize=False):
    """
    generates figure image from array and saves it to file

    :param numpy.core.multiarray.ndarray img: array of values to be displayed
    :param str title: title for the figure
    :param str filename: name of the file that the image will be saved to
    :param str img_dir: directory the file will be saved in
    :param str cmap: colormap used in rendering the figure
    :param bool maximize: whether or not to maximize the figure before saving
    """

    # TODO: review image usage and create directories beforehand, then remove these lines - directories should exist
    # before imaging is called
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    plt.ioff()  # turn off interactive mode (this should prevent a bunch of figures from popping up on screen)

    plt.imshow(img, interpolation="none", cmap=cmap)
    plt.colorbar()
    plt.suptitle(title)
    # TODO: figure out how to maximize (or at least make sufficiently large)
    if maximize:
        pass
    plt.savefig(img_dir + filename)
    plt.close()


def make_dir_if_not_existing(dir_path):
    """
    Checks to see if a directory exists and if it doesn't, creates it

    :param str dir_path: directory path to look for

    """

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
