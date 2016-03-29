"""
describes class and related functions for vision models
"""

import contrastmodel.functions.stimuli as stims
import numpy as np


class Model:
    """
    base class for model - contains variables and functions common to all models
    """
    def __init__(self):
        self._filterresponses = {}
        self._ap_filterresponses = {}



class Lapdog2Model(Model):
    """
    model parameter set specific to LAPDOG2 (LAPDOG with excitation and inhibition)

    """

    def __init__(self, npow, weights, variant):
        """
        initializes parameter values

        :param list[float] npow: power to which the correlation mask is raised - smaller values
        generally make for wider and stronger influence from presynaptic filters, larger values
        tend towards weaker more localized connections
        :param list[(int,int)] weights: list of weight tuples: (inhibitory weight, excitatory weight)
        :param string variant: type/version of model (LAPDOG, LAPDOG2, etc)
        """

        self.npow = npow
        self.variant = variant
        self.output = {}
        self.friendlyname = variant
        self.weights = weights

    def process_stim(self, stim):
        """
        processes stimulus

        :param stims.Stim stim: stimulus to be processed

        """

        # make copy of filter responses to stimulus so they can be weighted to represent differing frequency
        # sensitivities (roughly corresponding to CSF sensitivity)
        self._filterresponses = stim.filtresponses
        self._ap_filterresponses = stim.ap_filtresponses




