"""
describes class and related functions for vision models
"""

import contrastmodel.functions.stimuli as stims
import numpy as np
import contrastmodel.functions.imaging as imaging
import scipy.ndimage as ndi
import copy


class Model(object):
    """
    base class for model - contains variables and functions common to all models
    """
    def __init__(self):
        self._filterresponses = {}  # not needed?  using local vars in the functions should make them auto-cleanup?
        self._ap_filterresponses = {}


class Lapdog2Model(Model):
    """
    model parameter set specific to LAPDOG2 (LAPDOG with excitation and inhibition)

    """

    def __init__(self, npow, conn_weights, variant):
        """
        initializes parameter values

        :param list[float] npow: power to which the correlation mask is raised - smaller values
        generally make for wider and stronger influence from presynaptic filters, larger values
        tend towards weaker more localized connections
        :param list[(int,int)] conn_weights: list of connection weight tuples: (inhibitory weight, excitatory weight)
        :param string variant: type/version of model (LAPDOG, LAPDOG2, etc)
        """

        # call parent class' init function
        super(Lapdog2Model, self).__init__()

        # now start with model-specific stuff
        self.npow = npow
        self.variant = variant
        self.outDir = variant + "/"
        self.output = {}
        self.friendlyname = variant
        self.conn_weights = conn_weights
        self.filt_weights = (0.9107, 1.0000, 0.9923, 0.9080, 0.7709, 0.7353, 0.5139)

    def process_stim(self, stim):
        """
        processes stimulus

        :param stims.Stim stim: stimulus to be processed

        """

        # make copy of filter responses to stimulus so they can be weighted to represent differing frequency
        # sensitivities (roughly corresponding to CSF sensitivity)
        filterresponses = copy.deepcopy(stim.filtresponses)
        ap_filterresponses = copy.deepcopy(stim.ap_filtresponses)

        # make local references to variables we will be using often
        stdev_pixels = stim.params.filt_stdev_pixels
        orientations = stim.params.filt_orientations
        verbosity = stim.params.verbosity

        # generate output and processing variables - created as dictionary keyed by model parameter options
        model_out = {}

        # generate output folder
        outDir = stim.params.mainDir + stim.outDir + self.outDir
        imaging.make_dir_if_not_existing(outDir)

        for f in range(len(stdev_pixels)):
            filtweight = self.filt_weights[f]
            for o in range(len(orientations)):
                filterresponses[o][f] = filterresponses[o][f] * filtweight

                # output image file if needed
                if verbosity == 3:
                    filename = "z1-{}-weighted-prenormal-{}-{}.png".format(self.friendlyname, orientations[o],
                                                                           stdev_pixels[f])
                    title = "{} Normalization, weighted, prenormalization: orientation {}, frequency (pixels) " \
                            "{}".format(self.friendlyname, orientations[o], stdev_pixels[f])
                    imaging.generate_image(filterresponses[o][f], title, filename, outDir)

        for f in range(len(stdev_pixels)):
            filtweight = self.filt_weights[f]
            for o in range(len(orientations)):
                ap_filterresponses[o][f] = ap_filterresponses[o][f] * filtweight

                # output image file if needed
                if verbosity == 3:
                    filename = "z1-{}-weighted-prenormal-ap-{}-{}.png".format(self.friendlyname, orientations[o],
                                                                           stdev_pixels[f])
                    title = "{} Normalization, weighted, antiphase, prenormalization: orientation {}, frequency" \
                            " (pixels) {}".format(self.friendlyname, orientations[o], stdev_pixels[f])
                    imaging.generate_image(ap_filterresponses[o][f], title, filename, outDir)

        for o in range(len(orientations)):
            temp_orient = {}  # used to hold per-orientation results for intermediate value outputs

            for f in range(len(stdev_pixels)):
                filter_resp = filterresponses[o][f]
                ap_filter_resp = ap_filterresponses[o][f]

                # build inhibitory and excitatory responses for this filter based on responses of other orientations
                # and frequencies and their same-phase and anti-phase correlations.  sp = "standard phase" filter (
                # ON-centered), ap = "antiphase" filter (off-centered), apsp = antiphase connection to standard-phase
                #  filter, spap = standard phase connection to antiphase filter, etc.
                apsp_inh_values = np.zeros((stim.params.filt_rows, stim.params.filt_cols))
                spsp_inh_values = np.zeros((stim.params.filt_rows, stim.params.filt_cols))
                apap_inh_values = np.zeros((stim.params.filt_rows, stim.params.filt_cols))
                spap_inh_values = np.zeros((stim.params.filt_rows, stim.params.filt_cols))

                for o2 in range(len(orientations)):
                    for f2 in range(len(stdev_pixels)):
                        # generate mask for connection strength between filter types - masks are negative for
                        # anticorrelation (inhibition), and positive for correlation (excitation)

                        filtermask = stim.params.filtermasks[o][f][o2][f2]
                        ap_filtermask = stim.params.ap_filtermasks[o][f][o2][f2]

                        apsp_inh_mask = ap_filtermask(filtermask <= 0) * -1
                        spsp_inh_mask = filtermask(filtermask <= 0) * -1
                        apap_inh_mask = filtermask(filtermask <= 0) * -1
                        spap_inh_mask = ap_filtermask(filtermask <= 0) * -1

                        if self.variant == "lapdog2":
                            apsp_exc_mask = ap_filtermask(filtermask >= 0)
                            spsp_exc_mask = filtermask(filtermask >= 0)
                            apap_exc_mask = filtermask(filtermask >= 0)
                            spap_exc_mask = ap_filtermask(filtermask >= 0)

                        # raise masks to npow power
                        masks_temp = {}
                        for n in self.npow:
                            masks_temp[n] = (apsp_inh_mask**n, spsp_inh_mask**n,
                                             apap_inh_mask**n, spap_inh_mask**n)

                            if self.variant == "lapdog2":
                                masks_temp[n] = masks_temp[n] + (apsp_exc_mask**n, spsp_exc_mask**n,
                                                                 apap_exc_mask**n, spap_exc_mask**n)

                            # get standard phase and antiphase presynaptic filter responses
                            ps_response = filterresponses[o2][f2]
                            ps_ap_response = ap_filterresponses[o2][f2]

                            # convolve presynaptic filter responses with connection masks to get levels of inhibition
                            #  and excitation
                            ps_response = np.fliplr(np.flipud(ps_response)) # FLIP MASK, NOT FILTER
                            ps_ap_response = np.fliplr(np.flipud(ps_ap_response))

                            sdlfjskdlj = ndi.convolve(post_filter, pre_filter, mode='constant', cval=0.0) \
                                   / correlation_denominator

