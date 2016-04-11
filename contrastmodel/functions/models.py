"""
describes class and related functions for vision models
"""

import contrastmodel.functions.stimuli as stims
import numpy as np
import contrastmodel.functions.imaging as imaging
import contrastmodel.functions.gpufunc as gpuf
import contrastmodel.functions.filtergen as fgen
# import scipy.signal as ss
import copy
# from numba import cuda


class Model(object):
    # TODO: figure out how to make it so that I can create a new "Model" and have it figure out the subclass
    # from the init?
    """
    NOT NEEDED?
    base class for model - contains variables and functions common to all models
    """
    def __init__(self):
        self._filterresponses = {}  # not needed?  using local vars in the functions should make them auto-cleanup?
        self._ap_filterresponses = {}


class LapdogModel(object):
    """
    model parameter set specific to LAPDOG and LAPDOG2 (LAPDOG with excitation and inhibition)

    """

    def __init__(self, variant, npow, conn_weights):
        """
        initializes parameter values

        :param string variant: type/version of model (LAPDOG, LAPDOG2, etc)
        :param list[float] npow: power to which the correlation mask is raised - smaller values
        generally make for wider and stronger influence from presynaptic filters, larger values
        tend towards weaker more localized connections
        :param list(list(int)) or list(list(int,int)) conn_weights: list of connection weights: only inhibitory
        weight for LAPDOG, inhibitory weight and excitatory weight for LAPDOG2
        """

        self.npow = npow
        self.variant = variant
        self.output = {}
        self.friendlyname = variant.upper()
        self.outDir = self.friendlyname + "/"
        self.conn_weights = conn_weights
        self.filt_weights = (0.9107, 1.0000, 0.9923, 0.9080, 0.7709, 0.7353, 0.5139)

    def process_stim(self, stim):
        """
        processes stimulus

        :rtype: (dict[str, numpy.core.multiarray.ndarray], dict[str, str])
        :param stims.Stim stim: stimulus to be processed

        """
        print("Now processing " + self.friendlyname + " for stimulus " + stim.friendlyname + ":")
        # make copy of filter responses to stimulus so they can be weighted to represent differing frequency
        # sensitivities (roughly corresponding to CSF sensitivity)
        filterresponses = copy.deepcopy(stim.filtresponses)
        ap_filterresponses = copy.deepcopy(stim.ap_filtresponses)

        # make local references to variables we will be using often
        stdev_pixels = stim.params.filt_stdev_pixels
        orientations = stim.params.filt_orientations
        verbosity = stim.params.verbosity

        # generate output and processing variables - created as dictionary keyed by model parameter options
        model_out = {}      # holds final processed stimulus
        model_outdir = {}   # holds directory of model

        # generate output folder
        outDir = stim.params.mainDir + stim.outDir + self.outDir
        imaging.make_dir_if_not_existing(outDir)

        for f in range(len(stdev_pixels)):
            filtweight = self.filt_weights[f]
            for o in range(len(orientations)):
                filterresponses[o][f] = filterresponses[o][f] * filtweight
                ap_filterresponses[o][f] = ap_filterresponses[o][f] * filtweight

                # output image files if needed
                if verbosity > 2:
                    filename = "z1-{}-weighted-prenormal-{}-{}.png".format(self.friendlyname, orientations[o],
                                                                           stdev_pixels[f])
                    title = "{} Normalization, weighted, prenormalization: orientation {}, frequency (pixels) " \
                            "{}".format(self.friendlyname, orientations[o], stdev_pixels[f])
                    imaging.generate_image(filterresponses[o][f], title, filename, outDir)

                    filename = "z1-{}-weighted-prenormal-ap-{}-{}.png".format(self.friendlyname, orientations[o],
                                                                              stdev_pixels[f])
                    title = "{} Normalization, weighted, antiphase, prenormalization: orientation {}, frequency" \
                            " (pixels) {}".format(self.friendlyname, orientations[o], stdev_pixels[f])
                    imaging.generate_image(ap_filterresponses[o][f], title, filename, outDir)

        for o in range(len(orientations)):
            print("Processing orientation {} of {}".format(o + 1, len(orientations)))
            # used to hold per-orientation results
            temp_orient = [[np.zeros((stim.params.filt_rows, stim.params.filt_cols)) for _ in self.conn_weights] for _
                           in self.npow]

            # for intermediate value outputs
            inh_exc_vals = []  # used to hold intermediate values for each value of npow

            for f in range(len(stdev_pixels)):
                # make a copy of filter responses
                filter_resp = np.copy(filterresponses[o][f])
                ap_filter_resp = np.copy(ap_filterresponses[o][f])

                # build inhibitory and excitatory responses for this filter based on responses of other orientations
                # and frequencies and their same-phase and anti-phase correlations.  sp = "standard phase" filter (
                # ON-centered), ap = "antiphase" filter (off-centered), apsp = antiphase connection to standard-phase
                #  filter, spap = standard phase connection to antiphase filter, etc.
                apsp_inh_values = np.zeros((stim.params.filt_rows, stim.params.filt_cols))
                spsp_inh_values = np.zeros((stim.params.filt_rows, stim.params.filt_cols))
                apap_inh_values = np.zeros((stim.params.filt_rows, stim.params.filt_cols))
                spap_inh_values = np.zeros((stim.params.filt_rows, stim.params.filt_cols))

                # save values in one list per npow setting
                for _ in range(len(self.npow)):
                    inh_exc_vals = inh_exc_vals + [[apsp_inh_values, spsp_inh_values, apap_inh_values, spap_inh_values]]
                
                if self.variant == "lapdog2":
                    apsp_exc_values = np.zeros((stim.params.filt_rows, stim.params.filt_cols))
                    spsp_exc_values = np.zeros((stim.params.filt_rows, stim.params.filt_cols))
                    apap_exc_values = np.zeros((stim.params.filt_rows, stim.params.filt_cols))
                    spap_exc_values = np.zeros((stim.params.filt_rows, stim.params.filt_cols))

                    for n in range(len(self.npow)):
                        inh_exc_vals[n] = inh_exc_vals[n] + [apsp_exc_values, spsp_exc_values, apap_exc_values,
                                                             spap_exc_values]

                for o2 in range(len(orientations)):
                    for f2 in range(len(stdev_pixels)):
                        # generate mask for connection strength between filter types - masks are negative for
                        # anticorrelation (inhibition), and positive for correlation (excitation)

                        filtermask = stim.params.filtermasks[o][f][o2][f2]
                        ap_filtermask = stim.params.ap_filtermasks[o][f][o2][f2]

                        # get max mask value for use in flattening edges of mask to prevent issues in later convolution
                        filtermax = np.max(np.abs(filtermask))
                        filterthresh = filtermax * 0.001

                        apsp_inh_mask = np.copy(ap_filtermask) * -1
                        apsp_inh_mask[apsp_inh_mask < filterthresh] = 0  # only keep negative values of original mask
                        spsp_inh_mask = np.copy(filtermask) * -1
                        spsp_inh_mask[spsp_inh_mask < filterthresh] = 0  # only keep negative values of original mask
                        apap_inh_mask = np.copy(filtermask) * -1
                        apap_inh_mask[apap_inh_mask < filterthresh] = 0  # only keep negative values of original mask
                        spap_inh_mask = np.copy(ap_filtermask) * -1
                        spap_inh_mask[spap_inh_mask < filterthresh] = 0  # only keep negative values of original mask

                        if self.variant == "lapdog2":
                            apsp_exc_mask = np.copy(ap_filtermask)
                            apsp_exc_mask[apsp_exc_mask < filterthresh] = 0
                            spsp_exc_mask = np.copy(filtermask)
                            spsp_exc_mask[spsp_exc_mask < filterthresh] = 0
                            apap_exc_mask = np.copy(filtermask)
                            apap_exc_mask[apap_exc_mask < filterthresh] = 0
                            spap_exc_mask = np.copy(ap_filtermask)
                            spap_exc_mask[spap_exc_mask < filterthresh] = 0

                        # get standard phase and antiphase presynaptic filter responses (note: filter response
                        # values range from 0 to 1) - to be used in the loops below
                        prefilt_response = filterresponses[o2][f2]
                        prefilt_ap_response = ap_filterresponses[o2][f2]

                        # raise masks to npow power
                        for n in range(len(self.npow)):
                            # # apply mask exponent and flip for use in convolution
                            # apsp_inh_mask_temp = np.fliplr(np.flipud(apsp_inh_mask ** self.npow[n]))
                            # spsp_inh_mask_temp = np.fliplr(np.flipud(spsp_inh_mask ** self.npow[n]))
                            # apap_inh_mask_temp = np.fliplr(np.flipud(apap_inh_mask ** self.npow[n]))
                            # spap_inh_mask_temp = np.fliplr(np.flipud(spap_inh_mask ** self.npow[n]))
                            #
                            # if self.variant == "lapdog2":
                            #     apsp_exc_mask_temp = np.fliplr(np.flipud(apsp_exc_mask ** self.npow[n]))
                            #     spsp_exc_mask_temp = np.fliplr(np.flipud(spsp_exc_mask ** self.npow[n]))
                            #     apap_exc_mask_temp = np.fliplr(np.flipud(apap_exc_mask ** self.npow[n]))
                            #     spap_exc_mask_temp = np.fliplr(np.flipud(spap_exc_mask ** self.npow[n]))

                            # apply mask exponent
                            apsp_inh_mask_temp = apsp_inh_mask ** self.npow[n]
                            spsp_inh_mask_temp = spsp_inh_mask ** self.npow[n]
                            apap_inh_mask_temp = apap_inh_mask ** self.npow[n]
                            spap_inh_mask_temp = spap_inh_mask ** self.npow[n]

                            if self.variant == "lapdog2":
                                apsp_exc_mask_temp = apsp_exc_mask ** self.npow[n]
                                spsp_exc_mask_temp = spsp_exc_mask ** self.npow[n]
                                apap_exc_mask_temp = apap_exc_mask ** self.npow[n]
                                spap_exc_mask_temp = spap_exc_mask ** self.npow[n]

                            # convolve presynaptic filter responses with connection masks to get levels of inhibition
                            #  and excitation to filter o,f for current stimulus
                            apsp_inh_masked_vals = gpuf.lapconv(prefilt_ap_response, apsp_inh_mask_temp, 0.0)
                            spsp_inh_masked_vals = gpuf.lapconv(prefilt_response, spsp_inh_mask_temp, 0.0)
                            apap_inh_masked_vals = gpuf.lapconv(prefilt_ap_response, apap_inh_mask_temp, 0.0)
                            spap_inh_masked_vals = gpuf.lapconv(prefilt_response, spap_inh_mask_temp, 0.0)
                            
                            # cut off values above 1
                            apsp_inh_masked_vals[apsp_inh_masked_vals > 1] = 1
                            spsp_inh_masked_vals[spsp_inh_masked_vals > 1] = 1
                            apap_inh_masked_vals[apap_inh_masked_vals > 1] = 1
                            spap_inh_masked_vals[spap_inh_masked_vals > 1] = 1

                            # save values for this o2, f2 filter into the overall inhibition for filter o, f
                            inh_exc_vals[n][0] = inh_exc_vals[n][0] + apsp_inh_masked_vals
                            inh_exc_vals[n][1] = inh_exc_vals[n][1] + spsp_inh_masked_vals
                            inh_exc_vals[n][2] = inh_exc_vals[n][2] + apap_inh_masked_vals
                            inh_exc_vals[n][3] = inh_exc_vals[n][3] + spap_inh_masked_vals

                            if self.variant == "lapdog2":
                                apsp_exc_masked_vals = gpuf.lapconv(prefilt_ap_response, apsp_exc_mask_temp, 0.0)
                                spsp_exc_masked_vals = gpuf.lapconv(prefilt_response, spsp_exc_mask_temp, 0.0)
                                apap_exc_masked_vals = gpuf.lapconv(prefilt_ap_response, apap_exc_mask_temp, 0.0)
                                spap_exc_masked_vals = gpuf.lapconv(prefilt_response, spap_exc_mask_temp, 0.0)

                                # cut off values above 1
                                apsp_exc_masked_vals[apsp_exc_masked_vals > 1] = 1
                                spsp_exc_masked_vals[spsp_exc_masked_vals > 1] = 1
                                apap_exc_masked_vals[apap_exc_masked_vals > 1] = 1
                                spap_exc_masked_vals[spap_exc_masked_vals > 1] = 1

                                # save values for this o2, f2 filter into the overall inhibition for filter o, f
                                inh_exc_vals[n][4] = inh_exc_vals[n][4] + apsp_exc_masked_vals
                                inh_exc_vals[n][5] = inh_exc_vals[n][5] + spsp_exc_masked_vals
                                inh_exc_vals[n][6] = inh_exc_vals[n][6] + apap_exc_masked_vals
                                inh_exc_vals[n][7] = inh_exc_vals[n][7] + spap_exc_masked_vals

                # get average inhibition/excitation
                for n in range(len(self.npow)):
                    for x in range(len(inh_exc_vals[n])):
                        inh_exc_vals[n][x] /= (len(orientations) * len(stdev_pixels))

                if verbosity > 2:
                    # create image files of the inhibitory/excitatory values
                    for n in range(len(self.npow)):
                        new_outDir = outDir + "npow{}/".format(self.npow[n])
                        filename = "z2-{}-e{}-apsp_inh_normvals-{}-{}.png".format(self.friendlyname, self.npow[n], o, f)
                        title = "{}-e{} AP-SP Inhibitory mask: orientation {}, " \
                                "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                               stdev_pixels[f])
                        imaging.generate_image(inh_exc_vals[n][0], title, filename, new_outDir)

                        filename = "z2-{}-e{}-spsp_inh_normvals-{}-{}.png".format(self.friendlyname, self.npow[n], o, f)
                        title = "{}-e{} SP-SP Inhibitory mask: orientation {}, " \
                                "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                               stdev_pixels[f])
                        imaging.generate_image(inh_exc_vals[n][1], title, filename, new_outDir)

                        filename = "z2-{}-e{}-apap_inh_normvals-{}-{}.png".format(self.friendlyname, self.npow[n], o, f)
                        title = "{}-e{} AP-AP Inhibitory mask: orientation {}, " \
                                "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                               stdev_pixels[f])
                        imaging.generate_image(inh_exc_vals[n][2], title, filename, new_outDir)

                        filename = "z2-{}-e{}-spap_inh_normvals-{}-{}.png".format(self.friendlyname, self.npow[n], o, f)
                        title = "{}-e{} SP-AP Inhibitory mask: orientation {}, " \
                                "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                               stdev_pixels[f])
                        imaging.generate_image(inh_exc_vals[n][3], title, filename, new_outDir)

                        if self.variant == "lapdog2":
                            filename = "z2-{}-e{}-apsp_exc_normvals-{}-{}.png".format(self.friendlyname,
                                                                                      self.npow[n], o, f)
                            title = "{}-e{} AP-SP Inhibitory mask: orientation {}, " \
                                    "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                                   stdev_pixels[f])
                            imaging.generate_image(inh_exc_vals[n][4], title, filename, new_outDir)

                            filename = "z2-{}-e{}-spsp_exc_normvals-{}-{}.png".format(self.friendlyname,
                                                                                      self.npow[n], o, f)
                            title = "{}-e{} SP-SP Inhibitory mask: orientation {}, " \
                                    "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                                   stdev_pixels[f])
                            imaging.generate_image(inh_exc_vals[n][5], title, filename, new_outDir)

                            filename = "z2-{}-e{}-apap_exc_normvals-{}-{}.png".format(self.friendlyname,
                                                                                      self.npow[n], o, f)
                            title = "{}-e{} AP-AP Inhibitory mask: orientation {}, " \
                                    "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                                   stdev_pixels[f])
                            imaging.generate_image(inh_exc_vals[n][6], title, filename, new_outDir)

                            filename = "z2-{}-e{}-spap_exc_normvals-{}-{}.png".format(self.friendlyname,
                                                                                      self.npow[n], o, f)
                            title = "{}-e{} SP-AP Inhibitory mask: orientation {}, " \
                                    "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                                   stdev_pixels[f])
                            imaging.generate_image(inh_exc_vals[n][7], title, filename, new_outDir)

                # apply inhibition/excitation
                for n in range(len(self.npow)):
                    for c in range(len(self.conn_weights)):
                        # collect inhibitory totals and weigh them
                        sp_total_inh = (inh_exc_vals[n][0] + inh_exc_vals[n][1]) * self.conn_weights[c][0]
                        ap_total_inh = (inh_exc_vals[n][2] + inh_exc_vals[n][3]) * self.conn_weights[c][0]

                        # cut off any values above 1
                        sp_total_inh[sp_total_inh > 1] = 1
                        ap_total_inh[ap_total_inh > 1] = 1

                        # apply inhibition
                        temp_sp_filter_resp = filter_resp - sp_total_inh
                        temp_ap_filter_resp = ap_filter_resp - ap_total_inh

                        # any values inhibited past 0 should be cut off (you can't have a negative firing rate)
                        temp_sp_filter_resp[temp_sp_filter_resp < 0] = 0
                        temp_ap_filter_resp[temp_ap_filter_resp < 0] = 0

                        variant_conn_string = "-inhWeight{}".format(self.conn_weights[c][0])

                        if self.variant == "lapdog2":
                            sp_total_exc = (inh_exc_vals[n][4] + inh_exc_vals[n][5]) * self.conn_weights[c][1]
                            ap_total_exc = (inh_exc_vals[n][6] + inh_exc_vals[n][7]) * self.conn_weights[c][1]

                            # cut off any values above 1
                            sp_total_exc[sp_total_exc > 1] = 1
                            ap_total_exc[ap_total_exc > 1] = 1

                            temp_sp_filter_resp = filter_resp + sp_total_exc
                            temp_ap_filter_resp = ap_filter_resp + ap_total_exc

                            temp_sp_filter_resp[temp_sp_filter_resp > 1] = 1
                            temp_ap_filter_resp[temp_ap_filter_resp > 1] = 1

                            variant_conn_string += "-excweight{}".format(self.conn_weights[c][1])

                            if verbosity > 2:
                                # output images of total inhibition and excitation (outputting inhibition here so it
                                # uses proper directory name
                                new_outDir = outDir + "npow{}/".format(self.npow[n]) + variant_conn_string + "/"
                                filename = "z3b-{}-e{}-total-SP-inhibition-{}-{}.png".format(self.friendlyname,
                                                                                             self.npow[n],
                                                                                             orientations[o],
                                                                                             stdev_pixels[f])
                                title = "{}-e{} SP Total Inhibition Values: orientation {}, " \
                                        "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                                       stdev_pixels[f])
                                imaging.generate_image(sp_total_inh, title, filename, new_outDir)

                                filename = "z3b-{}-e{}-total-AP-inhibition-{}-{}.png".format(self.friendlyname,
                                                                                             self.npow[n],
                                                                                             orientations[o],
                                                                                             stdev_pixels[f])
                                title = "{}-e{} AP Total Inhibition Values: orientation {}, " \
                                        "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                                       stdev_pixels[f])
                                imaging.generate_image(ap_total_inh, title, filename, new_outDir)

                                filename = "z3b-{}-e{}-total-SP-excitation-{}-{}.png".format(self.friendlyname,
                                                                                             self.npow[n],
                                                                                             orientations[o],
                                                                                             stdev_pixels[f])
                                title = "{}-e{} SP Total Excitation Values: orientation {}, " \
                                        "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                                       stdev_pixels[f])
                                imaging.generate_image(sp_total_exc, title, filename, new_outDir)

                                filename = "z3b-{}-e{}-total-AP-excitation-{}-{}.png".format(self.friendlyname,
                                                                                             self.npow[n],
                                                                                             orientations[o],
                                                                                             stdev_pixels[f])
                                title = "{}-e{} AP Total Excitation Values: orientation {}, " \
                                        "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                                       stdev_pixels[f])
                                imaging.generate_image(ap_total_exc, title, filename, new_outDir)

                        else:  # not lapdog2 (doing this here to preserve correct output directories)
                            if verbosity > 2:
                                new_outDir = outDir + "npow{}/".format(self.npow[n]) + variant_conn_string + "/"
                                filename = "z3b-{}-e{}-total-SP-inhibition-{}-{}.png".format(self.friendlyname,
                                                                                             self.npow[n],
                                                                                             orientations[o],
                                                                                             stdev_pixels[f])
                                title = "{}-e{} SP Total Inhibition Values: orientation {}, " \
                                        "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                                       stdev_pixels[f])
                                imaging.generate_image(sp_total_inh, title, filename, new_outDir)

                                filename = "z3b-{}-e{}-total-AP-inhibition-{}-{}.png".format(self.friendlyname,
                                                                                             self.npow[n],
                                                                                             orientations[o],
                                                                                             stdev_pixels[f])
                                title = "{}-e{} AP Total Inhibition Values: orientation {}, " \
                                        "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                                       stdev_pixels[f])
                                imaging.generate_image(ap_total_inh, title, filename, new_outDir)

                        # subtract the AP filter response from the SP response to get a total response that ranges
                        # from -1 to 1
                        temp_filter_resp = temp_sp_filter_resp - temp_ap_filter_resp
                        temp_filter_resp[temp_filter_resp < 0] = 0
                        temp_filter_resp[temp_filter_resp > 1] = 1
                        temp_orient[n][c] += temp_filter_resp

                        variant_full_name = self.variant + "-e{}".format(self.npow[n]) + variant_conn_string
                        if o == 0:
                            model_out[variant_full_name] = temp_filter_resp
                        else:
                            model_out[variant_full_name] += temp_filter_resp

            # divide per-orientation results by the number of frequencies so we get an average response
            for n in range(len(self.npow)):
                for c in range(len(self.conn_weights)):
                    temp_orient[n][c] /= len(stdev_pixels)

            if verbosity > 1:
                # generate image of per-orientation values
                for n in range(len(self.npow)):
                    for c in range(len(self.conn_weights)):
                        new_outDir = outDir + "npow{}/".format(self.npow[n]) + \
                                     "-inhWeight{}".format(self.conn_weights[c][0])
                        if self.variant == "lapdog2":
                            new_outDir += "-excweight{}".format(self.conn_weights[c][1])
                        new_outDir += "/"

                        title = "{}-e{}, weighted, normalized, and combined: orientation: {}".format(self.variant,
                                                                                                     self.npow[n],
                                                                                                     orientations[o])
                        filename = "z4-{}-e{}-normalized-weighted-{}.png".format(self.variant, self.npow[n],
                                                                                 orientations[o])
                        imaging.generate_image(temp_orient[n][c], title, filename, new_outDir)

        # divide final output by number of orientations to find average response
        for output in model_out:
            output /= (len(stdev_pixels) * len(orientations))
        # generate image of finished model output and process patch differences and plot them
        for n in range(len(self.npow)):
            for c in range(len(self.conn_weights)):
                variant_conn_string = "-inhWeight{}".format(self.conn_weights[c][0])
                if self.variant == "lapdog2":
                    variant_conn_string += "-excweight{}".format(self.conn_weights[c][1])
                new_outDir = outDir + "npow{}/".format(self.npow[n]) + variant_conn_string + "/"

                filename = "{}-final-model.png".format(self.friendlyname)
                title = "{} Final Model".format(self.friendlyname)
                imaging.generate_image(model_out[self.variant + "-e{}".format(self.npow[n]) + variant_conn_string],
                                       title, filename, new_outDir)
                model_outdir[self.variant + "-e{}".format(self.npow[n]) + variant_conn_string] = new_outDir

        return model_out, model_outdir


class FlodogModel(object):
    """
    model parameter set specific to FLODOG and FLODOG2 (FLODOG with normalization from other orientations as well as
    frequencies)

    """

    def __init__(self, variant, sig1mult, sr, sdmix):
        """
        initializes parameter values

        :param string variant: name of the model variant (flodog, flodog2, etc)
        :param list(float) sig1mult: extent of normalization mask in the direction of the filter, as multiplier -
            larger numbers mean the normalization is pooled from a wider area around the filter
        :param list(float) sr: extent of normalization mask perpendicular to the filter, as multiplier to sig1mult
        :param list(list(float)) sdmix: controls the slope of the Gaussian used for weighting the normalization from
            other filters.  in flodog2, the first number is the slope of the between-frequency Gaussian,
            and the second is the slope of the between-orientation Gaussian
        """

        self.variant = variant
        self.sig1mult = sig1mult
        self.sr = sr
        self.sdmix = sdmix
        self.output = {}
        self.friendlyname = variant
        self.outDir = self.friendlyname + "/"

    def process_stim(self, stim):
        """
        processes stimulus

        :rtype: (dict[str, numpy.core.multiarray.ndarray], dict[str, str])
        :param stims.Stim stim: stimulus to be processed

        """
        print("Now processing " + self.friendlyname + " for stimulus " + stim.friendlyname + ":")
        # make copy of filter responses to stimulus so they can be weighted to represent differing frequency
        # sensitivities (roughly corresponding to CSF sensitivity)
        filterresponses = copy.deepcopy(stim.filtresponses)
        ap_filterresponses = copy.deepcopy(stim.ap_filtresponses)

        # make local references to variables we will be using often
        stdev_pixels = stim.params.filt_stdev_pixels
        orientations = stim.params.filt_orientations
        verbosity = stim.params.verbosity

        # generate output and processing variables - created as dictionary keyed by model parameter options
        model_out = {}  # holds final processed stimulus
        model_outdir = {}  # holds directory of model

        # generate output folder
        outDir = stim.params.mainDir + stim.outDir + self.outDir
        imaging.make_dir_if_not_existing(outDir)

        for f in range(len(stdev_pixels)):
            filtweight = stim.params.filt_w_val[f]
            for o in range(len(orientations)):
                filterresponses[o][f] = filterresponses[o][f] * filtweight
                ap_filterresponses[o][f] = ap_filterresponses[o][f] * filtweight

                # output image files if needed
                if verbosity > 2:
                    filename = "z1-{}-weighted-prenormal-{}-{}.png".format(self.friendlyname, orientations[o],
                                                                           stdev_pixels[f])
                    title = "{} Normalization, weighted, prenormalization: orientation {}, frequency (pixels) " \
                            "{}".format(self.friendlyname, orientations[o], stdev_pixels[f])
                    imaging.generate_image(filterresponses[o][f], title, filename, outDir)

                    filename = "z1-{}-weighted-prenormal-ap-{}-{}.png".format(self.friendlyname, orientations[o],
                                                                              stdev_pixels[f])
                    title = "{} Normalization, weighted, antiphase, prenormalization: orientation {}, frequency" \
                            " (pixels) {}".format(self.friendlyname, orientations[o], stdev_pixels[f])
                    imaging.generate_image(ap_filterresponses[o][f], title, filename, outDir)

        for o in range(len(orientations)):
            print("Processing orientation {} of {}".format(o + 1, len(orientations)))
            # used to hold per-orientation results
            temp_orient = [[[np.zeros((stim.params.filt_rows, stim.params.filt_cols)) for _ in self.sig1mult] for _
                           in self.sr] for _ in self.sdmix]

            for f in range(len(stdev_pixels)):
                # make a copy of filter responses
                filter_resp = np.copy(filterresponses[o][f])

                # build weighted normalizer
                normalizer = [[np.zeros((stim.params.filt_rows, stim.params.filt_cols))] for _ in self.sdmix]
                area = [[0] for _ in self.sdmix]

                for sdmix_index in range(len(self.sdmix)):
                    for wf in range(len(stdev_pixels)):
                        if self.variant == "flodog2":
                            gweight = fgen.gauss(np.array([f - wf]), self.sdmix[sdmix_index][0])
                            area[sdmix_index] = area[sdmix_index] + gweight
                            # LAST POINT OF EDIT HERE
                        else:  # self.variant == 'flodog'
                            gweight = fgen.gauss(np.array([f - wf]), self.sdmix[sdmix_index])
                            area[sdmix_index] = area[sdmix_index] + gweight
                            normalizer[sdmix_index] = normalizer[sdmix_index] + filterresponses[o][wf] * gweight


                # save values in one list per npow setting
                for _ in range(len(self.npow)):
                    inh_exc_vals = inh_exc_vals + [[apsp_inh_values, spsp_inh_values, apap_inh_values, spap_inh_values]]

                if self.variant == "lapdog2":
                    apsp_exc_values = np.zeros((stim.params.filt_rows, stim.params.filt_cols))
                    spsp_exc_values = np.zeros((stim.params.filt_rows, stim.params.filt_cols))
                    apap_exc_values = np.zeros((stim.params.filt_rows, stim.params.filt_cols))
                    spap_exc_values = np.zeros((stim.params.filt_rows, stim.params.filt_cols))

                    for n in range(len(self.npow)):
                        inh_exc_vals[n] = inh_exc_vals[n] + [apsp_exc_values, spsp_exc_values, apap_exc_values,
                                                             spap_exc_values]

                for o2 in range(len(orientations)):
                    for f2 in range(len(stdev_pixels)):
                        # generate mask for connection strength between filter types - masks are negative for
                        # anticorrelation (inhibition), and positive for correlation (excitation)

                        filtermask = stim.params.filtermasks[o][f][o2][f2]
                        ap_filtermask = stim.params.ap_filtermasks[o][f][o2][f2]

                        # get max mask value for use in flattening edges of mask to prevent issues in later convolution
                        filtermax = np.max(np.abs(filtermask))
                        filterthresh = filtermax * 0.001

                        apsp_inh_mask = np.copy(ap_filtermask) * -1
                        apsp_inh_mask[apsp_inh_mask < filterthresh] = 0  # only keep negative values of original mask
                        spsp_inh_mask = np.copy(filtermask) * -1
                        spsp_inh_mask[spsp_inh_mask < filterthresh] = 0  # only keep negative values of original mask
                        apap_inh_mask = np.copy(filtermask) * -1
                        apap_inh_mask[apap_inh_mask < filterthresh] = 0  # only keep negative values of original mask
                        spap_inh_mask = np.copy(ap_filtermask) * -1
                        spap_inh_mask[spap_inh_mask < filterthresh] = 0  # only keep negative values of original mask

                        if self.variant == "lapdog2":
                            apsp_exc_mask = np.copy(ap_filtermask)
                            apsp_exc_mask[apsp_exc_mask < filterthresh] = 0
                            spsp_exc_mask = np.copy(filtermask)
                            spsp_exc_mask[spsp_exc_mask < filterthresh] = 0
                            apap_exc_mask = np.copy(filtermask)
                            apap_exc_mask[apap_exc_mask < filterthresh] = 0
                            spap_exc_mask = np.copy(ap_filtermask)
                            spap_exc_mask[spap_exc_mask < filterthresh] = 0

                        # get standard phase and antiphase presynaptic filter responses (note: filter response
                        # values range from 0 to 1) - to be used in the loops below
                        prefilt_response = filterresponses[o2][f2]
                        prefilt_ap_response = ap_filterresponses[o2][f2]

                        # raise masks to npow power
                        for n in range(len(self.npow)):
                            # # apply mask exponent and flip for use in convolution
                            # apsp_inh_mask_temp = np.fliplr(np.flipud(apsp_inh_mask ** self.npow[n]))
                            # spsp_inh_mask_temp = np.fliplr(np.flipud(spsp_inh_mask ** self.npow[n]))
                            # apap_inh_mask_temp = np.fliplr(np.flipud(apap_inh_mask ** self.npow[n]))
                            # spap_inh_mask_temp = np.fliplr(np.flipud(spap_inh_mask ** self.npow[n]))
                            #
                            # if self.variant == "lapdog2":
                            #     apsp_exc_mask_temp = np.fliplr(np.flipud(apsp_exc_mask ** self.npow[n]))
                            #     spsp_exc_mask_temp = np.fliplr(np.flipud(spsp_exc_mask ** self.npow[n]))
                            #     apap_exc_mask_temp = np.fliplr(np.flipud(apap_exc_mask ** self.npow[n]))
                            #     spap_exc_mask_temp = np.fliplr(np.flipud(spap_exc_mask ** self.npow[n]))

                            # apply mask exponent
                            apsp_inh_mask_temp = apsp_inh_mask ** self.npow[n]
                            spsp_inh_mask_temp = spsp_inh_mask ** self.npow[n]
                            apap_inh_mask_temp = apap_inh_mask ** self.npow[n]
                            spap_inh_mask_temp = spap_inh_mask ** self.npow[n]

                            if self.variant == "lapdog2":
                                apsp_exc_mask_temp = apsp_exc_mask ** self.npow[n]
                                spsp_exc_mask_temp = spsp_exc_mask ** self.npow[n]
                                apap_exc_mask_temp = apap_exc_mask ** self.npow[n]
                                spap_exc_mask_temp = spap_exc_mask ** self.npow[n]

                            # convolve presynaptic filter responses with connection masks to get levels of inhibition
                            #  and excitation to filter o,f for current stimulus
                            apsp_inh_masked_vals = gpuf.lapconv(prefilt_ap_response, apsp_inh_mask_temp, 0.0)
                            spsp_inh_masked_vals = gpuf.lapconv(prefilt_response, spsp_inh_mask_temp, 0.0)
                            apap_inh_masked_vals = gpuf.lapconv(prefilt_ap_response, apap_inh_mask_temp, 0.0)
                            spap_inh_masked_vals = gpuf.lapconv(prefilt_response, spap_inh_mask_temp, 0.0)

                            # cut off values above 1
                            apsp_inh_masked_vals[apsp_inh_masked_vals > 1] = 1
                            spsp_inh_masked_vals[spsp_inh_masked_vals > 1] = 1
                            apap_inh_masked_vals[apap_inh_masked_vals > 1] = 1
                            spap_inh_masked_vals[spap_inh_masked_vals > 1] = 1

                            # save values for this o2, f2 filter into the overall inhibition for filter o, f
                            inh_exc_vals[n][0] = inh_exc_vals[n][0] + apsp_inh_masked_vals
                            inh_exc_vals[n][1] = inh_exc_vals[n][1] + spsp_inh_masked_vals
                            inh_exc_vals[n][2] = inh_exc_vals[n][2] + apap_inh_masked_vals
                            inh_exc_vals[n][3] = inh_exc_vals[n][3] + spap_inh_masked_vals

                            if self.variant == "lapdog2":
                                apsp_exc_masked_vals = gpuf.lapconv(prefilt_ap_response, apsp_exc_mask_temp, 0.0)
                                spsp_exc_masked_vals = gpuf.lapconv(prefilt_response, spsp_exc_mask_temp, 0.0)
                                apap_exc_masked_vals = gpuf.lapconv(prefilt_ap_response, apap_exc_mask_temp, 0.0)
                                spap_exc_masked_vals = gpuf.lapconv(prefilt_response, spap_exc_mask_temp, 0.0)

                                # cut off values above 1
                                apsp_exc_masked_vals[apsp_exc_masked_vals > 1] = 1
                                spsp_exc_masked_vals[spsp_exc_masked_vals > 1] = 1
                                apap_exc_masked_vals[apap_exc_masked_vals > 1] = 1
                                spap_exc_masked_vals[spap_exc_masked_vals > 1] = 1

                                # save values for this o2, f2 filter into the overall inhibition for filter o, f
                                inh_exc_vals[n][4] = inh_exc_vals[n][4] + apsp_exc_masked_vals
                                inh_exc_vals[n][5] = inh_exc_vals[n][5] + spsp_exc_masked_vals
                                inh_exc_vals[n][6] = inh_exc_vals[n][6] + apap_exc_masked_vals
                                inh_exc_vals[n][7] = inh_exc_vals[n][7] + spap_exc_masked_vals

                # get average inhibition/excitation
                for n in range(len(self.npow)):
                    for x in range(len(inh_exc_vals[n])):
                        inh_exc_vals[n][x] /= (len(orientations) * len(stdev_pixels))

                if verbosity > 2:
                    # create image files of the inhibitory/excitatory values
                    for n in range(len(self.npow)):
                        new_outDir = outDir + "npow{}/".format(self.npow[n])
                        filename = "z2-{}-e{}-apsp_inh_normvals-{}-{}.png".format(self.friendlyname, self.npow[n], o, f)
                        title = "{}-e{} AP-SP Inhibitory mask: orientation {}, " \
                                "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                               stdev_pixels[f])
                        imaging.generate_image(inh_exc_vals[n][0], title, filename, new_outDir)

                        filename = "z2-{}-e{}-spsp_inh_normvals-{}-{}.png".format(self.friendlyname, self.npow[n], o, f)
                        title = "{}-e{} SP-SP Inhibitory mask: orientation {}, " \
                                "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                               stdev_pixels[f])
                        imaging.generate_image(inh_exc_vals[n][1], title, filename, new_outDir)

                        filename = "z2-{}-e{}-apap_inh_normvals-{}-{}.png".format(self.friendlyname, self.npow[n], o, f)
                        title = "{}-e{} AP-AP Inhibitory mask: orientation {}, " \
                                "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                               stdev_pixels[f])
                        imaging.generate_image(inh_exc_vals[n][2], title, filename, new_outDir)

                        filename = "z2-{}-e{}-spap_inh_normvals-{}-{}.png".format(self.friendlyname, self.npow[n], o, f)
                        title = "{}-e{} SP-AP Inhibitory mask: orientation {}, " \
                                "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                               stdev_pixels[f])
                        imaging.generate_image(inh_exc_vals[n][3], title, filename, new_outDir)

                        if self.variant == "lapdog2":
                            filename = "z2-{}-e{}-apsp_exc_normvals-{}-{}.png".format(self.friendlyname,
                                                                                      self.npow[n], o, f)
                            title = "{}-e{} AP-SP Inhibitory mask: orientation {}, " \
                                    "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                                   stdev_pixels[f])
                            imaging.generate_image(inh_exc_vals[n][4], title, filename, new_outDir)

                            filename = "z2-{}-e{}-spsp_exc_normvals-{}-{}.png".format(self.friendlyname,
                                                                                      self.npow[n], o, f)
                            title = "{}-e{} SP-SP Inhibitory mask: orientation {}, " \
                                    "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                                   stdev_pixels[f])
                            imaging.generate_image(inh_exc_vals[n][5], title, filename, new_outDir)

                            filename = "z2-{}-e{}-apap_exc_normvals-{}-{}.png".format(self.friendlyname,
                                                                                      self.npow[n], o, f)
                            title = "{}-e{} AP-AP Inhibitory mask: orientation {}, " \
                                    "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                                   stdev_pixels[f])
                            imaging.generate_image(inh_exc_vals[n][6], title, filename, new_outDir)

                            filename = "z2-{}-e{}-spap_exc_normvals-{}-{}.png".format(self.friendlyname,
                                                                                      self.npow[n], o, f)
                            title = "{}-e{} SP-AP Inhibitory mask: orientation {}, " \
                                    "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                                   stdev_pixels[f])
                            imaging.generate_image(inh_exc_vals[n][7], title, filename, new_outDir)

                # apply inhibition/excitation
                for n in range(len(self.npow)):
                    for c in range(len(self.conn_weights)):
                        # collect inhibitory totals and weigh them
                        sp_total_inh = (inh_exc_vals[n][0] + inh_exc_vals[n][1]) * self.conn_weights[c][0]
                        ap_total_inh = (inh_exc_vals[n][2] + inh_exc_vals[n][3]) * self.conn_weights[c][0]

                        # cut off any values above 1
                        sp_total_inh[sp_total_inh > 1] = 1
                        ap_total_inh[ap_total_inh > 1] = 1

                        # apply inhibition
                        temp_sp_filter_resp = filter_resp - sp_total_inh
                        temp_ap_filter_resp = ap_filter_resp - ap_total_inh

                        # any values inhibited past 0 should be cut off (you can't have a negative firing rate)
                        temp_sp_filter_resp[temp_sp_filter_resp < 0] = 0
                        temp_ap_filter_resp[temp_ap_filter_resp < 0] = 0

                        variant_conn_string = "-inhWeight{}".format(self.conn_weights[c][0])

                        if self.variant == "lapdog2":
                            sp_total_exc = (inh_exc_vals[n][4] + inh_exc_vals[n][5]) * self.conn_weights[c][1]
                            ap_total_exc = (inh_exc_vals[n][6] + inh_exc_vals[n][7]) * self.conn_weights[c][1]

                            # cut off any values above 1
                            sp_total_exc[sp_total_exc > 1] = 1
                            ap_total_exc[ap_total_exc > 1] = 1

                            temp_sp_filter_resp = filter_resp + sp_total_exc
                            temp_ap_filter_resp = ap_filter_resp + ap_total_exc

                            temp_sp_filter_resp[temp_sp_filter_resp > 1] = 1
                            temp_ap_filter_resp[temp_ap_filter_resp > 1] = 1

                            variant_conn_string += "-excweight{}".format(self.conn_weights[c][1])

                            if verbosity > 2:
                                # output images of total inhibition and excitation (outputting inhibition here so it
                                # uses proper directory name
                                new_outDir = outDir + "npow{}/".format(self.npow[n]) + variant_conn_string + "/"
                                filename = "z3b-{}-e{}-total-SP-inhibition-{}-{}.png".format(self.friendlyname,
                                                                                             self.npow[n],
                                                                                             orientations[o],
                                                                                             stdev_pixels[f])
                                title = "{}-e{} SP Total Inhibition Values: orientation {}, " \
                                        "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                                       stdev_pixels[f])
                                imaging.generate_image(sp_total_inh, title, filename, new_outDir)

                                filename = "z3b-{}-e{}-total-AP-inhibition-{}-{}.png".format(self.friendlyname,
                                                                                             self.npow[n],
                                                                                             orientations[o],
                                                                                             stdev_pixels[f])
                                title = "{}-e{} AP Total Inhibition Values: orientation {}, " \
                                        "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                                       stdev_pixels[f])
                                imaging.generate_image(ap_total_inh, title, filename, new_outDir)

                                filename = "z3b-{}-e{}-total-SP-excitation-{}-{}.png".format(self.friendlyname,
                                                                                             self.npow[n],
                                                                                             orientations[o],
                                                                                             stdev_pixels[f])
                                title = "{}-e{} SP Total Excitation Values: orientation {}, " \
                                        "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                                       stdev_pixels[f])
                                imaging.generate_image(sp_total_exc, title, filename, new_outDir)

                                filename = "z3b-{}-e{}-total-AP-excitation-{}-{}.png".format(self.friendlyname,
                                                                                             self.npow[n],
                                                                                             orientations[o],
                                                                                             stdev_pixels[f])
                                title = "{}-e{} AP Total Excitation Values: orientation {}, " \
                                        "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                                       stdev_pixels[f])
                                imaging.generate_image(ap_total_exc, title, filename, new_outDir)

                        else:  # not lapdog2 (doing this here to preserve correct output directories)
                            if verbosity > 2:
                                new_outDir = outDir + "npow{}/".format(self.npow[n]) + variant_conn_string + "/"
                                filename = "z3b-{}-e{}-total-SP-inhibition-{}-{}.png".format(self.friendlyname,
                                                                                             self.npow[n],
                                                                                             orientations[o],
                                                                                             stdev_pixels[f])
                                title = "{}-e{} SP Total Inhibition Values: orientation {}, " \
                                        "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                                       stdev_pixels[f])
                                imaging.generate_image(sp_total_inh, title, filename, new_outDir)

                                filename = "z3b-{}-e{}-total-AP-inhibition-{}-{}.png".format(self.friendlyname,
                                                                                             self.npow[n],
                                                                                             orientations[o],
                                                                                             stdev_pixels[f])
                                title = "{}-e{} AP Total Inhibition Values: orientation {}, " \
                                        "frequency (pixels) {}".format(self.friendlyname, self.npow[n], orientations[o],
                                                                       stdev_pixels[f])
                                imaging.generate_image(ap_total_inh, title, filename, new_outDir)

                        # subtract the AP filter response from the SP response to get a total response that ranges
                        # from -1 to 1
                        temp_filter_resp = temp_sp_filter_resp - temp_ap_filter_resp
                        temp_filter_resp[temp_filter_resp < 0] = 0
                        temp_filter_resp[temp_filter_resp > 1] = 1
                        temp_orient[n][c] += temp_filter_resp

                        variant_full_name = self.variant + "-e{}".format(self.npow[n]) + variant_conn_string
                        if o == 0:
                            model_out[variant_full_name] = temp_filter_resp
                        else:
                            model_out[variant_full_name] += temp_filter_resp

            if verbosity > 1:
                # generate image of per-orientation values
                for n in range(len(self.npow)):
                    for c in range(len(self.conn_weights)):
                        new_outDir = outDir + "npow{}/".format(self.npow[n]) + \
                                     "-inhWeight{}".format(self.conn_weights[c][0])
                        if self.variant == "lapdog2":
                            new_outDir += "-excweight{}".format(self.conn_weights[c][1])
                        new_outDir += "/"

                        title = "{}-e{}, weighted, normalized, and combined: orientation: {}".format(self.variant,
                                                                                                     self.npow[n],
                                                                                                     orientations[o])
                        filename = "z4-{}-e{}-normalized-weighted-{}.png".format(self.variant, self.npow[n],
                                                                                 orientations[o])
                        imaging.generate_image(temp_orient[n][c], title, filename, new_outDir)

        # generate image of finished model output and process patch differences and plot them
        for n in range(len(self.npow)):
            for c in range(len(self.conn_weights)):
                variant_conn_string = "-inhWeight{}".format(self.conn_weights[c][0])
                if self.variant == "lapdog2":
                    variant_conn_string += "-excweight{}".format(self.conn_weights[c][1])
                new_outDir = outDir + "npow{}/".format(self.npow[n]) + variant_conn_string + "/"

                filename = "{}-final-model.png".format(self.friendlyname)
                title = "{} Final Model".format(self.friendlyname)
                imaging.generate_image(model_out[self.variant + "-e{}".format(self.npow[n]) + variant_conn_string],
                                       title, filename, new_outDir)
                model_outdir[self.variant + "-e{}".format(self.npow[n]) + variant_conn_string] = new_outDir

        return model_out, model_outdir


def get_model(variant="", npow=None, conn_weights=None, sig1mult=None, sr=None, sdmix=None):
    """
    Returns model of the correct type
    :param string variant: type of model (flodog, flodog2, lapdog, lapdog2)
    :param list(float) npow: (for LAPDOG) power to which the connectivity values are raised. higher power = less
        connectivity
    :param list(list(float)) or list(list(float, float)) conn_weights: (for LAPDOG) inhibitory or inhibitory/excitatory
        multipliers for LAPDOG inhibition and excitation
    :param sig1mult: (for FLODOG) extent of normalization mask in the direction of the filter, as multiplier
    :param sr: (for FLODOG) extent of normalization mask perpendicular to the filter, as multiplier to sig1mult
    :param sdmix: (for FLODOG) standard deviation of Gaussian used for weighting - for FLODOG2 the first value is the
        standard deviation across frequencies and the second value is the standard deviation across orientations
    :return: requested model
    :rtype: LapdogModel or FlodogModel
    """

    # default output for invalid model type just to make the warnings on the return statement shut up
    model = None

    if variant[0:6] == "lapdog":
        model = LapdogModel(variant, npow, conn_weights)
    elif variant[0:6] == "flodog":
        model = FlodogModel(sig1mult, sr, sdmix)

    return model



