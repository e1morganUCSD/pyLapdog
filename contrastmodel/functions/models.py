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

    def __init__(self, npow, conn_weights, variant):
        """
        initializes parameter values

        :param list[float] npow: power to which the correlation mask is raised - smaller values
        generally make for wider and stronger influence from presynaptic filters, larger values
        tend towards weaker more localized connections
        :param List[int] or list[int,int] conn_weights: list of connection weights: only inhibitory weight for
        LAPDOG, inhibitory weight and excitatory weight for LAPDOG2
        :param string variant: type/version of model (LAPDOG, LAPDOG2, etc)
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

                        apsp_inh_mask = ap_filtermask(filtermask <= 0) * -1
                        spsp_inh_mask = filtermask(filtermask <= 0) * -1
                        apap_inh_mask = filtermask(filtermask <= 0) * -1
                        spap_inh_mask = ap_filtermask(filtermask <= 0) * -1

                        if self.variant == "lapdog2":
                            apsp_exc_mask = ap_filtermask(filtermask >= 0)
                            spsp_exc_mask = filtermask(filtermask >= 0)
                            apap_exc_mask = filtermask(filtermask >= 0)
                            spap_exc_mask = ap_filtermask(filtermask >= 0)

                        # get standard phase and antiphase presynaptic filter responses (note: filter response
                        # values range from 0 to 1) - to be used in the loops below
                        prefilt_response = filterresponses[o2][f2]
                        prefilt_ap_response = ap_filterresponses[o2][f2]

                        # raise masks to npow power
                        for n in self.npow:
                            # apply mask exponent and flip for use in convolution
                            apsp_inh_mask_temp = np.fliplr(np.flipud(apsp_inh_mask**self.npow[n]))
                            spsp_inh_mask_temp = np.fliplr(np.flipud(spsp_inh_mask**self.npow[n]))
                            apap_inh_mask_temp = np.fliplr(np.flipud(apap_inh_mask**self.npow[n]))
                            spap_inh_mask_temp = np.fliplr(np.flipud(spap_inh_mask**self.npow[n]))

                            if self.variant == "lapdog2":
                                apsp_exc_mask_temp = np.fliplr(np.flipud(apsp_exc_mask**self.npow[n]))
                                spsp_exc_mask_temp = np.fliplr(np.flipud(spsp_exc_mask**self.npow[n]))
                                apap_exc_mask_temp = np.fliplr(np.flipud(apap_exc_mask**self.npow[n]))
                                spap_exc_mask_temp = np.fliplr(np.flipud(spap_exc_mask**self.npow[n]))

                            # convolve presynaptic filter responses with connection masks to get levels of inhibition
                            #  and excitation to filter o,f for current stimulus
                            apsp_inh_masked_vals = ndi.convolve(prefilt_ap_response, apsp_inh_mask_temp, 
                                                                mode='constant', cval=0.0)
                            spsp_inh_masked_vals = ndi.convolve(prefilt_response, spsp_inh_mask_temp, mode='constant',
                                                                cval=0.0)
                            apap_inh_masked_vals = ndi.convolve(prefilt_ap_response, apap_inh_mask_temp, 
                                                                mode='constant', cval=0.0)
                            spap_inh_masked_vals = ndi.convolve(prefilt_response, spap_inh_mask_temp, mode='constant',
                                                                cval=0.0)
                            
                            # save values for this o2, f2 filter into the overall inhibition for filter o, f
                            inh_exc_vals[n][0] = inh_exc_vals[n][0] + apsp_inh_masked_vals
                            inh_exc_vals[n][1] = inh_exc_vals[n][1] + spsp_inh_masked_vals
                            inh_exc_vals[n][2] = inh_exc_vals[n][2] + apap_inh_masked_vals
                            inh_exc_vals[n][3] = inh_exc_vals[n][3] + spap_inh_masked_vals

                            if self.variant == "lapdog2":
                                apsp_exc_masked_vals = ndi.convolve(prefilt_ap_response, apsp_exc_mask_temp,
                                                                    mode='constant', cval=0.0)
                                spsp_exc_masked_vals = ndi.convolve(prefilt_response, spsp_exc_mask_temp,
                                                                    mode='constant', cval=0.0)
                                apap_exc_masked_vals = ndi.convolve(prefilt_ap_response, apap_exc_mask_temp,
                                                                    mode='constant', cval=0.0)
                                spap_exc_masked_vals = ndi.convolve(prefilt_response, spap_exc_mask_temp,
                                                                    mode='constant', cval=0.0)
                                # save values for this o2, f2 filter into the overall inhibition for filter o, f
                                inh_exc_vals[n][0] = inh_exc_vals[n][0] + apsp_exc_masked_vals
                                inh_exc_vals[n][1] = inh_exc_vals[n][1] + spsp_exc_masked_vals
                                inh_exc_vals[n][2] = inh_exc_vals[n][2] + apap_exc_masked_vals
                                inh_exc_vals[n][3] = inh_exc_vals[n][3] + spap_exc_masked_vals

                # get average inhibition/excitation
                for n in range(len(self.npow)):
                    for x in range(len(inh_exc_vals[n])):
                        inh_exc_vals[n][x] /= (len(orientations) * len(stdev_pixels))

                if verbosity == 3:
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
                    for c in range(self.conn_weights):
                        # collect inhibitory totals and weigh them
                        sp_total_inh = (inh_exc_vals[n][0] + inh_exc_vals[n][1]) * self.conn_weights[c][0]
                        ap_total_inh = (inh_exc_vals[n][2] + inh_exc_vals[n][3]) * self.conn_weights[c][0]

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

                            temp_sp_filter_resp = filter_resp + sp_total_exc
                            temp_ap_filter_resp = ap_filter_resp + ap_total_exc

                            temp_sp_filter_resp[temp_sp_filter_resp > 1] = 1
                            temp_ap_filter_resp[temp_ap_filter_resp > 1] = 1

                            variant_conn_string += "-excweight{}".format(self.conn_weights[c][1])

                            # output images of total inhibition and excitation (outputting inhibition here so it uses
                            #  proper directory name
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
                        temp_filter_resp[temp_filter_resp > 0] = 1
                        temp_orient[n][c] += temp_filter_resp

                        variant_full_name = self.variant + "-e{}".format(self.npow[n]) + variant_conn_string
                        if o == 0:
                            model_out[variant_full_name] = temp_filter_resp
                        else:
                            model_out[variant_full_name] += temp_filter_resp
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

        # generate image of finished model output
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

        return model_out