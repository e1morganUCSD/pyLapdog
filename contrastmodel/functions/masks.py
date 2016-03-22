
def _compare_filters(pre_filter, post_filter):
    """
    Generates array of correlation values between two filters (as described
    in Troyer et al)

    At each point of the receptive field of a postsynaptic filter
    (post_filter), calculates level of same-phase correlation of
    receptive fields between a presynaptic_filter (pre_filter) centered at that
    point and the postsynaptic filter, using the formula:

    """
