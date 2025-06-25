import numpy as np


def step_order_to_coded_int(step, order):
    """
    Helper function which packs the step and order into a single integer
    """

    step_order = (step << 16) | (order & 0x0000ffff)
    # print("Step:  {0:032b}".format(step))
    # print("Order: {0:032b}".format(order))
    # print("Mix:   {0:032b}".format(step_order))
    return np.int32(step_order)
