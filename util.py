import numpy as np


def dirichlet_reward(nks):
    # dirichlet reward
    n = sum(nks)
    rc = 0
    for nk in nks:
        if nk:
            pck = nk / (0 + n - 1)
            rc += nk * np.log(pck)
    return rc
