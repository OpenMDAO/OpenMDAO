"""
Convert a Matlab matrix file (like those found at sparse.tamu.edu) to scipy sparse .npz file.
"""

import numpy as np
import scipy.io
from scipy.sparse import save_npz


def matlab2npy(mlfile):
    d = scipy.io.loadmat(mlfile)

    for i in [1, 2, 0]:
        a = d["Problem"][0][0][i]
        try:
            a.data = np.abs(a.data)
        except Exception:
            continue
        else:
            break
    else:
        raise RuntimeError("couldn't find the matrix!!!")


    nrows, ncols = a.shape

    print("Nonzeros:", np.count_nonzero(a.data), "of", nrows * ncols)
    print("Max:", a.data.max())
    print("Min nonzero:", a.data[a.data > 0.].min())
    print("Shape:", a.shape)

    return a


if __name__ == '__main__':
    import sys, os

    a = matlab2npy(sys.argv[1])
    base = os.path.splitext(sys.argv[1])[0]
    save_npz('{}.npz'.format(base), a)
