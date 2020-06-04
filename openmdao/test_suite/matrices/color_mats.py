"""
Loop over all of the matrix files in the test_suite/matrices dir, color them and summarize.
"""
import os
import sys

import numpy as np
from scipy.sparse import load_npz


import openmdao.test_suite
from openmdao.utils.coloring import _compute_coloring


if __name__ == '__main__':
    matdir = os.path.join(os.path.dirname(openmdao.test_suite.__file__), 'matrices')

    if len(sys.argv) == 1:
        fnames = os.listdir(matdir)
    else:
        fnames = [sys.argv[1]]

    c1 = ['Matname'] + fnames
    c1wid = len(sorted(c1, key=lambda s: len(s))[-1])

    template = "{:<{c1wid}} {:<6} {:<5} {:<5} {:<4} {:<4} {:<4} {:<5} {}"
    print(template.format('Matname', 'Mode', 'Rows', 'Cols', 'Fwd', 'Rev', 'Tot',
                          'Pct', 'Fallback?', c1wid=c1wid))
    template = "{:<{c1wid}} {:<6} {:<5} {:<5} {:<4} {:<4} {:<4} {:.2f} {}"
    for fname in fnames:
        mat = None
        if fname.endswith('.npz'):
            mat = load_npz(os.path.join(matdir, fname))
            matname = os.path.splitext(fname)[0]
            mat = np.asarray(mat.toarray(), dtype=bool)
            for mode in ['auto', 'fwd', 'rev']:
                coloring = _compute_coloring(mat, mode)
                fallback = 'True' if coloring._meta.get('fallback', False) else 'False'
                tot_size, tot_solves, fwd_solves, rev_solves, pct = coloring._solves_info()
                print(template.format(matname, mode.upper(),
                                      mat.shape[0], mat.shape[1], fwd_solves,
                                      rev_solves, tot_solves, pct, fallback, c1wid=c1wid))
                coloring = None
            print()
