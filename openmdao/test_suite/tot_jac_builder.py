"""
A tool to make it easier to investigate coloring of jacobians with different sparsity structures.
"""

import sys
import numpy as np

from openmdao.utils.general_utils import printoptions
from openmdao.utils.coloring import _compute_coloring

class TotJacBuilder(object):
    def __init__(self, rows, cols):
        self.J = np.zeros((rows, cols), dtype=bool)
        self.coloring = None

    def add_random_points(self, npoints):
        nrows, ncols = self.J.shape
        count = 0

        zro = self.J == False
        flat = self.J[zro].flatten()
        flat[:npoints] = True
        np.random.shuffle(flat)
        self.J[zro] = flat

    def add_row(self, idx, density=1.0):
        self.add_block(self.create_row(density=density), idx, 0)

    def add_col(self, idx, density=1.0):
        self.add_block(self.create_col(density=density), 0, idx)

    def create_row(self, density=1.0):
        return self.create_block((1, self.J.shape[1]), density=density)

    def create_col(self, density=1.0):
        return self.create_block((self.J.shape[0], 1), density=density)

    def create_block(self, shape, density=1.0):
        if density == 1.0:
            return np.ones(shape, dtype=bool)
        else:
            rows, cols = shape
            num = int((rows * cols) * density)
            vec = np.zeros(int(rows * cols), dtype=bool)
            vec[:num] = True
            np.random.shuffle(vec)
            return vec.reshape(shape)

    def add_block(self, block, start_row, start_col):
        rows, cols = block.shape
        self.J[start_row:start_row + rows, start_col:start_col + cols] = block

    def add_block_diag(self, shapes, start_row, start_col, density=1.0):
        row_idx = start_row
        col_idx = start_col

        for shape in shapes:
            self.add_block(self.create_block(shape, density=density), row_idx, col_idx)
            row_idx += shape[0]
            col_idx += shape[1]

    def color(self, mode='auto', fname=None):
        self.coloring = _compute_coloring(self.J, mode)
        if self.coloring is not None and fname is not None:
            self.coloring.save(fname)
        return self.coloring

    def show(self):
        self.coloring.display_txt()

        maxdeg_fwd = np.max(np.count_nonzero(self.J, axis=1))
        maxdeg_rev = np.max(np.count_nonzero(self.J, axis=0))

        print("Shape:", self.J.shape)
        print("Density:", np.count_nonzero(self.J) / self.J.size)
        print("Max degree (fwd, rev):", maxdeg_fwd, maxdeg_rev)

        self.coloring.summary()

    def shuffle_rows(self):
        np.random.shuffle(self.J)

    def density_info(self):
        J = self.J
        density = np.count_nonzero(J) / J.size
        row_density = np.count_nonzero(J, axis=1) / J.shape[1]
        max_row_density = np.max(row_density)
        n_dense_rows = row_density[row_density == 1.0].size
        col_density = np.count_nonzero(J, axis=0) / J.shape[0]
        max_col_density = np.max(col_density)
        n_dense_cols = col_density[col_density == 1.0].size
        return density, max_row_density, n_dense_rows, max_col_density, n_dense_cols

    @staticmethod
    def make_blocks(num_blocks, min_shape, max_shape):
        shapes = []
        row_size = col_size = 0
        min_rows, min_cols = min_shape
        max_rows, max_cols = max_shape

        for b in range(num_blocks):
            nrows = np.random.randint(min_rows, max_rows + 1)
            ncols = np.random.randint(min_cols, max_cols + 1)
            shapes.append((nrows, ncols))
            row_size += nrows
            col_size += ncols

        return shapes, row_size, col_size

    @staticmethod
    def make_jac(n_dense_rows=0, row_density=1.0, n_dense_cols=0, col_density=1.0,
                 n_blocks=0, min_shape=(1,1), max_shape=(2,2), n_random_pts=0):
        if n_blocks > 0:
            shapes, nrows, ncols = TotJacBuilder.make_blocks(n_blocks, min_shape, max_shape)
            builder = TotJacBuilder(nrows + n_dense_rows, ncols + n_dense_cols)
            builder.add_block_diag(shapes, n_dense_rows, n_dense_cols)
        else:
            nrows, ncols = (100, 50)
            builder = TotJacBuilder(nrows, ncols)

        J = builder.J
        shape = J.shape

        # dense rows
        for row in range(n_dense_rows):
            builder.add_row(row, density=row_density)

        # dense cols
        for col in range(n_dense_cols):
            builder.add_col(col, density=col_density)

        builder.add_random_points(n_random_pts)

        return builder

    @staticmethod
    def eisenstat(n):
        """
        Return a builder containing an Eisenstat's example Jacobian of size n+1 x n.

        Should be colorable with n/2 + 2 colors using bidirectional coloring.

        The columns in Eisenstat's example are pairwise structurally nonorthogonal,
        so a fwd directional coloring would require n groups.
        """
        assert n >= 6, "Eisenstat's example must have n >= 6."
        assert n % 2 == 0, "Eisenstat's example must have even 'n'."

        D1 = np.eye(n // 2, dtype=int)
        D2 = np.eye(n // 2, dtype=int)
        D3 = np.eye(n // 2, dtype=int)
        B = np.ones((n // 2, n // 2), dtype=int)
        idxs = np.arange(n // 2, dtype=int)
        B[idxs, idxs] = 0
        C = np.ones((1, n // 2), dtype=int)
        O = np.zeros((1, n // 2), dtype=int)

        A1 = np.hstack([D1, D2])
        A2 = np.vstack([np.hstack([C, O]), np.hstack([D3, B])])

        A = np.vstack([A1, A2])

        builder = TotJacBuilder(n + 1, n)
        builder.J[:, :] = A

        return builder


def rand_jac():
    rnd = np.random.randint
    minr = rnd(1, 10)
    minc = rnd(1, 10)

    return  TotJacBuilder.make_jac(n_dense_rows=rnd(5), row_density=np.random.rand(),
                                   n_dense_cols=rnd(5), col_density=np.random.rand(),
                                   n_blocks=rnd(3,8),
                                   min_shape=(minr,minc),
                                   max_shape=(minr+rnd(10),minc+rnd(10)),
                                   n_random_pts=rnd(15))


if __name__ == '__main__':
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--eisenstat",
                        help="Build an Eisenstat's example matrix of size n+1 x n.",
                        action="store", type=int, default=-1, dest="eisenstat")
    parser.add_argument("-m", "--mode", type=str, dest="mode",
                        help="Direction of coloring (default is auto). Only used with -e.",
                        default="auto")
    parser.add_argument('-s', '--save', dest="save", default=None,
                        help="Output file for jacobian so it can be reloaded and colored using"
                        " various methods for comparison.")
    parser.add_argument('-l', '--load', dest="load", default=None,
                        help="Input file for jacobian so it can be reloaded and colored using"
                        " various methods for comparison.")

    options = parser.parse_args()

    if options.load is not None:
        with open(options.load, "rb") as f:
            builder = pickle.load(f)
    elif options.eisenstat > -1:
        builder = TotJacBuilder.eisenstat(options.eisenstat)
    else:  # just do a random matrix
        builder = rand_jac()

    builder.color(options.mode)
    builder.show()

    if options.save is not None:
        with open(options.save, "wb") as f:
            pickle.dump(builder, f)
