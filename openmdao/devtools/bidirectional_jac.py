"""
A tool to make it easier to investigate coloring of jacobians with different sparsity structures.
"""

import sys
import numpy as np
from openmdao.utils.array_utils import array_viz
from openmdao.utils.coloring import get_simul_meta, simul_coloring_summary, _solves_info

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

    def color(self, simul_coloring_excludes=(), stream=sys.stdout):
        self.coloring = get_simul_meta(None, include_sparsity=False, setup=False, run_model=False,
                                       bool_jac=self.J,
                                       simul_coloring_excludes=simul_coloring_excludes,
                                       stream=stream)
        return self.coloring

    def show(self, stream=sys.stdout):
        array_viz(self.J)
        print("Shape:", self.J.shape, file=stream)
        print("Density:", np.count_nonzero(self.J) / self.J.size)
        simul_coloring_summary(self.coloring, stream=stream)

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

if __name__ == '__main__':
    # Running this from the command line will just give a visualization of a random jacobian and
    # its coloring.

    rnd = np.random.randint
    minr = rnd(1, 10)
    minc = rnd(1, 10)
    builder = TotJacBuilder.make_jac(n_dense_rows=rnd(3), row_density=np.random.rand(),
                                     n_dense_cols=rnd(3), col_density=np.random.rand(),
                                     n_blocks=rnd(3,8),
                                     min_shape=(minr,minc),
                                     max_shape=(minr+rnd(5),minc+rnd(5)),
                                     n_random_pts=rnd(2))

    with open("color_test.out", "w") as f:
        builder.color(stream=f)

    builder.show()
