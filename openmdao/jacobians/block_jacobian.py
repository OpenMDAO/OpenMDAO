import numpy as np

from openmdao.jacobians.jacobian import Jacobian


class BlockJacobian(Jacobian):
    def __init__(self, system):
        super().__init__(system)

        # def __init__(self, M: List[int], N: List[int], blocks: List[Tuple[int, int, Union[np.ndarray, sparse.csr_matrix, None], str]]):
        """
        Initialize block-sparse Jacobian with caller-provided block types.

        Args:
            M: List of row variable sizes [M_1, ..., M_m]
            N: List of column variable sizes [N_1, ..., N_n]
            blocks: List of (i, j, J_ij, block_type) for non-zero sub-Jacobians
                    - J_ij: None (identity), 1D array (diagonal), np.ndarray (dense), or sparse.csr_matrix (sparse)
                    - block_type: "identity", "diagonal", "dense", or "sparse"
        """
        rowvars = {}
        colvars = {}
        for i, v in enumerate(self.system.get_var_names('input')):
            rowvars[v] = i
        for i, v in enumerate(self.system.get_var_names('output')):
            colvars[v] = i

        self.M = np.array(M, dtype=np.int64)
        self.N = np.array(N, dtype=np.int64)
        self.m, self.n = len(M), len(N)

        self.blocks = blocks

        # Validate inputs
        self._validate()

        # Compute offsets
        self.R = np.cumsum([0] + list(M))
        self.C = np.cumsum([0] + list(N))

    def matvec(self, v):
        """
        Compute y = J * v.

        Args:
            v: Input vector of size sum(N_j)

        Returns:
            y: Output vector of size sum(M_i)
        """
        if v.size != self.C[-1]:
            raise ValueError(f"Input vector size {v.size} does not match expected {self.C[-1]}")

        y = np.zeros(self.R[-1], dtype=v.dtype)

        for i, j, J_ij, block_type in self.blocks:
            v_j = v[self.C[j] : self.C[j] + self.N[j]]
            if block_type == "identity":
                y[self.R[i] : self.R[i] + self.M[i]] += v_j
            elif block_type == "diagonal":
                k = min(self.M[i], self.N[j])
                y[self.R[i] : self.R[i] + k] += J_ij * v_j[:k]
            else:  # dense or sparse
                y[self.R[i] : self.R[i] + self.M[i]] += J_ij @ v_j

        return y

# Example usage
M = [3, 2]  # Row sizes
N = [2, 3, 3]  # Column sizes
v = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)  # Input vector: size 2+3+3=8

# Caller-provided blocks: J_00 = diag([2, 3]), J_10 = I, J_01 = dense
blocks = [
    (0, 1, np.array([[1, 3, 0], [0, 1, 0], [4, -1, 0]], dtype=np.float64), "dense"),  # J_01: 3x3
    (1, 0, None, "identity"),  # J_10 = I (2x2)
    (0, 2, np.array([2, 3, 1], dtype=np.float64), "diagonal"),  # J_00 = diag([2, 3, 1]) (3x3)
]
jac = BlockJacobian(M, N, blocks)
y = jac.matvec(v)
print(y)  # Output: [5, 8, 0, 1, 2]