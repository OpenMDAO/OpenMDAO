import numpy as np

from openmdao.surrogate_models.nn_interpolators.nn_base import NNBase
from six.moves import range
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


class RBFInterpolator(NNBase):
    # Compactly Supported Radial Basis Function
    def _find_R(self, npp, T, loc):
        R = np.zeros((npp, self._ntpts), dtype="float")
        # Choose type of CRBF R matrix
        if self.comp == -1:
            # Comp #1 - a
            Cf = np.power((1. - T), 5)
            cb_poly = [5., 72., 48., 40., 8.]
            # Cb = (8. + (40. * T) + (48. * T * T) +
            #       (72. * T * T * T) + (5. * T * T * T * T))
        elif self.comp == -2:
            # Comp #2
            Cf = np.power((1. - T), 6)
            cb_poly = [5., 30., 72., 82., 36., 6.]
            # Cb = (6. + (36. * T) + (82. * T * T) + (72. * T * T * T) +
            #       (30. * T * T * T * T) + (5. * T * T * T * T * T))
        elif self.comp == -3:
            # Comp #3
            # Cf = np.ones_like(T)
            # Cb = np.sqrt((T * T) + 1.)

            # Re-arranged to fit polyval scheme below.
            Cf = np.sqrt(np.square(T) + 1.)
            cb_poly = [1.]
        else:
            # The above options did not specify a dimensional requirement
            # in the paper found but the rest are said to only be guaranteed
            # as positive definite iff the dimensional requirements are.
            # Because of this, the user can select 0 through 4 to adjust to a
            # level of order trying to be attained.
            dims = self._indep_dims + 1
            if dims <= 2:
                if self.comp == 0:
                    # This starts the dk comps, here d=1, k=0
                    Cf = 1. - T
                    cb_poly = [1.]
                    # Cb = T ** 0.
                elif self.comp == 1:
                    Cf = np.power(1. - T, 3.) / 12.
                    cb_poly = [3., 1.]
                    # Cb = 1. + (3. * T)
                elif self.comp == 2:
                    Cf = np.power(1. - T, 5.) / 840.
                    cb_poly = [24., 15., 3.]
                    # Cb = 3. + (15. * T) + (24. * T * T)
                elif self.comp == 3:
                    Cf = np.power(1. - T, 7.) / 151200.
                    cb_poly = [315., 285., 105., 15.]
                    # Cb = 15. + (105. * T) + (285. * T * T) + (315. * T * T * T)
                elif self.comp == 4:
                    Cf = np.power(1. - T, 9.) / 51891840.
                    cb_poly = [5760., 6795., 3555., 945., 105.]
                    # Cb = (105. + (945. * T) + (3555. * T * T) + (6795. * T * T * T) +
                    #       (5760. * T * T * T * T))
            elif dims <= 4:
                if self.comp == 0:
                    Cf = (1. - T)
                    # Cb = (1. - T)
                    cb_poly = [-1., 1.]
                elif self.comp == 1:
                    Cf = np.power(1. - T, 4) / 20.
                    # Cb = 1. + (4. * T)
                    cb_poly = [4., 1.]
                elif self.comp == 2:
                    Cf = np.power(1. - T, 6.) / 1680.
                    # Cb = 3. + (18. * T) + (35. * T * T)
                    cb_poly = [35., 18., 3.]
                elif self.comp == 3:
                    Cf = np.power(1. - T, 8.) / 332640.
                    cb_poly = [480., 375., 120., 15.]
                    # Cb = 15. + (120. * T) + (375. * T * T) + (480. * T * T * T)
                elif self.comp == 4:
                    Cf = np.power(1. - T, 10.) / 121080960.
                    cb_poly = [9009., 9450., 4410., 1050., 105.]
                    # Cb = (105. + (1050. * T) + (4410. * T * T) + (9450. * T * T * T) +
                    #       (9009. * T * T * T * T))
            elif dims <= 6:
                if self.comp == 0:
                    Cf = np.power(1. - T, 2.)
                    cb_poly = [-1., 1.]
                    # Cb = (1. - T)
                elif self.comp == 1:
                    Cf = np.power(1. - T, 5.) / 30.
                    cb_poly = [5., 1.]
                    # Cb = 1. + (5. * T)
                elif self.comp == 2:
                    Cf = np.power(1. - T, 7.) / 3024.
                    cb_poly = [48., 21., 3.]
                    # Cb = 3. + (21. * T) + (48. * T * T)
                elif self.comp == 3:
                    Cf = np.power(1. - T, 9.) / 665280.
                    cb_poly = [693., 477., 135., 15.]
                    # Cb = 15. + (135. * T) + (477. * T * T) + (693. * T * T * T)
                elif self.comp == 4:
                    Cf = np.power(1. - T, 11.) / 259459200.
                    cb_poly = [13440., 12705., 5355., 1155., 105.]
                    # Cb = (105. + (1155. * T) + (5355. * T * T) + (12705. * T * T * T) +
                    #       (13440. * T * T * T * T))
            else:
                # Although not listed, this is ideally for 8 dim or less
                if self.comp == 0:
                    Cf = np.power(1. - T, 2.)
                    cb_poly = [1., -2., 1.]
                    # Cb = (1. - T) * (1. - T)
                elif self.comp == 1:
                    Cf = np.power(1. - T, 6.) / 42.
                    cb_poly = [6., 1.]
                    # Cb = 1. + (6. * T)
                elif self.comp == 2:
                    Cf = np.power(1. - T, 8.) / 5040.
                    cb_poly = [63., 24., 3.]
                    # Cb = 3. + (24. * T) + (63. * T * T)
                elif self.comp == 3:
                    Cf = np.power(1. - T, 10.) / 1235520.
                    cb_poly = [960., 591., 150., 15.]
                    # Cb = 15. + (150. * T) + (591. * T * T) + (960. * T * T * T)
                elif self.comp == 4:
                    Cf = np.power(1. - T, 12.) / 518918400.
                    cb_poly = [19305., 16620., 6390., 1260., 105.]
                    # Cb = (105. + (1260. * T) + (6390. * T * T) + (16620. * T * T * T) +
                    #       (19305. * T * T * T * T))

        Cb = np.polyval(cb_poly, T)

        for i in range(npp):
            R[i, loc[i, :-1]] = Cf[i, :] * Cb[i, :]

        return R

    def _find_dR(self, PrdPts, ploc, pdist):
        T = (pdist[:, :-1] / pdist[:, -1:])
        # Solve for the gradient analytically
        # The first quantity needed is dRp/dt
        if self.comp == -1:
            frnt = np.power((1. - T), 4)
            dRp_poly = [-45., -556., -120., -144., 0.]
            # dRp = frnt * ((-5. * (8. + (40. * T) + (48. * T * T) +
            #                       (72. * T * T * T) + (5. * T * T * T * T))) +
            #               ((1. - T) * (40. + (96. * T) +
            #                            (216. * T * T) + (20. * T * T * T))))
        elif self.comp == -2:
            frnt = np.power((1. - T), 5.)
            dRp_poly = [-55., -275., -528., -440., -88., 0.]
            # dRp = frnt * ((-6. * (6. + (36. * T) +
            #                       (82. * T * T) + (72. * T * T * T) +
            #                       (30. * T * T * T * T) + (5. * T * T * T * T * T))) +
            #               ((1. - T)) * (36. + (164. * T) +
            #                             (216. * T * T) + (120. * T * T * T) +
            #                             (25. * T * T * T * T)))
        elif self.comp == -3:
            frnt = T / np.sqrt((T * T) * 1.)
            dRp_poly = [1.]
        else:
            dims = self._indep_dims + 1
            # Start dim dependent comps, review first occurrence for more info
            if dims <= 2:
                if self.comp == 0:
                    # This starts the dk comps(Wendland Functs), here d=1, k=0
                    frnt = 1.
                    dRp_poly = [-1.]
                    # dRp = -1.
                elif self.comp == 1:
                    frnt = 1.
                    dRp_poly = [1., -2., 1., 0.]
                    # dRp = -T * (1. - T) * (1. - T)
                elif self.comp == 2:
                    frnt = np.power(1. - T, 4.) / -20.
                    dRp_poly = [4., 1., 0.]
                    # dRp = frnt * (T + (4. * T * T))
                elif self.comp == 3:
                    frnt = np.power(1. - T, 6.) / -1680.
                    dRp_poly = [35., 18., 3., 0.]
                    # dRp = frnt * ((3. * T) + (18. * T * T) + (35. * T * T * T))
                elif self.comp == 4:
                    frnt = np.power(1. - T, 8.) / -22176.
                    dRp_poly = [32., 25., 8., 1., 0.]
                    # dRp = frnt * (T + (8. * T * T) + (25. * T * T * T) + (32. * T * T * T * T))
            elif dims <= 4:
                if self.comp == 0:
                    frnt = 1.
                    dRp_poly = [2., -2.]
                    # dRp = -2. * (1 - T)
                elif self.comp == 1:
                    frnt = 1.
                    dRp_poly = [1., -3., 3., -1., 0.]
                    # dRp = -T * (1. - T) * (1. - T) * (1. - T)
                elif self.comp == 2:
                    frnt = np.power(1. - T, 5.) / -30.
                    dRp_poly = [5., 1., 0.]
                    # dRp = frnt * (T + (5. * T * T))
                elif self.comp == 3:
                    frnt = np.power(1. - T,  7.) / -1008.
                    dRp_poly = [16., 7., 1., 0.]
                    # dRp = frnt * (T + (7. * T * T) + (16. * T * T * T))
                elif self.comp == 4:
                    frnt = np.power(1. - T, 9.) / -221760.
                    dRp_poly = [231., 159., 45., 5., 0.]
                    # dRp = frnt * ((5. * T) + (45. * T * T) + (159. * T * T * T) + (231. * T * T * T * T))
            elif dims <= 6:
                if self.comp == 0:
                    frnt = 1.
                    dRp_poly = [-3., 6., -3.]
                    # dRp = -3. * (1. - T) * (1. - T)
                elif self.comp == 1:
                    frnt = 1.
                    dRp_poly = [-1., 4., -6., 4., -1., 0.]
                    # dRp = -T * ((1. - T) ** 4)
                elif self.comp == 2:
                    frnt = np.power(1. - T, 6.) / -42.
                    dRp_poly = [6., 1., 0.]
                    # dRp = frnt * (T + (6. * T * T))
                elif self.comp == 3:
                    frnt = np.power(1. - T, 8) / -1680.
                    dRp_poly = [21., 8., 1., 0.]
                    # dRp = frnt * (T + (8. * T * T) + (21. * T * T * T))
                elif self.comp == 4:
                    frnt = np.power(1. - T, 10.) / -411840.
                    dRp_poly = [320., 197., 50., 5., 0.]
                    # dRp = frnt * ((5. * T) + (50. * T * T) + (197. * T * T * T) + (320. * T * T * T * T))
            else:
                # Although not listed, this is ideally for 8 dim or less
                if self.comp == 0:
                    frnt = 1.
                    dRp_poly = [4., -12., 12., -4.]
                    # dRp = -4. * (1. - T) * (1. - T) * (1. - T)
                elif self.comp == 1:
                    frnt = 1.
                    dRp_poly = [1., -5., 10., -10., 5., -1., 0.]
                    # dRp = -T * ((1. - T) ** 5)
                elif self.comp == 2:
                    frnt = np.power(1. - T, 7.) / -56.
                    dRp_poly = [7., 1., 0.]
                    # dRp = frnt * (T + (7. * T * T))
                elif self.comp == 3:
                    frnt = np.power(1. - T, 9.) / -7920.
                    dRp_poly = [80., 27., 3., 0.]
                    # dRp = frnt * ((3. * T) + (27. * T * T) + (80. * T * T * T))
                elif self.comp == 4:
                    frnt = np.power(1. - T, 11.) / -720720.
                    dRp_poly = [429., 239., 55., 5., 0.]
                    # dRp = frnt * ((5. * T) + (55. * T * T) + (239. * T * T * T) + (429. * T * T * T * T))

        dRp = frnt * np.polyval(dRp_poly, T)

        # dt/dx becomes unstable at the training points, so perturb T slightly.
        zero_idx = np.where(T==0.0)
        T[zero_idx] += 1.0e-11

        # Now need dt/dx
        xpi = np.subtract(PrdPts, self._tp[ploc[:, :-1], :])
        xpm = PrdPts - self._tp[ploc[:, -1:], :]
        dtx = (xpi - (np.square(T) * xpm)) / (np.square(pdist[:, -1:, :]) * T)

        # The gradient then is the summation across neighs of w*df/dt*dt/dx
        grad = np.einsum('ijk,ijk,ijl...->ilk...', dRp, dtx, self.weights[ploc[:, :-1]])

        return grad.reshape((PrdPts.shape[0], self._dep_dims, self._indep_dims))

    def __init__(self, training_points, training_values, num_leaves=2, n=5, comp=2):
        super(RBFInterpolator, self).__init__(training_points, training_values, num_leaves)

        if self._ntpts < n:
            raise ValueError('RBFInterpolator only given {0} training points, '
                             'but requested n={1}.'
                             .format(self._ntpts, n))

        # Comp is an arbitrary value that picks a function to use
        self.comp = comp

        # For weights, first find the training points radial neighbors
        tdist, tloc = self._KData.query(self._tp, n)
        Tt = tdist[:, :-1] / tdist[:, -1:]
        # Next determine weight matrix
        Rt = self._find_R(self._ntpts, Tt, tloc)
        weights = (spsolve(csc_matrix(Rt), self._tv))[..., np.newaxis]

        self.N = n
        self.weights = weights

    def __call__(self, prediction_points):

        if len(prediction_points.shape) == 1:
            # Reshape vector to n x 1 array
            prediction_points.shape = (1, prediction_points.shape[0])

        normalized_pts = (prediction_points - self._tpm) / self._tpr
        nppts = normalized_pts.shape[0]
        # Setup prediction points and find their radial neighbors
        ndist, nloc = self._KData.query(normalized_pts, self.N)
        # Check if complex step is being run
        if np.any(np.abs(normalized_pts[0, :].imag)) > 0:
            dimdiff = np.subtract(normalized_pts.reshape((nppts, 1, self._indep_dims)),
                                  self._tp[nloc, :])
            # KD Tree ignores imaginary part, muse redo ndist if complex
            ndist = np.sqrt(np.sum((dimdiff * dimdiff), axis=2))

        # Take farthest distance of each point
        Tp = ndist[:, :-1] / ndist[:, -1:]

        Rp = self._find_R(nppts, Tp, nloc)
        predz = ((np.dot(Rp, self.weights[..., 0]) * self._tvr) + self._tvm).reshape(nppts, self._dep_dims)

        self._pt_cache = (normalized_pts, ndist, nloc)

        return predz

    def gradient(self, prediction_points):

        if len(prediction_points.shape) == 1:
            # Reshape vector to n x 1 array
            prediction_points.shape = (1, prediction_points.shape[0])

        normalized_pts = (prediction_points - self._tpm) / self._tpr
        # Setup prediction points and find their radial neighbors
        if self._pt_cache is not None and \
                np.allclose(self._pt_cache[0], normalized_pts):
            pdist, ploc = self._pt_cache[1:]
        else:
            pdist, ploc = self._KData.query(normalized_pts, self.N)

        # Find Gradient
        grad = self._find_dR(normalized_pts[:, np.newaxis, :], ploc,
                           pdist[:, :, np.newaxis]) * (self._tvr[..., np.newaxis] / self._tpr)

        return grad
