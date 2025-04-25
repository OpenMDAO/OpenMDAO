"""
BlockJacobian is a Jacobian that is a collection of sub-Jacobians.

"""

from openmdao.jacobians.jacobian import Jacobian


class BlockJacobian(Jacobian):
    """
    A BlockJacobian is a Jacobian that is a collection of sub-Jacobians.

    Parameters
    ----------
    system : System
        System that is updating this jacobian.

    Attributes
    ----------
    _subjacs : dict
        Dictionary of sub-Jacobians.
    _ordered : bool
        Whether the sub-Jacobians are ordered.
    """

    def __init__(self, system):
        """
        Initialize block-sparse Jacobian with sub-jacobian information.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.
        """
        super().__init__(system)
        self._ordered = False

    def __getitem__(self, key):
        """
        Get sub-Jacobian.

        Parameters
        ----------
        key : (str, str)
            Promoted or relative name pair of sub-Jacobian.

        Returns
        -------
        ndarray or spmatrix or list[3]
            sub-Jacobian as an array, sparse mtx, or AIJ/IJ list or tuple.
        """
        try:
            return self._subjacs[self._get_abs_key(key)].get_val()
        except KeyError:
            msg = '{}: Variable name pair ("{}", "{}") not found.'
            raise KeyError(msg.format(self.msginfo, key[0], key[1]))

    def __setitem__(self, key, subjac):
        """
        Set sub-Jacobian.

        Parameters
        ----------
        key : (str, str)
            Promoted or relative name pair of sub-Jacobian.
        subjac : int or float or ndarray or sparse matrix
            sub-Jacobian as a scalar, vector, array, or AIJ list or tuple.
        """
        abs_key = self._get_abs_key(key)
        if abs_key is None:
            raise KeyError(f'{self.msginfo}: Variable name pair {key} not found.')

        print(f"setting {key} to {subjac}")
        # You can only set declared subjacobians.
        try:
            self._subjacs[abs_key].set_val(subjac)
        except KeyError:
            raise KeyError(f'{self.msginfo}: Variable name pair {key} must first be declared.')
        except ValueError as err:
            raise ValueError(f"{self.msginfo}: for subjacobian {key}: {err}")

    # def _get_subjacs(self, system):
    #     """
    #     Get the sub-Jacobians.

    #     Parameters
    #     ----------
    #     system : System
    #         System that is updating this jacobian.

    #     Returns
    #     -------
    #     dict
    #         Dictionary of sub-Jacobians.
    #     """
    #     if not self._ordered:
    #         # determine the set of remote keys (keys where either of or wrt is remote somewhere)
    #         # only if we're under MPI with comm size > 1 and the given system is a Group that
    #         # computes its derivatives using finite difference or complex step.
    #         include_remotes = system.pathname and \
    #             system.comm.size > 1 and system._owns_approx_jac and system._subsystems_allprocs

    #         subjacs = self._subjacs

    #         if include_remotes:
    #             ofnames = system._var_allprocs_abs2meta['output']
    #             wrtnames = system._var_allprocs_abs2meta
    #         else:
    #             ofnames = system._var_abs2meta['output']
    #             wrtnames = system._var_abs2meta

    #         self._subjacs = {}
    #         for res_name in ofnames:
    #             for type_ in ('output', 'input'):
    #                 for name in wrtnames[type_]:
    #                     key = (res_name, name)
    #                     if key in subjacs:
    #                         self._subjacs[key] = subjacs[key]

    #         self._ordered = True

    #     return self._subjacs

    def _apply(self, system, d_inputs, d_outputs, d_residuals, mode):
        """
        Compute matrix-vector product.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.
        d_inputs : Vector
            inputs linear vector.
        d_outputs : Vector
            outputs linear vector.
        d_residuals : Vector
            residuals linear vector.
        mode : str
            'fwd' or 'rev'.
        """
        randgen = self._randgen
        d_out_names = d_outputs._names
        d_inp_names = d_inputs._names

        with system._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
            if randgen is None:
                if mode == 'fwd':
                    for key, subjac in self._get_subjacs(system).items():
                        of, wrt = key
                        if wrt in d_inp_names or wrt in d_out_names:
                            subjac.apply_fwd(d_inputs, d_outputs, d_residuals)
                else:  # rev
                    for key, subjac in self._get_subjacs(system).items():
                        of, wrt = key
                        if wrt in d_inp_names or wrt in d_out_names:
                            subjac.apply_rev(d_inputs, d_outputs, d_residuals)
            else:
                if mode == 'fwd':
                    for key, subjac in self._get_subjacs(system).items():
                        of, wrt = key
                        if wrt in d_inp_names or wrt in d_out_names:
                            subjac.apply_rand_fwd(d_inputs, d_outputs, d_residuals, randgen)
                else:  # rev
                    for key, subjac in self._get_subjacs(system).items():
                        of, wrt = key
                        if wrt in d_inp_names or wrt in d_out_names:
                            subjac.apply_rand_rev(d_inputs, d_outputs, d_residuals, randgen)
