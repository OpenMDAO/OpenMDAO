"""Define the AssembledJacobian class."""
import sys
from collections import defaultdict

import numpy as np

from openmdao.jacobians.jacobian import SplitJacobian
from openmdao.matrices.dense_matrix import DenseMatrix
from openmdao.matrices.coo_matrix import COOMatrix
from openmdao.matrices.csr_matrix import CSRMatrix
from openmdao.matrices.csc_matrix import CSCMatrix
from openmdao.utils.iter_utils import meta2range_iter

_empty_dict = {}


class AssembledJacobian(SplitJacobian):
    """
    Assemble a global <Jacobian>.

    Parameters
    ----------
    matrix_class : type
        Class to use to create internal matrices.
    system : System
        Parent system to this jacobian.

    Attributes
    ----------
    _view_ranges : dict
        Maps system pathnames to jacobian sub-view ranges
    _int_mtx : <Matrix>
        Global internal Jacobian.
    _ext_mtx : {str: <Matrix>, ...}
        External Jacobian for each viewing subsystem.
    _mask_caches : dict
        Contains masking arrays for when a subset of the variables are present in a vector, keyed
        by the input._names set.
    _matrix_class : type
        Class used to create Matrix objects.
    _subjac_iters : dict
        Mapping of system pathname to tuple of lists of absolute key tuples used to index into
        the jacobian.
    _in_ranges : dict
        Column ranges for inputs.
    _out_ranges : dict
        Row ranges for outputs.
    """

    def __init__(self, matrix_class, system):
        """
        Initialize all attributes.
        """
        global Component
        # avoid circular imports
        from openmdao.core.component import Component

        super().__init__(system)
        self._view_ranges = {}
        self._int_mtx = None
        self._ext_mtx = {}
        self._mask_caches = {}
        self._matrix_class = matrix_class
        self._out_ranges = self._get_ranges(system, 'output')
        self._in_ranges = self._get_ranges(system, 'input')
        self._subjac_iters = defaultdict(lambda: None)

    def _get_ranges(self, system, vtype):
        """
        Return an ordered dict of ranges for each var of a particular type (input or output).

        Parameters
        ----------
        system : System
            System owning this jacobian.
        vtype : str
            Type of variable, must be one of ('input', 'output').

        Returns
        -------
        dict
            Tuples of the form (start, end) keyed on variable name.
        """
        return {
            name: (start, end)
            for name, start, end in meta2range_iter(system._var_abs2meta[vtype].items(),
                                                    size_name='size')
        }

    def _initialize(self, system):
        """
        Allocate the global matrices.

        Parameters
        ----------
        system : System
            Parent system to this jacobian.
        """
        # var_indices are the *global* indices for variables on this proc
        # is_top = system.pathname == ''
        self._int_mtx = int_mtx = self._matrix_class(system.comm)
        ext_mtx = self._matrix_class(system.comm)

        int_subjacs, ext_subjacs = self._get_split_subjacs(system)
        for key, subjac in int_subjacs.items():
            int_mtx._add_submat(key, subjac.info, subjac.row_slice.start, subjac.col_slice.start,
                                subjac.src_indices, subjac.shape, subjac.factor)

        if ext_subjacs:
            for key, subjac in ext_subjacs.items():
                ext_mtx._add_submat(key, subjac.info, subjac.row_slice.start,
                                    subjac.col_slice.start, None, subjac.shape)

        out_size = len(system._outputs)
        int_mtx._build(out_size, out_size, system)

        if ext_mtx._submats:
            ext_mtx._build(out_size, len(system._dinputs))
        else:
            ext_mtx = None

        self._ext_mtx[system.pathname] = ext_mtx

    def _init_ranges(self, system):
        in_ranges = self._in_ranges
        out_ranges = self._out_ranges

        input_names = list(system._var_abs2meta['input'])
        if input_names:
            min_in_offset = in_ranges[input_names[0]][0]
            max_in_offset = in_ranges[input_names[-1]][1]
        else:
            min_in_offset = sys.maxsize
            max_in_offset = 0

        output_names = list(system._var_abs2meta['output'])
        if output_names:
            min_res_offset = out_ranges[output_names[0]][0]
            max_res_offset = out_ranges[output_names[-1]][1]
        else:
            min_res_offset = sys.maxsize
            max_res_offset = 0

        self._view_ranges[system.pathname] = (min_res_offset, max_res_offset,
                                              min_in_offset, max_in_offset)

    def _init_view(self, system):
        """
        Determine the _ext_mtx for a sub-view of the assembled jacobian.

        Parameters
        ----------
        system : <System>
            The system being solved using a sub-view of the jacobian.
        """
        ranges = self._view_ranges[system.pathname]

        ext_mtx = self._matrix_class(system.comm, False)
        conns = {} if isinstance(system, Component) else system._conn_global_abs_in2out

        iproc = system.comm.rank
        sizes = system._var_sizes['input']
        abs2idx = system._var_allprocs_abs2idx
        in_offset = {n: np.sum(sizes[iproc, :abs2idx[n]]) for n in
                     system._var_abs2meta['input'] if n not in conns}

        subjacs_info = self._subjacs_info

        sizes = system._var_sizes['output']
        for s in system.system_iter(recurse=True, include_self=True, typ=Component):
            for res_abs_name in s._var_abs2meta['output']:
                res_offset = np.sum(sizes[iproc, :abs2idx[res_abs_name]])

                for in_abs_name in s._var_abs2meta['input']:
                    if in_abs_name not in conns:  # unconnected input
                        abs_key = (res_abs_name, in_abs_name)

                        if abs_key not in subjacs_info:
                            continue

                        info = subjacs_info[abs_key]
                        ext_mtx._add_submat(abs_key, info, res_offset - ranges[0],
                                            in_offset[in_abs_name] - ranges[2], None, info['shape'])

        if ext_mtx._submats:
            ext_mtx._build(len(system._doutputs), len(system._dinputs))
        else:
            ext_mtx = None

        self._ext_mtx[system.pathname] = ext_mtx

    def _get_subjac_iters(self, system):
        # this determines the subjacs that get updated during _update()

        global _empty_dict

        subjac_iters = self._subjac_iters[system.pathname]
        if subjac_iters is None:
            int_mtx = self._int_mtx
            ext_mtx = self._ext_mtx[system.pathname]
            subjacs = system._subjacs_info

            if isinstance(system, Component):
                global_conns = _empty_dict
            else:
                global_conns = system._conn_global_abs_in2out

            output_names = system._var_abs2meta['output']
            input_names = system._var_abs2meta['input']

            rev_conns = defaultdict(list)
            for tgt, src in global_conns.items():
                rev_conns[src].append(tgt)

            # This is the level where the AssembledJacobian is slotted.
            # The of and wrt are the inputs and outputs that it sees, if they are in the subjacs.
            # TODO - For top level FD, the subjacs might not contain all derivs.

            iters = []
            iters_in_ext = []

            for abs_key in subjacs:
                _, wrtname = abs_key
                if wrtname in output_names:
                    if abs_key in int_mtx._submats:
                        iters.append(abs_key)
                    else:
                        # This happens when the src is an indepvarcomp that is
                        # contained in the system.
                        of, wrt = abs_key
                        if wrt in rev_conns:
                            for tgt in rev_conns[wrt]:
                                if (of, tgt) in int_mtx._submats:
                                    iters.append(abs_key)
                                    break
                elif wrtname in input_names:  # wrt is an input
                    if wrtname in global_conns:
                        iters.append(abs_key)
                    elif ext_mtx is not None:
                        iters_in_ext.append(abs_key)

            self._subjac_iters[system.pathname] = subjac_iters = (iters, iters_in_ext)

        return subjac_iters

    def _update(self, system):
        """
        Read the user's sub-Jacobians and set into the global matrix.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.
        """
        # _initialize has been delayed until the first _update call
        if self._int_mtx is None:
            self._initialize(system)

        int_mtx = self._int_mtx
        ext_mtx = self._ext_mtx[system.pathname]
        int_subjacs, ext_subjacs = self._get_split_subjacs(system)

        int_mtx._pre_update()
        if ext_mtx is not None:
            ext_mtx._pre_update()

        if self._randgen and system._problem_meta['randomize_subjacs']:
            for key, subjac in int_subjacs.items():
                int_mtx._update_submat(key, self._randomize_subjac(subjac.get_val(), key))

            if ext_subjacs:
                for key, subjac in ext_subjacs.items():
                    ext_mtx._update_submat(key, self._randomize_subjac(subjac.get_val(), key))
        else:

            for key, subjac in int_subjacs.items():
                int_mtx._update_submat(key, subjac.get_val())

            if ext_subjacs:
                for key, subjac in ext_subjacs.items():
                    ext_mtx._update_submat(key, subjac.get_val())

        int_mtx._post_update()

        if ext_mtx is not None:
            ext_mtx._post_update()

        if self._under_complex_step:
            # If we create a new _int_mtx while under complex step, we need to convert it to a
            # complex data type.
            self._int_mtx.set_complex_step_mode(True)

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
        ext_mtx = self._ext_mtx[system.pathname]
        if ext_mtx is None and not d_outputs._names:  # avoid unnecessary unscaling
            return

        int_mtx = self._int_mtx
        with system._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
            do_mask = ext_mtx is not None and d_inputs._names
            if do_mask:
                # Masking
                try:
                    mask = self._mask_caches[(d_inputs._names, mode)]
                except KeyError:
                    mask = d_inputs.get_mask()
                    self._mask_caches[(d_inputs._names, mode)] = mask

            dresids = d_residuals.asarray()

            if mode == 'fwd':
                if d_outputs._names:
                    dresids += int_mtx._prod(d_outputs.asarray(), mode)
                if do_mask:
                    dresids += ext_mtx._prod(d_inputs.asarray(mask=mask), mode)  # , mask=mask)

            else:  # rev
                if d_outputs._names:
                    d_outputs += int_mtx._prod(dresids, mode)
                if do_mask:
                    arr = ext_mtx._prod(dresids, mode)  # , mask=mask)
                    arr[mask] = 0.0
                    d_inputs += arr

    def set_complex_step_mode(self, active):
        """
        Turn on or off complex stepping mode.

        When turned on, the value in each subjac is cast as complex, and when turned
        off, they are returned to real values.

        Parameters
        ----------
        active : bool
            Complex mode flag; set to True prior to commencing complex step.
        """
        super().set_complex_step_mode(active)

        if self._int_mtx is not None:
            self._int_mtx.set_complex_step_mode(active)
            for mtx in self._ext_mtx.values():
                if mtx:
                    mtx.set_complex_step_mode(active)


class DenseJacobian(AssembledJacobian):
    """
    Assemble dense global <Jacobian>.

    Parameters
    ----------
    system : System
        Parent system to this jacobian.
    """

    def __init__(self, system):
        """
        Initialize all attributes.
        """
        super().__init__(DenseMatrix, system=system)


class COOJacobian(AssembledJacobian):
    """
    Assemble sparse global <Jacobian> in Coordinate list format.

    Parameters
    ----------
    system : System
        Parent system to this jacobian.
    """

    def __init__(self, system):
        """
        Initialize all attributes.
        """
        super().__init__(COOMatrix, system=system)


class CSRJacobian(AssembledJacobian):
    """
    Assemble sparse global <Jacobian> in Compressed Row Storage format.

    Parameters
    ----------
    system : System
        Parent system to this jacobian.
    """

    def __init__(self, system):
        """
        Initialize all attributes.
        """
        super().__init__(CSRMatrix, system=system)


class CSCJacobian(AssembledJacobian):
    """
    Assemble sparse global <Jacobian> in Compressed Col Storage format.

    Parameters
    ----------
    system : System
        Parent system to this jacobian.
    """

    def __init__(self, system):
        """
        Initialize all attributes.
        """
        super().__init__(CSCMatrix, system=system)
