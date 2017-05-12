"""
Class definition for BaseRecorder, the base class for all recorders.
"""
from fnmatch import fnmatchcase
import sys

from six import StringIO

from openmdao.utils.options_dictionary import OptionsDictionary
from openmdao.utils.general_utils import warn_deprecation
from openmdao.core.system import System
from openmdao.core.driver import Driver
from openmdao.solvers.solver import Solver

import warnings


class BaseRecorder(object):
    """
    Base class for all case recorders and is not a functioning case recorder on its own.

    Options
    -------
    options['record_metadata'] :  bool(True)
        Tells recorder whether to record variable attribute metadata.
    options['record_outputs'] :  bool(True)
        Tells recorder whether to record the outputs vector.
    options['record_inputs'] :  bool(False)
        Tells recorder whether to record the inputs vector.
    options['record_residuals'] :  bool(False)
        Tells recorder whether to record the residuals vector.
    options['record_derivatives'] :  bool(True)
        Tells recorder whether to record derivatives that are requested by a `Driver`.
    options['includes'] :  list of strings
        Patterns for variables to include in recording.
    options['excludes'] :  list of strings
        Patterns for variables to exclude in recording (processed after includes).
    """

    def __init__(self):
        """
        initialize.
        """
        self.options = OptionsDictionary()
        # Options common to all objects
        self.options.declare('record_metadata', type_=bool, desc='Record metadata', default=True)
        self.options.declare('includes', type_=list, default=['*'],
                             desc='Patterns for variables to include in recording')
        self.options.declare('excludes', type_=list, default=[],
                             desc='Patterns for vars to exclude in recording '
                                  '(processed post-includes)')

        # Old options that will be deprecated
        self.options.declare('record_unknowns', type_=bool, default=False,
                             desc='Deprecated option to record unknowns.')
        self.options.declare('record_params', type_=bool, default=False,
                             desc='Deprecated option to record params.',)
        self.options.declare('record_resids', type_=bool, default=False,
                             desc='Deprecated option to record residuals.')
        self.options.declare('record_derivs', type_=bool, default=False,
                             desc='Deprecated option to record derivatives.')
        # System options
        self.options.declare('record_outputs', type_=bool, default=True,
                             desc='Set to True to record outputs at the system level')
        self.options.declare('record_inputs', type_=bool, default=True,
                             desc='Set to True to record inputs at the system level')
        self.options.declare('record_residuals', type_=bool, default=True,
                             desc='Set to True to record residuals at the system level')
        self.options.declare('record_derivatives', type_=bool, default=False,
                             desc='Set to True to record derivatives at the system level')
        # Driver options
        self.options.declare('record_desvars', type_=bool, default=True,
                             desc='Set to True to record design variables at the driver level')
        self.options.declare('record_responses', type_=bool, default=False,
                             desc='Set to True to record responses at the driver level')
        self.options.declare('record_objectives', type_=bool, default=False,
                             desc='Set to True to record objectives at the driver level')
        self.options.declare('record_constraints', type_=bool, default=False,
                             desc='Set to True to record constraints at the driver level')
        # Solver options
        self.options.declare('record_abs_error', type_=bool, default=True,
                             desc='Set to True to record absolute error at the solver level')
        self.options.declare('record_rel_error', type_=bool, default=True,
                             desc='Set to True to record relative error at the solver level')
        self.options.declare('record_output', type_=bool, default=False,
                             desc='Set to True to record output at the solver level')
        self.options.declare('record_solver_residuals', type_=bool, default=False,
                             desc='Set to True to record residuals at the solver level')

        self.out = None

        # global counter that is used in iteration coordinate
        self._counter = 0

        # dicts in which to keep the included items for recording
        self._filtered_driver = {}
        self._filtered_system = {}
        self._filtered_solver = {}

    def startup(self, object_requesting_recording):
        """
        Prepare for a new run.

        Args
        ----
        object_requesting_recording :
            Object to which this recorder is attached.
        """
        # Deprecated options here, but need to preserve backward compatibility if possible.
        if self.options['record_params']:
            warn_deprecation("record_params is deprecated, please use record_inputs.")
            # set option to what the user intended.
            self.options['record_inputs'] = True

        if self.options['record_unknowns']:
            warn_deprecation("record_ is deprecated, please use record_inputs.")
            # set option to what the user intended.
            self.options['record_outputs'] = True

        if self.options['record_resids']:
            warn_deprecation("record_params is deprecated, please use record_inputs.")
            # set option to what the user intended.
            self.options['record_residuals'] = True

        # Compute the inclusion lists

        if (isinstance(object_requesting_recording, System)):
            myinputs = myoutputs = myresiduals = set()
            incl = self.options['includes']
            excl = self.options['excludes']

            if self.options['record_inputs']:
                if object_requesting_recording._inputs:
                    myinputs = [n for n in object_requesting_recording._inputs._names
                                if self._check_path(n, incl, excl)]
            if self.options['record_outputs']:
                if object_requesting_recording._outputs:
                    myoutputs = [n for n in object_requesting_recording._outputs._names
                                 if self._check_path(n, incl, excl)]
                if self.options['record_residuals']:
                    myresiduals = myoutputs  # outputs and residuals have same names
            elif self.options['record_residuals']:
                if object_requesting_recording._residuals:
                    myresiduals = [n for n in object_requesting_recording._residuals._names
                                   if self._check_path(n, incl, excl)]

            # if self.options['record_derivs']:
            #     myinputs = [n for n in object_requesting_recording._derivs._names
            #                 if self._check_path(n, incl, excl)]

            self._filtered_system = {
                'i': myinputs,
                'o': myoutputs,
                'r': myresiduals
                # 'd': myderivatives
            }

        if (isinstance(object_requesting_recording, Driver)):
            mydesvars = myobjectives = myconstraints = myresponses = set()
            incl = self.options['includes']
            excl = self.options['excludes']

            if self.options['record_desvars']:
                mydesvars = [n for n in object_requesting_recording._designvars
                             if self._check_path(n, incl, excl)]

            if self.options['record_objectives']:
                myobjectives = [n for n in object_requesting_recording._objs
                                if self._check_path(n, incl, excl)]

            if self.options['record_constraints']:
                myconstraints = [n for n in object_requesting_recording._cons
                                 if self._check_path(n, incl, excl)]

            if self.options['record_responses']:
                myresponses = [n for n in object_requesting_recording._responses
                               if self._check_path(n, incl, excl)]

            self._filtered_driver = {
                'des': mydesvars,
                'obj': myobjectives,
                'con': myconstraints,
                'res': myresponses
            }

        if (isinstance(object_requesting_recording, Solver)):
            myabserr = myrelerr = myderivs = set()
            incl = self.options['includes']
            excl = self.options['excludes']

            if self.options['record_abs_error']:
                myabserr = [n for n in object_requesting_recording.get_abs_err()
                            if self._check_path(n, incl, excl)]

            if self.options['record_rel_error']:
                myrelerr = [n for n in object_requesting_recording.get_rel_err()
                            if self._check_path(n, incl, excl)]

            if self.options['record_derivatives']:
                myderivs = [n for n in object_requesting_recording.get_derivs()
                            if self._check_path(n, incl, excl)]

            self._filtered_solver = {
                'ae': myabserr,
                're': myrelerr,
                'r': myderivs
            }

    def _check_path(self, path, includes, excludes):
        """
        Return True if `path` should be recorded.
        """
        # First see if it's included
        for pattern in includes:
            if fnmatchcase(path, pattern):
                # We found a match. Check to see if it is excluded.
                for ex_pattern in excludes:
                    if fnmatchcase(path, ex_pattern):
                        return False
                return True

        # Did not match anything in includes.
        return False

    def record_metadata(self, object_requesting_recording):
        """
        Write the metadata of the given object.

        Args
        ----
        object_requesting_recording :
            System, Solver, Driver in need of recording.
        """
        raise NotImplementedError()

    # TODO_RECORDER: change the signature to match what we decided to do with sqlite, hdf5,...
    def record_iteration(self, object_requesting_recording, metadata):
        """
        Write the provided data.

        Args
        ----
        inputs : dict
            Dictionary containing inputs.

        outputs : dict
            Dictionary containing outputs and states.

        residuals : dict
            Dictionary containing residuals.

        metadata : dict, optional
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """
        self._counter += 1
        # raise NotImplementedError()

    def close(self):
        """
        Close `out` unless it's ``sys.stdout``, ``sys.stderr``, or StringIO.

        Note that a closed recorder will do nothing in :meth:`record`, and
        closing a closed recorder also does nothing.
        """
        # Closing a StringIO deletes its contents.
        if self.out not in (None, sys.stdout, sys.stderr):
            if not isinstance(self.out, StringIO):
                self.out.close()
            self.out = None
