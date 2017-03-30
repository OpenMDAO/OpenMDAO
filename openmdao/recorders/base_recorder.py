""" Class definition for BaseRecorder, the base class for all recorders."""

from fnmatch import fnmatchcase
import sys

from six import StringIO

# from openmdao.util.options import OptionsDictionary

class BaseRecorder(object):
    """ This is a base class for all case recorders and is not a functioning
    case recorder on its own.

    Options
    -------
    options['record_metadata'] :  bool(True)
        Tells recorder whether to record variable attribute metadata.
    options['record_unknowns'] :  bool(True)
        Tells recorder whether to record the unknowns vector.
    options['record_params'] :  bool(False)
        Tells recorder whether to record the params vector.
    options['record_resids'] :  bool(False)
        Tells recorder whether to record the ressiduals vector.
    options['record_derivs'] :  bool(True)
        Tells recorder whether to record derivatives that are requested by a `Driver`.
    options['includes'] :  list of strings
        Patterns for variables to include in recording.
    options['excludes'] :  list of strings
        Patterns for variables to exclude in recording (processed after includes).
    """

    def __init__(self):
        # self.options = OptionsDictionary()
        # self.options.add_option('record_metadata', True)
        # self.options.add_option('record_unknowns', True)
        # self.options.add_option('record_params', False)
        # self.options.add_option('record_resids', False)
        # self.options.add_option('record_derivs', True,
        #                         desc='Set to True to record derivatives at the driver level')
        # self.options.add_option('includes', ['*'],
        #                         desc='Patterns for variables to include in recording')
        # self.options.add_option('excludes', [],
        #                         desc='Patterns for variables to exclude from recording '
        #                         '(processed after includes)')
        # self.out = None

        # # This is for drivers to determine if a recorder supports
        # # real parallel recording (recording on each process), because
        # # if it doesn't, the driver figures out what variables must
        # # be gathered to rank 0 if running under MPI.
        # #
        # # By default, this is False, but it should be set to True
        # # if the recorder will record data on each process to avoid
        # # unnecessary gathering.
        # self._parallel = False

        # self._filtered = {}
        # TODO: System specific includes/excludes

        self._owners = []


    def startup(self, group):
        """ Prepare for a new run.

        Args
        ----
        group : `Group`
            Group that owns this recorder.
        """

        # myparams = myunknowns = myresids = set()

        # check = self._check_path
        # incl = self.options['includes']
        # excl = self.options['excludes']

        # # Compute the inclusion lists for recording
        # if self.options['record_params']:
        #     myparams = [n for n in group.params if check(n, incl, excl)]
        # if self.options['record_unknowns']:
        #     myunknowns = [n for n in group.unknowns if check(n, incl, excl)]
        #     if self.options['record_resids']:
        #         myresids = myunknowns # unknowns and resids have same names
        # elif self.options['record_resids']:
        #     myresids = [n for n in group.resids if check(n, incl, excl)]

        # self._filtered[group.pathname] = {
        #     'p': myparams,
        #     'u': myunknowns,
        #     'r': myresids
        # }
        pass

    def _check_path(self, path, includes, excludes):
        """ Return True if `path` should be recorded. """

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

    def _get_pathname(self, iteration_coordinate):
        '''
        Converts an iteration coordinate to key to index
        `_filtered` to retrieve names of variables to be recorded.
        '''
        return '.'.join(iteration_coordinate[5::2])

    def _filter_vector(self, vecwrapper, key, iteration_coordinate):
        '''
        Returns a dict that is a subset of the given vecwrapper
        to be recorded.
        '''
        if not vecwrapper:
            return vecwrapper

        pathname = self._get_pathname(iteration_coordinate)
        return {n:vecwrapper[n] for n in self._filtered[pathname][key]}

    def record_metadata(self, group):
        """Writes the metadata of the given group

        Args
        ----
        group : `System`
            `System` containing vectors
        """
        raise NotImplementedError()

    def record_iteration(self, params, unknowns, resids, metadata):
        """
        Writes the provided data.

        Args
        ----
        params : dict
            Dictionary containing parameters. (p)

        unknowns : dict
            Dictionary containing outputs and states. (u)

        resids : dict
            Dictionary containing residuals. (r)

        metadata : dict, optional
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """
        raise NotImplementedError()

    def record_derivatives(self, derivs, metadata):
        """Writes the metadata of the given group

        Args
        ----
        derivs : dict
            Dictionary containing derivatives

        metadata : dict, optional
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """
        raise NotImplementedError()

    def close(self):
        """Closes `out` unless it's ``sys.stdout``, ``sys.stderr``, or StringIO.
        Note that a closed recorder will do nothing in :meth:`record`, and
        closing a closed recorder also does nothing.
        """
        # Closing a StringIO deletes its contents.
        if self.out not in (None, sys.stdout, sys.stderr):
            if not isinstance(self.out, StringIO):
                self.out.close()
            self.out = None
