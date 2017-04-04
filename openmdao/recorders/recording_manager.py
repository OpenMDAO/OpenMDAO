""" RecordingManager class definition. """

import sys
import os
import itertools
import time
import traceback

from six import iteritems

# from openmdao.core.mpi_wrap import MPI, debug

trace = os.environ.get('OPENMDAO_TRACE')

class RecordingManager(object):
    """ Object that routes function calls to all attached recorders. """

    def __init__(self):
        self._vars_to_record = {
            'pnames': set(),
            'unames': set(),
            'rnames': set(),
            }

        self._recorders = []
        self._has_serial_recorders = False
        self._casecomm = None  # comm used to gather parallel DOE cases

        # if MPI:
        #     self.rank = MPI.COMM_WORLD.rank
        # else:
        #     self.rank = 0
        self.rank = 0

    def append(self, recorder):
        """ Add a recorder for recording.

        Args
        ----
        recorder : `BaseRecorder`
           Recorder instance.
        """
        self._recorders.append(recorder)

    def __getitem__(self, index):
        return self._recorders[index]

    def __iter__(self):
        return iter(self._recorders)

    def _gather_vars(self, root, local_vars):
        """Gathers and returns only variables listed in
        `local_vars` from the `root` System.
        """

        if trace:
            debug("gathering vars for recording in %s" % root.pathname)
        all_vars = root.comm.gather(local_vars, root=0)
        if trace:
            debug("DONE gathering rec vars for %s" % root.pathname)

        if root.comm.rank == 0:
            dct = all_vars[-1]
            for d in all_vars[:-1]:
                dct.update(d)
            return dct

    def startup(self, root):
        """ Initialization during setup.

        Args
        ----
        root : `System`
           System containing variables.
        """
        pathname = root.pathname
        if MPI and root.is_active():
            rrank = root.comm.rank
            rowned = root._owning_ranks

        self._record_p = self._record_u = self._record_r = False

        for recorder in self._recorders:
            recorder.startup(root)

            if not recorder._parallel:
                self._has_serial_recorders = True

            pnames = recorder._filtered[pathname]['p']
            unames = recorder._filtered[pathname]['u']
            rnames = recorder._filtered[pathname]['r']

            if pnames:
                self._record_p = True
            if unames:
                self._record_u = True
            if rnames:
                self._record_r = True

            # now localize the lists to only
            # include local vars.  We need to do this after determining
            # if any mpi procs need to record each of params, unknowns,
            # and resids.  If none of them do, we can skip the mpi gather
            # for that group of vars.
            if MPI:
                pnames = [n for n in pnames if rrank==rowned[n]]
                unames = [n for n in unames if rrank==rowned[n]]
                rnames = [n for n in rnames if rrank==rowned[n]]

                # reduce the filter set for any parallel recorders to only
                # those variables that are owned by that rank
                if recorder._parallel:
                    recorder._filtered[pathname]['p'] = pnames
                    recorder._filtered[pathname]['u'] = unames
                    recorder._filtered[pathname]['r'] = rnames

            self._vars_to_record['pnames'].update(pnames)
            self._vars_to_record['unames'].update(unames)
            self._vars_to_record['rnames'].update(rnames)

    def close(self):
        """ Close all recorders. """
        for recorder in self._recorders:
            recorder.close()

    def record_metadata(self, root):
        """ Record metadata for all variables of interest.

        Args
        ----
        root : `System`
           System containing variables.
        """

        for recorder in self._recorders:
            # If the recorder does not support parallel recording
            # we need to make sure we only record on rank 0.
            if recorder._parallel or self.rank == 0:
                if recorder.options['record_metadata']:
                    recorder.record_metadata(root)

    def _get_local_case_data(self, root):
        """get names and values of all locally owned variables."""
        params = root.params
        unknowns = root.unknowns
        resids = root.resids
        params = [(p, params[p]) for p in self._vars_to_record['pnames']]
        unknowns = [(u, unknowns[u]) for u in self._vars_to_record['unames']]
        resids = [(r, resids[r]) for r in self._vars_to_record['rnames']]

        return params, unknowns, resids

    def record_completed_case(self, root, case):
        """Record the variables in the given case."""
        if not self._recorders:
            return

        case['meta']['timestamp'] = time.time()

        for recorder in self._recorders:
            recorder.record_iteration(case['p'], case['u'], case['r'], case['meta'])

    # def record_iteration(self, root, metadata, dummy=False):
    def record_iteration(self, object_requesting_recording, metadata):
        """ Gathers variables for non-parallel case recorders and calls
        record for all recorders.

        Args
        ----
        root : `System`
           System containing variables.

        metadata : dict
            Metadata for iteration coordinate

        dummy : bool, optional
            If True, this is a dummy iteration, so no data will be colllected
            from the model, but collective gather call will still be made.
        """
        if not self._recorders:
            return


        for recorder in self._recorders:
            recorder.record_iteration(object_requesting_recording, metadata)


        return

        # TODO_RECORDERS: get rid of the rest of this as needed

        if metadata is not None:
            metadata['timestamp'] = time.time()

        params = root.params
        unknowns = root.unknowns
        resids = root.resids

        cases = None

        if MPI:
            if dummy and self._casecomm is not None:
                case = (None, None, None, None)
                if trace: debug("DUMMY gathering cases")
                cases = self._casecomm.gather(case, root=0)
                if trace: debug("DUMMY done gathering cases:")
                return

            pnames = self._vars_to_record['pnames']
            unames = self._vars_to_record['unames']
            rnames = self._vars_to_record['rnames']

            # get names and values of all locally owned variables
            params = {p: params[p] for p in pnames}
            unknowns = {u: unknowns[u] for u in unames}
            resids = {r: resids[r] for r in rnames}

            if self._has_serial_recorders:
                params = self._gather_vars(root, params) if self._record_p else {}
                unknowns = self._gather_vars(root, unknowns) if self._record_u else {}
                resids = self._gather_vars(root, resids) if self._record_r else {}

                if self._casecomm is not None:
                    # our parent driver is running a parallel DOE, so we need to
                    # gather all of the cases to this rank and loop over them
                    case = (params, unknowns, resids, metadata)
                    if trace: debug("gathering cases")
                    cases = self._casecomm.gather(case, root=0)
                    if trace: debug("done gathering cases")
                    if cases is None:
                        cases = []

        if cases is None:
            cases = [(params, unknowns, resids, metadata)]

        # If the recorder does not support parallel recording
        # we need to make sure we only record on rank 0.
        for params, unknowns, resids, meta in cases:
            if params is None: # dummy cases have None in place of params, etc.
                continue
            for recorder in self._recorders:
                if recorder._parallel or MPI is None or self.rank == 0:
                    recorder.record_iteration(params, unknowns, resids, meta)

    def record_derivatives(self, derivs, metadata):
        """" Records derivatives if requested.

        Args
        ----
        derivs : dict
            Dictionary containing derivatives
        metadata : dict
            Metadata for iteration coordinate
        """

        metadata['timestamp'] = time.time()

        # If the recorder does not support parallel recording
        # we need to make sure we only record on rank 0.
        for recorder in self._recorders:
            if recorder.options['record_derivs']:
                if recorder._parallel or self.rank == 0:
                    recorder.record_derivatives(derivs, metadata)
