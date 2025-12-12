"""
Classes and functions to support the realtime plotting.
"""

import os
import sqlite3
import subprocess

from openmdao.recorders.sqlite_reader import SqliteCaseReader
from openmdao.recorders.case import Case
from openmdao.utils import hooks
from openmdao.utils.file_utils import _load_and_exec, is_python_file
from openmdao.utils.gui_testing_utils import get_free_port
from openmdao.utils.record_util import check_valid_sqlite3_db
from openmdao.visualization.realtime_plot.realtime_analysis_driver_plot \
    import _RealTimeAnalysisDriverPlot
from openmdao.visualization.realtime_plot.realtime_optimizer_plot import _RealTimeOptimizerPlot

try:
    from bokeh.server.server import Server
    from bokeh.application.application import Application
    from bokeh.application.handlers import FunctionHandler
    from tornado.ioloop import PeriodicCallback
    from tornado.web import StaticFileHandler

    bokeh_and_dependencies_available = True
except ImportError:
    bokeh_and_dependencies_available = False


# the time between calls to the udpate method
# if this is too small, the GUI interactions get delayed because
# code is busy trying to keep up with the periodic callbacks
_time_between_callbacks_in_ms = 300

# Number of milliseconds for unused session lifetime
_unused_session_lifetime_milliseconds = 1000 * 60 * 10


def _realtime_plot_setup_parser(parser):
    """
    Set up the realtime plot subparser for the 'openmdao realtime_plot' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument(
        "case_recorder_filename",
        type=str,
        help="Name of openmdao case recorder filename. It should contain driver cases",
    )

    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Process ID of calling optimization script, "
        "defaults to None if called by the user directly",
    )

    parser.add_argument('--script', type=str, default=None,
                        help='The name of the script that created the case recorder file.')


def _realtime_plot_cmd(options, user_args):
    """
    Run the realtime_plot command.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    if bokeh_and_dependencies_available:
        realtime_plot(
            options.case_recorder_filename,
            _time_between_callbacks_in_ms,
            options.pid,
            options.script,
        )
    else:
        print(
            "The bokeh library and dependencies are not installed so the realtime "
            "plot is not available. "
        )
        return


def _rtplot_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao rtplot' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')


def _rtplot_cmd(options, user_args):
    """
    Return the post_setup hook function for 'openmdao rtplot'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    if not bokeh_and_dependencies_available:
        print(
            "The bokeh library and dependencies are not installed so the rtplot "
            "command is not available. "
        )
        return

    file_path = options.file[0]
    if is_python_file(file_path):
        script_path = file_path
    else:
        script_path = None

    def _view_realtime_plot_hook(problem):
        driver = problem.driver
        if not driver:
            raise RuntimeError(
                "Unable to run realtime optimization progress plot because no Driver")
        if len(problem.driver._rec_mgr._recorders) == 0:
            raise RuntimeError(
                "Unable to run realtime optimization progress plot "
                    "because no case recorder attached to Driver"
            )

        case_recorder_file = str(problem.driver._rec_mgr._recorders[0]._filepath)

        cmd = ['openmdao', 'realtime_plot', '--pid', str(os.getpid()), case_recorder_file]
        if script_path:
            cmd.insert(-1, '--script')
            cmd.insert(-1, script_path)
        cp = subprocess.Popen(cmd)  # nosec: trusted input

        # Do a quick non-blocking check to see if it immediately failed
        # This will catch immediate failures but won't wait for the process to finish
        quick_check = cp.poll()
        if quick_check is not None and quick_check != 0:
            # Process already terminated with an error
            stderr = cp.stderr.read().decode()
            raise RuntimeError(
                f"Failed to start up the realtime plot server with code {quick_check}: {stderr}.")

    def _view_realtime_plot(case_recorder_file):
        cmd = [
            "openmdao",
            "realtime_plot",
            "--pid",
            str(os.getpid()),
            case_recorder_file,
        ]

        cp = subprocess.Popen(cmd)  # nosec: trusted input

        # Do a quick non-blocking check to see if it immediately failed
        # This will catch immediate failures but won't wait for the process to finish
        quick_check = cp.poll()
        if quick_check is not None and quick_check != 0:
            # Process already terminated with an error
            stderr = cp.stderr.read().decode()
            raise RuntimeError(
                f"Failed to start up the realtime plot server with code {quick_check}: {stderr}."
            )

    # check to see if options.file is python script, sqlite file or neither
    file_path = options.file[0]
    try:
        check_valid_sqlite3_db(file_path)
        _view_realtime_plot(file_path)
        return
    except IOError:
        pass
    if is_python_file(file_path):
        # register the hook
        hooks._register_hook(
            "_setup_recording", "Problem", post=_view_realtime_plot_hook, ncalls=1
        )
        # run the script
        _load_and_exec(file_path, user_args)
    else:
        raise RuntimeError(
            "The argument to the openmdao rtplot command must be either a "
            "case recorder file or an OpenMDAO python script."
        )


class _CaseRecorderTracker:
    """
    A class that is used to get information from a case recorder.

    This class was created to handle the realtime reading of a case recorder file.
    These features are not provided by the SqliteCaseReader class.
    """

    def __init__(self, case_recorder_filename):
        self._case_recorder_filename = case_recorder_filename
        self._cr = None
        self._initial_case = (
            None  # need the initial case to get info about the variables
        )
        self._next_id_to_read = 1

    def get_case_reader(self):
        return self._cr

    def _open_case_recorder(self):
        if self._cr is None:
            self._cr = SqliteCaseReader(self._case_recorder_filename)

    def get_case_recorder_filename(self):
        return self._case_recorder_filename

    def is_driver_optimizer(self):
        self._open_case_recorder()
        return self._cr.problem_metadata["driver"]["supports"]["optimization"]["val"]

    def _get_case_by_counter(self, counter):
        # use SQL to see if a case with this counter exists
        with sqlite3.connect(self._case_recorder_filename) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute(
                "SELECT * FROM driver_iterations WHERE " "counter=:counter",
                {"counter": counter},
            )
            row = cur.fetchone()
        con.close()

        # use SqliteCaseReader code to get the data from this case
        if row:
            self._open_case_recorder()
            var_info = self._cr.problem_metadata["variables"]
            case = Case(
                "driver",
                row,
                self._cr._prom2abs,
                self._cr._abs2prom,
                self._cr._abs2meta,
                self._cr._conns,
                var_info,
                self._cr._format_version,
            )

            return case
        else:
            return None

    def _get_data_from_case(self, driver_case):
        objs = driver_case.get_objectives(scaled=False)
        design_vars = driver_case.get_design_vars(scaled=False)
        constraints = driver_case.get_constraints(scaled=False)

        new_data = {
            "counter": int(driver_case.counter),
        }

        # get objectives
        objectives = {}
        for name, value in objs.items():
            objectives[name] = value
        new_data["objs"] = objectives

        # get des vars
        desvars = {}
        for name, value in design_vars.items():
            desvars[name] = value
        new_data["desvars"] = desvars

        # get cons
        cons = {}
        for name, value in constraints.items():
            cons[name] = value
        new_data["cons"] = cons

        return new_data

    def _get_new_case(self):
        # get the next unread case from the recorder
        driver_case = self._get_case_by_counter(self._next_id_to_read)
        if driver_case is None:
            return None

        if self._initial_case is None:
            self._initial_case = driver_case

        self._next_id_to_read += 1

        return driver_case

    def _get_obj_names(self):
        obj_vars = self._initial_case.get_objectives()
        return obj_vars.keys()

    def _get_desvar_names(self):
        design_vars = self._initial_case.get_design_vars()
        return design_vars.keys()

    def _get_cons_names(self):
        cons = self._initial_case.get_constraints()
        return cons.keys()

    def _get_constraint_bounds(self, name):
        cons = self._initial_case.get_constraints()
        var_info = cons._var_info[name]
        return (var_info['lower'], var_info['upper'])

    def _get_desvar_bounds(self, name):
        lower = self._cr.problem_metadata['variables'][name]['lower']
        upper = self._cr.problem_metadata['variables'][name]['upper']
        return lower, upper

    def _get_units(self, name):
        try:
            units = self._initial_case._get_units(name)
        except RuntimeError as err:
            if str(err).startswith("Can't get units for the promoted name"):
                return "Ambiguous"
            raise
        except KeyError:
            return "Unavailable"

        if units is None:
            units = "Unitless"
        return units

    def _get_shape(self, name):
        item = self._initial_case[name]
        return item.shape

    def _get_size(self, name):
        item = self._initial_case[name]
        return item.size


def realtime_plot(case_recorder_filename, callback_period,
                  pid_of_calling_script, script):
    """
    Visualize the objectives, desvars, and constraints during an optimization or analysis process.

    Parameters
    ----------
    case_recorder_filename : str
        The path to the case recorder file that is the source of the data for the plot.
    callback_period : float
        The time period between when the application calls the update method.
    pid_of_calling_script : int
        The process id of the calling optimization script, if called this way.
    script : str or None
        If not None, the file path of the script that created the case recorder file.
    """
    def _make_realtime_plot_doc(doc):
        case_tracker = _CaseRecorderTracker(case_recorder_filename)
        if case_tracker.is_driver_optimizer():
            _RealTimeOptimizerPlot(
                case_tracker,
                callback_period,
                doc=doc,
                pid_of_calling_script=pid_of_calling_script,
                script=script,
            )
        else:
            _RealTimeAnalysisDriverPlot(
                case_tracker,
                callback_period,
                doc=doc,
                pid_of_calling_script=pid_of_calling_script,
                script=script,
            )

    _port_number = get_free_port()

    try:
        server = Server(
            {"/": Application(FunctionHandler(_make_realtime_plot_doc))},
            port=_port_number,
            unused_session_lifetime_milliseconds=_unused_session_lifetime_milliseconds,
            extra_patterns=[
                (
                    "/images/(.*)",
                    StaticFileHandler,
                    {"path": os.path.normpath(os.path.dirname(__file__) + "/images/")},
                ),
            ],
        )
        server.start()

        testflo_running = os.environ.pop('TESTFLO_RUNNING', None)

        if not testflo_running:
            server.io_loop.add_callback(server.show, "/")
        else:
            # for testing, we are, for now, just testing that the command runs.
            # So can stop the plot process right away
            def update_data():
                raise KeyboardInterrupt("end plotting process when in testing mode")

            periodic_callback = PeriodicCallback(update_data, 1000)  # 1 second
            periodic_callback.start()

        print(
            f"Real-time optimization plot server running on http://localhost:{_port_number}"
        )
        server.io_loop.start()
    except KeyboardInterrupt as e:
        print(
            f"Real-time optimization plot server stopped due to keyboard interrupt: {e}"
        )
    except Exception as e:
        print(f"Error starting real-time optimization plot server: {e}")
    finally:
        print("Stopping real-time optimization plot server")
        if "server" in globals():
            server.stop()
