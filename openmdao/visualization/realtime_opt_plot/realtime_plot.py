import sqlite3

from openmdao.recorders.sqlite_reader import SqliteCaseReader
from openmdao.recorders.case import Case
from openmdao.visualization.realtime_opt_plot.realtime_analysis_driver_plot import (
    _RealTimeAnalysisDriverPlot,
)
from openmdao.visualization.realtime_opt_plot.realtime_optimizer_plot import (
    _RealTimeOptimizerPlot,
)

try:
    from bokeh.server.server import Server
    from bokeh.application.application import Application
    from bokeh.application.handlers import FunctionHandler

    bokeh_available = True
except ImportError:
    bokeh_available = False

# the time between calls to the udpate method
# if this is too small, the GUI interactions get delayed because
# code is busy trying to keep up with the periodic callbacks
_time_between_callbacks_in_ms = 1000

# Number of milliseconds for unused session lifetime
_unused_session_lifetime_milliseconds = 1000 * 60 * 10

try:
    from openmdao.utils.gui_testing_utils import get_free_port
except ImportError:
    # If get_free_port is unavailable, the default port will be used
    def get_free_port():
        return 5000


def _realtime_plot_setup_parser(parser):
    """
    Set up the realtime plot subparser for the 'openmdao opt_plot' command.

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
    parser.add_argument(
        "--no-display",
        action="store_false",
        dest="show",
        help="Do not launch browser showing plot. Used for CI testing",
    )


def _realtime_plot_cmd(options, user_args):
    """
    Run the opt_plot command.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    if bokeh_available:
        realtime_opt_plot(
            options.case_recorder_filename,
            _time_between_callbacks_in_ms,
            options.pid,
            options.show,
        )
    else:
        print(
            "The bokeh library is not installed so the real-time optimizaton "
            "lot is not available. "
        )
        return


class _CaseRecorderTracker:
    """
    A class that is used to get information from a case recorder.

    These methods are not provided by the SqliteCaseReader class.
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
            # TODO would be better to not have to open up the file each time
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


def realtime_opt_plot(
    case_recorder_filename, callback_period, pid_of_calling_script, show
):
    """
    Visualize the objectives, desvars, and constraints during an optimization process.

    Parameters
    ----------
    case_recorder_filename : MetaModelStructuredComp or MetaModelUnStructuredComp
        The metamodel component.
    callback_period : float
        The time period between when the application calls the update method.
    pid_of_calling_script : int
        The process id of the calling optimization script, if called this way.
    show : bool
        If true, launch the browser display of the plot.
    """

    def _make_realtime_opt_plot_doc(doc):
        case_tracker = _CaseRecorderTracker(case_recorder_filename)
        if case_tracker.is_driver_optimizer():
            _RealTimeOptimizerPlot(
                case_tracker,
                callback_period,
                doc=doc,
                pid_of_calling_script=pid_of_calling_script,
            )
        else:
            _RealTimeAnalysisDriverPlot(
                case_tracker,
                callback_period,
                doc=doc,
                pid_of_calling_script=pid_of_calling_script,
            )

    _port_number = get_free_port()

    try:
        server = Server(
            {"/": Application(FunctionHandler(_make_realtime_opt_plot_doc))},
            port=_port_number,
            unused_session_lifetime_milliseconds=_unused_session_lifetime_milliseconds,
        )
        server.start()
        if show:
            server.io_loop.add_callback(server.show, "/")

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
