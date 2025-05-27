"""A real-time plot monitoring the analysis driver process as an OpenMDAO script runs."""

import ctypes
import errno
import gc
import os
import sys
from collections import defaultdict
import sqlite3
from itertools import product
import time
from datetime import datetime, timedelta

try:
    from bokeh.models import (
        ColumnDataSource,
        Range1d,
        Button,
        Row,
        Div,
        BasicTicker,
        LinearColorMapper,
        ColorBar,
        Select,
        Toggle,
        Column,
        ScrollBox,
    )
    from bokeh.layouts import gridplot, Spacer
    from bokeh.transform import transform
    from bokeh.plotting import figure
    from bokeh.server.server import Server
    from bokeh.application.application import Application
    from bokeh.application.handlers import FunctionHandler
    from bokeh.palettes import Viridis256

    bokeh_available = True
except ImportError:
    bokeh_available = False

import numpy as np

from openmdao.recorders.sqlite_reader import SqliteCaseReader
from openmdao.recorders.case import Case

try:
    from openmdao.utils.gui_testing_utils import _get_free_port
except ImportError:
    # If _get_free_port is unavailable, the default port will be used
    def _get_free_port():
        return 5000


# Constants
# the time between calls to the udpate method
# if this is too small, the GUI interactions get delayed because
# code is busy trying to keep up with the periodic callbacks
_time_between_callbacks_in_ms = 1000
# _time_between_callbacks_in_ms = 100
# Number of milliseconds for unused session lifetime
_unused_session_lifetime_milliseconds = 1000 * 60 * 10
_toggle_styles = """
    font-size: 22px;
    box-shadow:
        0 4px 6px rgba(0, 0, 0, 0.1),    /* Distant shadow */
        0 1px 3px rgba(0, 0, 0, 0.08),   /* Close shadow */
        inset 0 2px 2px rgba(255, 255, 255, 0.2);  /* Top inner highlight */
"""

_max_number_initial_visible_sampled_variables = 4
_grid_plot_height_and_width = 240 

_max_label_length = 25
def _elide_variable_name_with_units(variable_name, units):
    elide_string = "..."
    if units:
        un_elided_string_length = len(f"{variable_name} ({units})")
    else:
        un_elided_string_length = len(variable_name)
    chop_length = max(un_elided_string_length - _max_label_length + len(elide_string),0)
    if chop_length:
        variable_name = elide_string + variable_name[chop_length:]

    if units:
        return f"{variable_name} ({units})"
    else:
        return variable_name

def _is_process_running(pid):
    if sys.platform == "win32":
        # PROCESS_QUERY_LIMITED_INFORMATION is available on Windows Vista and later.
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

        # Attempt to open the process.
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        else:
            # If OpenProcess fails, check if it's due to access being denied.
            ERROR_ACCESS_DENIED = 5
            if ctypes.windll.kernel32.GetLastError() == ERROR_ACCESS_DENIED:
                return True
            return False
    else:
        try:
            os.kill(pid, 0)
        except OSError as err:
            if err.errno == errno.ESRCH:  # No such process
                return False
            elif err.errno == errno.EPERM:  # Process exists, no permission to signal
                return True
            else:
                raise
        else:
            return True


def _realtime_opt_plot_setup_parser(parser):
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


def _realtime_opt_plot_cmd(options, user_args):
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


class _RealTimeOptPlot(object):
    """
    A class that handles all of the real-time plotting.

    Parameters
    ----------
    case_recorder_filename : str
        The path to the case recorder file.
    callback_period : double
        The time between Bokeh callback calls (in seconds).
    doc : bokeh.document.Document
        The Bokeh document which is a collection of plots, layouts, and widgets.
    pid_of_calling_script : int or None
        The process ID of the process that called the command to start the realtime plot.

        None if the plot was called for directly by the user.
    """

    def __init__(
        self, case_recorder_filename, callback_period, doc, pid_of_calling_script
    ):
        """
        Construct and initialize _RealTimeOptPlot instance.
        """
        self._case_recorder_filename = case_recorder_filename
        self._case_tracker = _CaseRecorderTracker(case_recorder_filename)
        self._pid_of_calling_script = pid_of_calling_script

        self._sampled_variables = None
        self._sampled_variables_visibility = {}
        self._prom_response = None

        self._num_samples_plotted = 0

        self._responses = None

        self._prom_responses = None

        self._scatter_plots = {}

        self._scatter_plots_figure = {}

        self._doc = doc

        self._source = None
        self._lines = []
        # flag to prevent updating label with units each time we get new data
        self._labels_updated_with_units = False
        self._source_stream_dict = None

        self._hist_source = {}

        self._plotted_response_variable = None

        # used to keep track of the y min and max of the data so that
        # the axes ranges can be adjusted as data comes in
        # self._y_min = defaultdict(lambda: float("inf"))
        # self._y_max = defaultdict(lambda: float("-inf"))

        # self._prom_response_min = float("inf")
        # self._prom_response_max = float("-inf")

        self._prom_response_min = defaultdict(lambda: float("inf"))
        self._prom_response_max = defaultdict(lambda: float("-inf"))

        self._start_time = time.time()

        self._sampled_variables_toggles = []

        self._hist_figures = {}

        def _update():
            print("_update")

            # print(f"{self.access_by_id('p1014')=}")

            # print (f"{self._sampled_variables_visibility=}")
            # this is the main method of the class. It gets called periodically by Bokeh
            # It looks for new data and if found, updates the plot with the new data
            new_case = self._case_tracker._get_new_case()

            if new_case is None:
                if self._pid_of_calling_script is None or not _is_process_running(
                    self._pid_of_calling_script
                ):
                    # no more new data in the case recorder file and the
                    #   optimization script stopped running, so no possible way to
                    #   get new data.
                    # But just keep sending the last data point.
                    # This is a hack to force the plot to re-draw.
                    # Otherwise if the user clicks on the variable buttons, the
                    #   lines will not change color because of the set_value hack done to get
                    #   get around the bug in setting the line color from JavaScript
                    self._source.stream(self._source_stream_dict)
                return

            # See if Bokeh source object is defined yet. If not, set it up
            # since now we have data from the case recorder with info about the
            # variables to be plotted.
            if self._source is None:
                self._setup_data_source()
                self._setup_figure()
                graph = Row(self.plot_figure, sizing_mode="stretch_both")
                doc.add_root(graph)
                # end of self._source is None - plotting is setup

            # Do the actual update of the plot including updating the plot range and adding the new
            # data to the Bokeh plot stream
            self._source_stream_dict = {}

            self._source_stream_dict[self._prom_response] = new_case.get_val(
                self._prom_response
            )[:1]

            for response in self._prom_responses:
                self._source_stream_dict[response] = new_case.get_val(
                    response
                )[:1]

            for sampled_variable in self._sampled_variables:
                self._source_stream_dict[sampled_variable] = new_case.get_val(
                    sampled_variable
                )[:1]

            self._source.stream(self._source_stream_dict)
            self._num_samples_plotted += 1

            for response in self._prom_responses:
                self._prom_response_min[response] = min(
                    self._prom_response_min[response], new_case.get_val(response)[:1][0]
                )
                self._prom_response_max[response] = max(
                    self._prom_response_max[response], new_case.get_val(response)[:1][0]
                )

            self._color_mapper.low = self._prom_response_min[self._prom_response]
            self._color_mapper.high = self._prom_response_max[self._prom_response]

            # Get current date and time
            now = datetime.now()
            formatted_time = now.strftime("%H:%M:%S on %B %d, %Y")

            # Format it as requested

            current_time = time.time()
            elapsed_total = current_time - self._start_time
            elapsed_formatted = str(timedelta(seconds=int(elapsed_total)))

            # styles={"font-size": "20px", "font-weight": "bold"},

            stats_text = f"""<div style="padding: 10px; ">
                            <p>Number of samples: {self._num_samples_plotted}</p>
                            <p>Last updated: {formatted_time}</p>
                            <p>Elapsed time: {elapsed_formatted}</p>
                            </div>"""

            self._text_box.text = stats_text

            self._update_histograms()

            # end of _update method

        doc.add_periodic_callback(_update, callback_period)
        doc.title = "OpenMDAO Analysis Driver Progress Plot"

    # Alternative: Access objects by ID
    def access_by_id(self, object_id):
        """Retrieve a Bokeh object by its ID"""
        try:
            for obj in self._doc.roots[0].references():
                if obj.id == object_id:
                    return obj
        except IndexError as e:
            return None
        return None

    def _update_histograms(self):
        for sampled_variable in self._sampled_variables:
            x_data = self._source.data[sampled_variable]

            # Compute histogram with 30 bins
            # TODO make 30 a variable
            hist, edges = np.histogram(
                x_data, bins=30, range=(np.min(x_data), np.max(x_data))
            )

            self._hist_source[sampled_variable].data.update(
                {
                    "top": hist,
                    "left": edges[:-1],  # left edge of each bin
                    "right": edges[1:],  # right edge of each bin
                }
            )

            self._hist_figures[sampled_variable].x_range = Range1d(
                np.min(x_data), np.max(x_data)
            )

    def _setup_data_source(self):

        self._source_dict = {}

        # return prom
        outputs = self._case_tracker.get_case_reader().list_source_vars("driver")[
            "outputs"
        ]

        # returns abs
        self._responses = list(
            self._case_tracker.get_case_reader().problem_metadata["responses"].keys()
        )

        # convert to prom
        self._prom_responses = []
        for response in self._responses:
            if response in self._case_tracker.get_case_reader()._abs2prom["output"]:
                self._prom_responses.append(
                    self._case_tracker.get_case_reader()._abs2prom["output"][response]
                )
            else:
                raise RuntimeError(f"No prom for abs variable {response}")

        # driver_case = self._case_tracker._get_case_by_counter(1)
        # objs = driver_case.get_objectives()
        # for now assume one response
        self._prom_response = self._prom_responses[0]
        self._source_dict[self._prom_response] = []

        for response in self._prom_responses:
            self._source_dict[response] = []

        # need to make unavailble for this any variables that are not scalars
        # self._case_tracker._get_shape
        self._sampled_variables = [
            varname for varname in outputs if varname not in self._prom_responses
        ]

        self._sampled_variables = []

        self._sampled_variables_non_scalar = []

        for varname in outputs:
            if varname not in self._prom_responses:
                shape = self._case_tracker._get_shape(varname)
                if shape == () or shape == (1,): # is scalar ??
                    self._sampled_variables.append(varname)
                else:
                    self._sampled_variables.append(varname)
                    self._sampled_variables_non_scalar.append(varname)

        self._sampled_variables.sort()
        self._sampled_variables_non_scalar.sort()

        # for sampled_variable in self._sampled_variables:
        #     self._sampled_variables_visibility[sampled_variable] = True

        for sampled_variable in self._sampled_variables:
            self._source_dict[sampled_variable] = []

        self._source = ColumnDataSource(self._source_dict)

    def _make_variable_button(self, varname, active, is_scalar, callback):
        toggle = Toggle(
            label=varname,
            active=active,
            margin=(0, 0, 8, 0),
        )
        # Add custom CSS styles for both active and inactive states
        if is_scalar:
            toggle.stylesheets = [
                f"""
                    .bk-btn {{
                        {_toggle_styles}
                    }}
                    .bk-btn.bk-active {{
                        background-color: rgb(from #000000 R G B / 0.3);
                        {_toggle_styles}
                    }}
                """
            ]
        else:
            toggle.stylesheets = [
                f"""
                    .bk-btn {{
                        cursor:help;
                        pointer-events: none !important;
                        opacity: 0.5 !important;
                        {_toggle_styles}
                    }}
                    .bk-btn.bk-active {{
                        pointer-events: none !important;
                        background-color: rgb(from #000000 R G B / 0.3);
                        {_toggle_styles}
                    }}
                """
            ]


        if is_scalar:
            toggle.on_change("active", callback)
        else:
            # Create a div for instructions/tooltip
            tooltip_div = Div(
                # text="<i>Non-scalar var.</i>",
                text=f"<div style='text-align:center;font-size:12px;cursor:help;' title='Plotting of non-scalars is not currently supported'>Non-scalar var</div>",
                styles={'font-size': '12px', 'color': 'gray', 'margin-top': '5px', 'cursor':'help'},
            )
            toggle = Row(toggle, tooltip_div)

        self._sampled_variables_toggles.append(toggle)
        return toggle

    def _setup_figure(self):

        N = len(self._sampled_variables)

        def _sampled_variable_callback(var_name):
            def toggle_callback(attr, old, new):
                # The callback "closes over" the var_name variable
                # lines[var_name].visible = toggles[var_name].active
                self._sampled_variables_visibility[var_name] = new
                self._hist_figures[var_name].visible = new

                for i, (y, x) in enumerate(
                    product(self._sampled_variables, self._sampled_variables)
                ):

                    icolumn = i % N
                    irow = i // N

                    # only do the lower half
                    if x != y:
                        self._scatter_plots_figure[(x,y)].visible = (icolumn < irow \
                            and self._sampled_variables_visibility[x] and self._sampled_variables_visibility[y] )

            return toggle_callback

        number_initial_visible_sampled_variables = 0
        for sampled_var in self._sampled_variables:
            if sampled_var in self._sampled_variables_non_scalar:
                self._sampled_variables_visibility[sampled_var] = False
                self._make_variable_button(sampled_var,False, False, _sampled_variable_callback(sampled_var))            
            else:  # is scalar
                if number_initial_visible_sampled_variables < _max_number_initial_visible_sampled_variables:
                    self._sampled_variables_visibility[sampled_var] = True
                    self._make_variable_button(sampled_var,True, True, _sampled_variable_callback(sampled_var))
                    number_initial_visible_sampled_variables += 1
                else:
                    self._sampled_variables_visibility[sampled_var] = False
                    self._make_variable_button(sampled_var,False, True, _sampled_variable_callback(sampled_var))

        # xdrs = [DataRange1d(bounds=None) for _ in range(N)]
        # ydrs = [DataRange1d(bounds=None) for _ in range(N)]

        # print(f"{xdrs=}")
        # print(f"{ydrs=}")

        # Create a color mapper using Viridis (colorblind-friendly)
        self._color_mapper = LinearColorMapper(palette=Viridis256, low=0, high=200)

        plots = []

        for i, (y, x) in enumerate(  #  Why y,x here ??
            product(self._sampled_variables, self._sampled_variables)
        ):

            icolumn = i % N
            irow = i // N

            if x != y:

                x_units = self._case_tracker._get_units(x)
                y_units = self._case_tracker._get_units(y)

                p = figure(
                    # x_range=xdrs[i % N],
                    # y_range=ydrs[i // N],
                    background_fill_color="#fafafa",
                    border_fill_color="white",
                    width=_grid_plot_height_and_width,
                    height=_grid_plot_height_and_width,
                    output_backend="webgl",
                )

                self._scatter_plots_figure[(x,y)] = p

                # p.xaxis.axis_label = f"{x} ({x_units} {i})"
                # p.yaxis.axis_label = f"{y} ({y_units})"
                p.axis.visible = True

                self._scatter_plots[(x,y)] = p.scatter(
                    x=x,
                    source=self._source,
                    y=y,
                    size=5,
                    line_color=None,
                    fill_color=transform(
                        self._prom_response, self._color_mapper
                    ),  # This maps f to colors
                )
            else:  # on the diagonal
                # Extract the x column data for the histogram
                x_data = self._source.data[x]

                # Compute histogram with 30 bins

                # DO I EVEN NEED THIS HERE? OR JUST in update?
                if x_data:
                    hist, edges = np.histogram(
                        x_data, bins=30, range=(np.min(x_data), np.max(x_data))
                    )
                else:
                    hist, edges = np.histogram(x_data, bins=30)

                # Create a new ColumnDataSource for the histogram data
                # hist_source = ColumnDataSource(data={
                self._hist_source[x] = ColumnDataSource(
                    data={
                        "top": hist,  # height of each bin
                        "bottom": np.zeros(len(hist)),  # bottom of each bin starts at 0
                        "left": edges[:-1],  # left edge of each bin
                        "right": edges[1:],  # right edge of each bin
                    }
                )

                units = self._case_tracker._get_units(x)

                # Create the figure
                p = figure(
                    # title=f"Histogram of {x} Values",
                    # For reasons TBD, the width and height in the Plot constructor
                    # and the width and height here mean different things.
                    # so need to add 40. TODO do this better
                    width=_grid_plot_height_and_width,
                    height=_grid_plot_height_and_width,
                    # tools="pan,wheel_zoom,box_zoom,reset,save",
                    # x_axis_label=f"{x} ({units})",
                    # y_axis_label="Frequency",
                    output_backend="webgl",
               )

                # Add the histogram bars using quad glyphs
                p.quad(
                    source=self._hist_source[x],
                    top="top",
                    bottom="bottom",
                    left="left",
                    right="right",
                    fill_color="#3288bd",
                    line_color="white",
                    alpha=0.7,
                )

                self._hist_figures[x] = p

            if icolumn > irow: # only lower half and diagonal
                p.visible = False
            else:
                if self._sampled_variables_visibility[x] and self._sampled_variables_visibility[y]:
                    p.visible = True
                else:
                    p.visible = False

            plots.append(p)

        # insert var name labels along edges

#           writing-mode: vertical-lr;
#   text-orientation: upright;

            # writing-mode: vertical-rl;
            # text-orientation: mixed;

#               writing-mode: vertical-lr;
#   transform: rotate(180deg);


        # row labels
        for i, sampled_variable in enumerate(reversed(self._sampled_variables)):
            irow = N - i - 1
            idx = irow * N
            elided_variable_name_with_units = _elide_variable_name_with_units(sampled_variable, None)
            p = Div(
                # text=f"{sampled_variable}",
                text=f"<div style='text-align:center;font-size:12px;writing-mode:vertical-lr;transform:rotate(180deg); cursor:help;' title='{sampled_variable}'>{elided_variable_name_with_units}</div>",
                # styles={"font-size": "12px",
                #         "text-align":"center",
                #         "writing-mode":"vertical-lr",
                #         "text-orientation":"upright",
                #         },
                align="center",
            )
            plots.insert(idx,p)

        # column labels
        # need to push one blank one for the bottom left corner
        p = Div(
            text=f"",
        )
        plots.append(p)
        for sampled_variable in self._sampled_variables:
            units = self._case_tracker._get_units(x)
            elided_variable_name_with_units = _elide_variable_name_with_units(sampled_variable, units)
            p = Div(
                # text=f"{sampled_variable} ({units})",
                text=f"<div style='text-align:center;font-size:14px; cursor:help;' title='{sampled_variable}'>{elided_variable_name_with_units}</div>",
                # text=f"<center>{sampled_variable} ({units})</center>",
                styles={"font-size": "12px", "text-align":"center"},
                align="center",
            )
            plots.append(p)

        gp = gridplot(
            plots,
            # ncols=N,
            ncols=N+1,
            toolbar_location=None,
        )

        toggle_column = Column(
            children=self._sampled_variables_toggles,
            sizing_mode="stretch_both",
            height_policy="fit",
            styles={
                "overflow-y": "auto",
                "border": "1px solid #ddd",
                "padding": "8px",
                # "background-color": "#f0f0f0",
                'max-height': '100vh'  # Ensures it doesn't exceed viewport
            },
        )

        # header for the variable list
        sampled_variables_label = Div(
            text="Sampled Variables",
            width=200,
            styles={"font-size": "20px", "font-weight": "bold"},
        )
        label_and_toggle_column = Column(
            sampled_variables_label,
            toggle_column,
            sizing_mode="stretch_height",
            height_policy="fit",
            styles={
                'max-height': '100vh',  # Ensures it doesn't exceed viewport
                'border-radius': '5px',
                'border': '5px solid black',
            },
        )

        scroll_box = ScrollBox(
            child=label_and_toggle_column,
            sizing_mode="stretch_height",
            height_policy="max",
        )

        # TODO avoid the [1]
        p = figure(height=2 * _grid_plot_height_and_width, width=0, toolbar_location=None)

        # Define two points (x0, y0) and (x1, y1)
        x = [0, 1]
        y = [0, 1]

        # Plot a line between the two points
        line = p.line(
            x,
            y,
        )

        p.grid.grid_line_color = None

        # Add the color bar to this figure
        color_bar = ColorBar(
            color_mapper=self._color_mapper,
            title=f"Response variable: '{self._prom_response}'",
            border_line_color=None,
            width=20,
            label_standoff=14,
            title_text_font_size="20px",
            ticker=BasicTicker(),  # This is the key part you're missing
            location=(0, 0),
        )

        p.add_layout(color_bar, "right")

        line.visible = False
        # Hide axes
        p.xaxis.visible = False
        p.yaxis.visible = False

        # Hide grid lines
        p.xgrid.visible = False
        p.ygrid.visible = False

        # Hide the title
        p.title.visible = False

        p.outline_line_alpha = 0

        analysis_progress_label = Div(
            text="Analysis Driver Progress",
            width=200,
            styles={"font-size": "20px", "font-weight": "bold"},
        )

        # show number of samples plotted
        self._text_box = Div(
            text="""<div style="padding: 10px; border-radius: 5px;">
                    <p>Waiting for data...</p>
                    </div>""",
            width=600,
            height=100,
        )

        analysis_progress_box = Column(
            analysis_progress_label,
            self._text_box,
            styles={
                # "max-height": "100vh",  # Ensures it doesn't exceed viewport
                "border-radius": "5px",
                "border": "5px solid black",
            },
        )

        p.min_border_right = 100

        spacer2 = Spacer(width=150, height=50)  # Between plot 2 (with colorbar) and text box
        spacer3 = Spacer(width=100, height=20)
        spacer4 = Spacer(width=100, height=20)
        spacer5 = Spacer(width=100, height=20)

        script_name = self._case_recorder_filename

        title_div = Div(
            text=f"Analysis Driver Progress for {script_name}",
            styles={
                "font-size": "20px", 
                "font-weight": "bold",
                "border-radius": "5px",
                "border": "5px solid black",
                "padding": "8px",
                },
        )

        quit_button = Button(label="Quit Application", button_type="danger")

        # Define callback function for the quit button
        def quit_app():
            raise KeyboardInterrupt("Quit button pressed")

        # Attach the callback to the button
        quit_button.on_click(quit_app)

        # Create a Select widget with some options
        menu = Select(
            # title="Choose a response variable:",
            options=self._prom_responses,
            value=self._prom_responses[0],  # Default value
        #     styles={
        #         "border": "5px solid black",
        #         "padding": "8px",
        #         "background-color": "#f0f0f0",
        # },
    )

        # header for the variable list
        response_variable_label = Div(
            text="Response variable",
            width=200,
            styles={"font-size": "20px", "font-weight": "bold"},
        )

        response_varible_box = Column(response_variable_label, menu,
                                                  styles={
                "border": "5px solid black",
                "padding": "8px",
                # "background-color": "#f0f0f0",
        },
        )

        # Python callback function that will run when selection changes
        def cb_select_response_variable(attr, old, new):
            # Print the selected value to the console
            print(f"Selected value: {new}")

            self._plotted_response_variable = new

            self._prom_response = new 

            color_bar.title = f"Response variable: '{new}'"

            self._color_mapper.low = self._prom_response_min[self._prom_response]
            self._color_mapper.high = self._prom_response_max[self._prom_response]

            for (x,y), scatter_plot in self._scatter_plots.items():
                t = transform(
                        self._prom_response, self._color_mapper
                    )
                # t is <class 'bokeh.core.property.vectorization.Field'>
                scatter_plot.glyph.fill_color = transform(
                        self._prom_response, self._color_mapper
                    )

                # line.glyph.y = {'field': selected_column}
                # scatter_plot.glyph.fill_color is a Field

        # Attach the callback to the Select widget
        menu.on_change("value", cb_select_response_variable)

        # sampled_variables_non_scalar_div = Div(
        #     text=f"The following non-scalar variables cannot be plotted {self._sampled_variables_non_scalar}",
        #     styles={
        #         "font-size": "12px", 
        #         "border-radius": "5px",
        #         "border": "5px solid black",
        #         "padding": "8px",
        #         },
        # )

        final_layout = Row(
            Column(title_div, gp),
            p,
            spacer2,
            Column(spacer2, quit_button, spacer3, analysis_progress_box, spacer4, response_varible_box, spacer5, 
                #    sampled_variables_non_scalar_div, spacer3,
                   scroll_box, sizing_mode="stretch_both"),
            # sizing_mode="fixed",
            sizing_mode="stretch_both",
        )

        self.plot_figure = final_layout


def print_bokeh_objects(doc):
    # Get all objects in the document
    all_objects = doc.roots[0].references()
    
    # Print information about each object
    print("\n=== BOKEH OBJECTS IN DOCUMENT ===")
    for obj in all_objects:
        obj_id = obj.id
        obj_type = type(obj).__name__
        
        # Get additional information based on object type
        additional_info = {}
        if hasattr(obj, 'name') and obj.name:
            additional_info['name'] = obj.name
        if hasattr(obj, 'tags') and obj.tags:
            additional_info['tags'] = obj.tags
        if hasattr(obj, 'label') and obj.label:
            additional_info['label'] = obj.label
        if hasattr(obj, 'title') and obj.title:
            if hasattr(obj.title, 'text'):
                additional_info['title'] = obj.title.text
            else:
                additional_info['title'] = str(obj.title)
        
        # Print the information
        print(f"ID: {obj_id} | Type: {obj_type}", end="")
        if additional_info:
            print(f" | Info: {additional_info}")
        else:
            print()


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


        # Print to console when the document is loaded
        def on_document_ready(event):
            print_bokeh_objects(doc)

        # doc.on_event('document_ready', on_document_ready)



        _RealTimeOptPlot(
            case_recorder_filename,
            callback_period,
            doc=doc,
            pid_of_calling_script=pid_of_calling_script,
        )



    _port_number = _get_free_port()

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
