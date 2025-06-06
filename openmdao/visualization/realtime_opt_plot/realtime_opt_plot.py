"""A real-time plot monitoring the analysis driver process as an OpenMDAO script runs."""

import ctypes
import errno
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
    from openmdao.utils.gui_testing_utils import get_free_port
except ImportError:
    # If get_free_port is unavailable, the default port will be used
    def get_free_port():
        return 5000


# Constants

# the time between calls to the udpate method
# if this is too small, the GUI interactions get delayed because
# code is busy trying to keep up with the periodic callbacks
_time_between_callbacks_in_ms = 1000

# Number of milliseconds for unused session lifetime
_unused_session_lifetime_milliseconds = 1000 * 60 * 10

# styling for the sampled variables buttons
_sampled_variable_button_styles = """
    font-size: 22px;
    box-shadow:
        0 4px 6px rgba(0, 0, 0, 0.1),    /* Distant shadow */
        0 1px 3px rgba(0, 0, 0, 0.08),   /* Close shadow */
        inset 0 2px 2px rgba(255, 255, 255, 0.2);  /* Top inner highlight */
"""
# some models have a large number of variables. Putting them all in a plot
#   is not practical. Initially show no more than this number
_max_number_initial_visible_sampled_variables = 3
# how big the individual plots should be in the grid
_grid_plot_height_and_width = 240
# number of histogram bins in the histogram plots for the sampled variables
_num_histogram_bins = 30 

# variable names can be very long and too large to show completely
#  in the plot axes. This is the largest number of characters that
#  can be shown in the axes. The rest are elided. Full variable
#  shown as a tooltip
_max_label_length = 25
_elide_string = "..."

# palette for the color bar
_color_palette = Viridis256

# the color bar showing response needs an initial value before new data comes in
_initial_response_range_for_plots = (0,200)

# function to create the labels for the plots, need to elide long variable names
# include the units, if given
def _elide_variable_name_with_units(variable_name, units):
    if units:
        un_elided_string_length = len(f"{variable_name} ({units})")
    else:
        un_elided_string_length = len(variable_name)
    chop_length = max(un_elided_string_length - _max_label_length + len(_elide_string),0)
    if chop_length:
        variable_name = _elide_string + variable_name[chop_length:]
    if units:
        return f"{variable_name} ({units})"
    else:
        return variable_name

# Alternative: Access objects by ID
def access_by_id(doc, object_id):
    """Retrieve a Bokeh object by its ID"""
    try:
        for obj in doc.roots[0].references():
            if obj.id == object_id:
                return obj
    except IndexError as e:
        return None
    return None

def _make_sampled_variable_button(varname, active, is_scalar, callback):
    sampled_variable_button = Toggle(
        label=varname,
        active=active,
    )
    # Add custom CSS styles for both active and inactive states
    if is_scalar:
        sampled_variable_button.stylesheets = [
            f"""
                .bk-btn {{
                    {_sampled_variable_button_styles}
                }}
                .bk-btn.bk-active {{
                    background-color: rgb(from #000000 R G B / 0.3);
                    {_sampled_variable_button_styles}
                }}
            """
        ]
    else:
        sampled_variable_button.stylesheets = [
            f"""
                .bk-btn {{
                    cursor:help;
                    pointer-events: none !important;
                    opacity: 0.5 !important;
                    {_sampled_variable_button_styles}
                }}
                .bk-btn.bk-active {{
                    pointer-events: none !important;
                    background-color: rgb(from #000000 R G B / 0.3);
                    {_sampled_variable_button_styles}
                }}
            """
        ]


    if is_scalar:
        sampled_variable_button.on_change("active", callback)
    else:
        # Create a div for instructions/tooltip
        tooltip_div = Div(
            # text="<i>Non-scalar var.</i>",
            text=f"<div style='text-align:center;font-size:12px;cursor:help;' title='Plotting of non-scalars is not currently supported'>Non-scalar var</div>",
            styles={'font-size': '12px', 'color': 'gray', 'margin-top': '5px', 'cursor':'help'},
        )
        sampled_variable_button = Row(sampled_variable_button, tooltip_div)

    return sampled_variable_button


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


def print_ids():
    import gc
    from bokeh.models import widgets

    # Get all widget instances in memory
    all_widgets = []
    for obj in gc.get_objects():
        if isinstance(obj, widgets.Widget):
            all_widgets.append(obj)

    for widget in all_widgets:
        print(f"Widget type: {type(widget).__name__}, ID: {widget.id}")
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
        self._doc = doc

        # lists of the sampled variables in the analysis and the responses
        self._sampled_variables = None
        self._num_sampled_variables = None
        self._prom_responses = None

        # A dict indicating which sampled variables are visible in the grid
        #   based on what the user selected
        self._sampled_variables_visibility = {}

        # The current response being plotted
        self._prom_response = None

        # just used for the access_by_id function TODO remove when debugging done
        # self._doc = doc

        # data source items for doing streaming in bokeh
        self._source = None
        self._source_stream_dict = {}
        self._hist_source = {}

        # used to keep track of the y min and max of the data so that
        # the axes ranges can be adjusted as data comes in
        self._prom_response_min = defaultdict(lambda: float("inf"))
        self._prom_response_max = defaultdict(lambda: float("-inf"))

        # This list of widgets used to let user turn on and off variable plots
        self._sampled_variables_buttons = []

        # the actually scatter plots. Need access to them to change the
        #  fill color for the dots when the response variable range changes
        #  due to new data.
        self._scatter_plots = []

        # dictionaries for the items in the grid.
        #  used to turn their visibility on and off
        self._scatter_plots_figure = {}
        self._hist_figures = {}
        self._row_labels = {}
        self._column_labels = {}

        # data used to populate the Analysis Driver Progress block in the plot
        self._start_time = time.time()
        self._num_samples_plotted = 0

        # this is the main method of the class. It gets called periodically by Bokeh
        # It looks for new data and if found, updates the plot with the new data
        def _update():
            new_case = self._case_tracker._get_new_case()


            if new_case is None:
                if self._pid_of_calling_script is None or not _is_process_running(
                    self._pid_of_calling_script
                ):
                    # no more new data in the case recorder file and the
                    #   analysis script stopped running, so no possible way to
                    #   get new data.
                    # But just keep sending the last data point.
                    # This is a hack to force the plot to re-draw.
                    # Otherwise if the user clicks on the variable buttons, the
                    #   lines will not change color because of the set_value hack done to get
                    #   get around the bug in setting the line color from JavaScript.
                    self._source.stream(self._source_stream_dict)
                return


            # See if Bokeh source object is defined yet. If not, set it up
            # since now we have data from the case recorder with info about the
            # variables to be plotted.
            # Also setup the overall page
            if self._source is None:
                self._setup_data_source()
                self._setup_figure()
                doc.add_root(self._overall_layout)
                # end of self._source is None - plotting is setup


            # TODO - is the case of new_case None handled correctly?
            self._update_source_stream(new_case)

            self._update_scatter_plots(new_case)

            self._update_histograms()

            self._update_analysis_driver_progress_text_box()
            # end of _update method

        doc.add_periodic_callback(_update, callback_period)
        doc.title = "OpenMDAO Analysis Driver Progress Plot"

    def _update_analysis_driver_progress_text_box(self):
        self._num_samples_plotted += 1
        last_updated_time_formatted = datetime.now().strftime("%H:%M:%S on %B %d, %Y")
        elapsed_total_time = time.time() - self._start_time
        elapsed_total_time_formatted = str(timedelta(seconds=int(elapsed_total_time)))
        self._analysis_driver_progress_text_box.text = f"""<div style="padding: 10px; ">
                        <p>Number of samples: {self._num_samples_plotted}</p>
                        <p>Last updated: {last_updated_time_formatted}</p>
                        <p>Elapsed time: {elapsed_total_time_formatted}</p>
                        </div>"""

    def _update_source_stream(self, new_case):
        # fill up the stream dict with the values.
        # These are fed to the bokeh source stream
        for response in self._prom_responses:
            self._source_stream_dict[response] = \
                new_case.get_val(response)[:1]
        for sampled_variable in self._sampled_variables:
            self._source_stream_dict[sampled_variable] = \
                new_case.get_val(sampled_variable)[:1]
        self._source.stream(self._source_stream_dict)

    def _update_scatter_plots(self, new_case):
        # update the min and max for the response variable
        for response in self._prom_responses:
            self._prom_response_min[response] = min(
                self._prom_response_min[response], new_case.get_val(response)[:1][0]
            )
            self._prom_response_max[response] = max(
                self._prom_response_max[response], new_case.get_val(response)[:1][0]
            )

        # update the color mapper that is used to color the dots in the scatter plots
        self._color_mapper.low = self._prom_response_min[self._prom_response]
        self._color_mapper.high = self._prom_response_max[self._prom_response]

    def _update_histograms(self):
        for sampled_variable in self._sampled_variables:
            x_data = self._source.data[sampled_variable]

            hist, edges = np.histogram(
                x_data, bins=_num_histogram_bins, range=(np.min(x_data), np.max(x_data))
            )
            self._hist_source[sampled_variable].data.update(
                {
                    "top": hist,
                    "left": edges[:-1],  # left edge of each bin
                    "right": edges[1:],  # right edge of each bin
                }
            )

            self._hist_figures[sampled_variable].x_range.start = np.min(x_data)
            self._hist_figures[sampled_variable].x_range.end = np.max(x_data)

    def _setup_data_source(self):
        self._source_dict = {}

        outputs = self._case_tracker.get_case_reader().list_source_vars("driver", out_stream=None)[
            "outputs"
        ]
        responses = list(
            self._case_tracker.get_case_reader().problem_metadata["responses"].keys()
        )

        # convert to promoted names
        self._prom_responses = []
        for response in responses:
            if response in self._case_tracker.get_case_reader()._abs2prom["output"]:
                self._prom_responses.append(
                    self._case_tracker.get_case_reader()._abs2prom["output"][response]
                )
            else:
                raise RuntimeError(f"No prom for abs variable {response}")

        if not self._prom_responses:
            raise RuntimeError(f"Need at least one response variable.")

        # Don't include response variables in sampled variabales.
        # Also, split up the remaining sampled variables into scalars and
        # non-scalars
        self._sampled_variables = []
        self._sampled_variables_non_scalar = []
        for varname in outputs:
            if varname not in self._prom_responses:
                shape = self._case_tracker._get_shape(varname)
                if shape == () or shape == (1,): # is scalar ??
                    self._sampled_variables.append(varname)
                else:
                    self._sampled_variables_non_scalar.append(varname)
        self._num_sampled_variables = len(self._sampled_variables)

        # want them sorted in the Sampled Variables selection box
        self._sampled_variables.sort()
        self._sampled_variables_non_scalar.sort()

        # setup the source
        for response in self._prom_responses:
            self._source_dict[response] = []
        for sampled_variable in self._sampled_variables:
            self._source_dict[sampled_variable] = []
        self._source = ColumnDataSource(self._source_dict)

    def _sampled_variable_callback(self,var_name):
        def toggle_callback(attr, old, new):
            self._sampled_variables_visibility[var_name] = new
            self._hist_figures[var_name].visible = new
            self._row_labels[var_name].visible = new
            self._column_labels[var_name].visible = new

            for i, (y, x) in enumerate(
                product(self._sampled_variables, self._sampled_variables)
            ):
                icolumn = i % self._num_sampled_variables
                irow = i // self._num_sampled_variables
                # only do the lower half
                if x != y:
                    self._scatter_plots_figure[(x,y)].visible = (icolumn < irow \
                        and self._sampled_variables_visibility[x] and self._sampled_variables_visibility[y] )

        return toggle_callback

    def _set_initial_number_initial_visible_sampled_variables(self):
        number_initial_visible_sampled_variables = 0
        for sampled_var in self._sampled_variables:
            if number_initial_visible_sampled_variables < _max_number_initial_visible_sampled_variables:
                self._sampled_variables_visibility[sampled_var] = True
                number_initial_visible_sampled_variables += 1
            else:
                self._sampled_variables_visibility[sampled_var] = False

    def _make_sampled_variables_selection_buttons(self):
        # Make all the buttons for the Sample Variables area to the right of the plot
        #   that lets the user select what to plot
        number_initial_visible_sampled_variables = 0
        for sampled_var in self._sampled_variables:
            if self._sampled_variables_visibility[sampled_var]:
                sampled_variable_button = _make_sampled_variable_button(sampled_var,True, True, self._sampled_variable_callback(sampled_var))
            else:
                sampled_variable_button = _make_sampled_variable_button(sampled_var,False, True, self._sampled_variable_callback(sampled_var))
            self._sampled_variables_buttons.append(sampled_variable_button)

        for sampled_var in self._sampled_variables_non_scalar:
            sampled_variable_button = _make_sampled_variable_button(sampled_var,False, False, self._sampled_variable_callback(sampled_var))  # TODO need a callback ? Does it do anything?           
            self._sampled_variables_buttons.append(sampled_variable_button)

        sampled_variables_button_column = Column(
            children=self._sampled_variables_buttons,
            sizing_mode="stretch_both",
            height_policy="fit",
            styles={
                "overflow-y": "auto",
                "border": "1px solid #ddd",
                "padding": "8px",
                'max-height': '100vh'  # Ensures it doesn't exceed viewport
            },
        )

        # header for the variable list
        sampled_variables_label = Div(
            text="Sampled Variables",
            width=200,
            styles={"font-size": "20px", "font-weight": "bold"},
        )
        label_and_buttons_column = Column(
            sampled_variables_label,
            sampled_variables_button_column,
            sizing_mode="stretch_height",
            height_policy="fit",
            styles={
                'max-height': '100vh',  # Ensures it doesn't exceed viewport
                'border-radius': '5px',
                'border': '5px solid black',
            },
        )

        sampled_variable_selector_box = ScrollBox(
            child=label_and_buttons_column,
            sizing_mode="stretch_height",
            height_policy="max",
        )

        return sampled_variable_selector_box

    def _make_color_bar(self):
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

        # Need the color bar to be associated with a plot figure.
        # So make a basic one and hide it
        p = figure(height=2 * _grid_plot_height_and_width, width=0, toolbar_location=None)
        # Plot a line between the two points
        line = p.line([0, 1],[0, 1])
        p.grid.grid_line_color = None
        line.visible = False
        p.xaxis.visible = False
        p.yaxis.visible = False
        p.xgrid.visible = False
        p.ygrid.visible = False
        p.title.visible = False
        p.outline_line_alpha = 0
        p.min_border_right = 100

        p.add_layout(color_bar, "right")

        return p

    def _make_plots(self, plots_and_labels_in_grid):
        # x and y here refer to the x axis and y axis of the grid
        # Add plots by row starting from top
        for i, (y, x) in enumerate(  #  TODO Why y,x here ??
            product(self._sampled_variables, self._sampled_variables)
        ):
            icolumn = i % self._num_sampled_variables
            irow = i // self._num_sampled_variables

            if x != y:
                p = figure(
                    background_fill_color="#fafafa",
                    border_fill_color="white",
                    width=_grid_plot_height_and_width,
                    height=_grid_plot_height_and_width,
                    output_backend="webgl",
                )
                self._scatter_plots_figure[(x,y)] = p
                p.axis.visible = True
                self._scatter_plots.append(p.scatter(
                    x=x,
                    source=self._source,
                    y=y,
                    size=5,
                    line_color=None,
                    fill_color=transform(
                        self._prom_response, self._color_mapper
                    ),  # This maps value ofself._prom_response variable to colors
                ))
            else:  # on the diagonal
                # Extract the x column data for the histogram
                x_data = self._source.data[x]

                # TODO : DO I EVEN NEED THIS HERE? OR JUST in update?
                if x_data:
                    hist, edges = np.histogram(
                        x_data, bins=_num_histogram_bins, range=(np.min(x_data), np.max(x_data))
                    )
                else:
                    hist, edges = np.histogram(x_data, bins=_num_histogram_bins)

                # Create a new ColumnDataSource for the histogram data
                self._hist_source[x] = ColumnDataSource(
                    data={
                        "top": hist,  # height of each bin
                        "bottom": np.zeros(len(hist)),  # bottom of each bin starts at 0
                        "left": edges[:-1],  # left edge of each bin
                        "right": edges[1:],  # right edge of each bin
                    }
                )

                p = figure(
                    width=_grid_plot_height_and_width,
                    height=_grid_plot_height_and_width,
                    output_backend="webgl",
               )

                # Add the histogram bars using quad glyphs
                glyphs = p.quad(
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
                p = None
            else:
                # is it visible based on what the user has selected
                if self._sampled_variables_visibility[x] and self._sampled_variables_visibility[y]:
                    p.visible = True
                else:
                    p.visible = False

            plots_and_labels_in_grid.append(p)

    def _make_plot_labels(self, visible_variables, plots_and_labels_in_grid):
        # row labels
        for i, sampled_variable in enumerate(reversed(self._sampled_variables)):
            irow = self._num_sampled_variables - i - 1
            idx = irow * self._num_sampled_variables
            elided_variable_name_with_units = _elide_variable_name_with_units(sampled_variable, None)
            p = Div(
                text=f"<div style='text-align:center;font-size:12px;writing-mode:vertical-lr;transform:rotate(180deg); cursor:help;' title='{sampled_variable}'>{elided_variable_name_with_units}</div>",
                align="center",
            )
            self._row_labels[sampled_variable] = p
            p.visible = sampled_variable in visible_variables
            plots_and_labels_in_grid.insert(idx,p)

        # need to push one blank one for the bottom left corner in the grid
        p = Div(text=f"")
        plots_and_labels_in_grid.append(p)

        # column labels
        for sampled_variable in self._sampled_variables:
            units = self._case_tracker._get_units(sampled_variable)
            elided_variable_name_with_units = _elide_variable_name_with_units(sampled_variable, units)
            p = Div(
                text=f"<div style='text-align:center;font-size:14px; cursor:help;' title='{sampled_variable}'>{elided_variable_name_with_units}</div>",
                styles={"font-size": "12px", "text-align":"center"},
                align="center",
            )
            self._column_labels[sampled_variable] = p
            p.visible = sampled_variable in visible_variables
            plots_and_labels_in_grid.append(p)

    def _make_analysis_driver_box(self):
        analysis_progress_label = Div(
            text="Analysis Driver Progress",
            styles={"font-size": "20px", "font-weight": "bold"},
        )

        # placeholder until we have data
        self._analysis_driver_progress_text_box = Div(
            text="""<div style="padding: 10px; border-radius: 5px;">
                    <p>Waiting for data...</p>
                    </div>""",
            width=600,
            height=100,
        )

        analysis_progress_box = Column(
            analysis_progress_label,
            self._analysis_driver_progress_text_box,
            styles={
                "border-radius": "5px",
                "border": "5px solid black",
            },
        )
        return analysis_progress_box

    def _make_quit_button(self):
        quit_button = Button(label="Quit Application", button_type="danger")

        # Define callback function for the quit button
        def quit_app():
            raise KeyboardInterrupt("Quit button pressed")

        quit_button.on_click(quit_app)
        return quit_button

    def _make_title_div(self):
        script_name = self._case_recorder_filename

        title_box = Div(
            text=f"Analysis Driver Progress for {script_name}",
            styles={
                "font-size": "20px", 
                "font-weight": "bold",
                "border-radius": "5px",
                "border": "5px solid black",
                "padding": "8px",
                },
        )

        return title_box

    def _make_response_variable_selector(self, color_bar):
        menu = Select(
            options=self._prom_responses,
            value=self._prom_responses[0],  # Default value
        )

        response_variable_header = Div(
            text="Response variable",
            width=200,
            styles={"font-size": "20px", "font-weight": "bold"},
        )

        response_variable_box = Column(
            response_variable_header,
            menu,
            styles={
                "border": "5px solid black",
                "padding": "8px",
            },
        )

        def cb_select_response_variable(color_bar):
            def toggle_callback(attr, old, new):
                self._prom_response = new 
                color_bar.title = f"Response variable: '{new}'"
                self._color_mapper.low = self._prom_response_min[self._prom_response]
                self._color_mapper.high = self._prom_response_max[self._prom_response]

                for scatter_plot in self._scatter_plots:
                    scatter_plot.glyph.fill_color = transform(
                            self._prom_response, self._color_mapper
                        )
            return toggle_callback

        menu.on_change("value", cb_select_response_variable(color_bar))

        return response_variable_box

    def _make_overall_layout(self, title_box, grid_of_plots, color_bar, quit_button,
                             analysis_progress_box, response_variable_box, sampled_variable_selector_box ):

        self._overall_layout = Row(
            Column(title_box, grid_of_plots),
            color_bar,
            Spacer(width=150, height=0), # move the column away from the color bar
            Column(
                Spacer(height=20), # move quit button away from the top
                quit_button,
                Spacer(height=20),
                analysis_progress_box,
                Spacer(height=20),
                response_variable_box,
                Spacer(height=20),
                sampled_variable_selector_box,
                Spacer(height=20),
                sizing_mode="stretch_both",
            ),
            sizing_mode="stretch_both",
        )

    def _setup_figure(self):
        # Initially the response variable plotted is the first one
        self._prom_response = self._prom_responses[0]

        self._set_initial_number_initial_visible_sampled_variables()

        visible_variables = [var for var in self._sampled_variables if self._sampled_variables_visibility[var]]

        title_box = self._make_title_div()

        # Create a color mapper using Viridis (colorblind-friendly)
        self._color_mapper = LinearColorMapper(palette=_color_palette, 
                                               low=_initial_response_range_for_plots[0], 
                                               high=_initial_response_range_for_plots[1])

        color_bar = self._make_color_bar()

        plots_and_labels_in_grid = []
        self._make_plots(plots_and_labels_in_grid)

        self._make_plot_labels(visible_variables, plots_and_labels_in_grid)

        grid_of_plots = gridplot(
            plots_and_labels_in_grid,
            ncols=self._num_sampled_variables+1,  # need one extra row and column in the grid for the axes labels
            toolbar_location=None,
        )

        quit_button = self._make_quit_button()

        sampled_variable_selector_box = self._make_sampled_variables_selection_buttons()

        analysis_progress_box = self._make_analysis_driver_box()

        response_variable_box = self._make_response_variable_selector(color_bar)

        self._make_overall_layout(title_box, grid_of_plots, color_bar, quit_button, analysis_progress_box, 
            response_variable_box, sampled_variable_selector_box)

def print_bokeh_objects(doc):  # TODO remove!
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

        # Print to console when the document is loaded  TODO remove
        def on_document_ready(event):
            print_bokeh_objects(doc)
        # doc.on_event('document_ready', on_document_ready)

        _RealTimeOptPlot(
            case_recorder_filename,
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
