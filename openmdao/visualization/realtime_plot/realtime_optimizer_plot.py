"""A real-time plot monitoring the optimization process as an OpenMDAO script runs."""

from collections import defaultdict
import re

from openmdao.utils.shell_proc import _is_process_running
from openmdao.core.constants import INF_BOUND

try:
    from openmdao.visualization.realtime_plot.realtime_plot_class import _RealTimePlot
    from bokeh.models import (
        ColumnDataSource,
        LinearAxis,
        Range1d,
        Toggle,
        Button,
        Column,
        Row,
        CustomJS,
        Div,
        ScrollBox,
        BasicTicker,
        Checkbox,
    )

    from bokeh.models.tools import (
        BoxZoomTool,
        ResetTool,
        HoverTool,
        PanTool,
        WheelZoomTool,
        SaveTool,
        ZoomInTool,
        ZoomOutTool,
    )
    from bokeh.plotting import figure
    from bokeh.palettes import Category20, Colorblind

    bokeh_and_dependencies_available = True
except ImportError:
    bokeh_and_dependencies_available = False

import numpy as np


# Constants
# color of the plot line for the objective function
_obj_color = "black"
# color of the buttons for variables not being shown
_non_active_plot_color = "black"
_plot_line_width = 3
# how transparent is the area part of the plot for desvars that are vectors
_varea_alpha = 0.3
# for variables that are vectors, we use bokeh's varea plotting element
#   which is invisible if the min and max of the varea is the same.
#   Use this width to keep it from disappearing.
#   0.001 seems to be a good fraction, not too thick but still visible
_varea_min_width = 0.001
# the CSS for the toggle buttons to let user choose what variables to plot
_variable_list_header_font_size = "14"
_toggle_styles = """
    font-size: 22px;
    box-shadow:
        0 4px 6px rgba(0, 0, 0, 0.1),    /* Distant shadow */
        0 1px 3px rgba(0, 0, 0, 0.08),   /* Close shadow */
        inset 0 2px 2px rgba(255, 255, 255, 0.2);  /* Top inner highlight */
"""

# Used for the outer bounds for the hstrips used to indicate the out of bounds for variables
# TODO need to come up with a better way. But if made too big, get glitches in plots
_bounds_infinity = 1e5

# colors used for the plot lines and associated buttons and axes labels
# start with color-blind friendly colors and then use others if needed
if bokeh_and_dependencies_available:
    _colorPalette = Colorblind[8] + Category20[20]

# This is the JavaScript code that gets run when a user clicks on
#   one of the toggle buttons that change what variables are plotted
variable_button_callback_code = f"""
// The ColorManager provides color from a palette. When the
//   user turns off the plotting of a variable, the color is returned to the ColorManager
//   for later use with a different variable plot
if (typeof window.ColorManager === 'undefined') {{
    window.ColorManager = class {{
        constructor() {{
            if (ColorManager.instance) {{
                return ColorManager.instance;
            }}

            this.palette = colorPalette;
            this.usedColors = new Set();
            this.variableColorMap = new Map();
            ColorManager.instance = this;
        }} //  end of constructor

        getColor(variableName) {{
            if (this.variableColorMap.has(variableName)) {{
                return this.variableColorMap.get(variableName);
            }}

            const availableColor = this.palette.find(color => !this.usedColors.has(color));
            const newColor = availableColor ||
                                this.palette[this.usedColors.size % this.palette.length];

            this.usedColors.add(newColor);
            this.variableColorMap.set(variableName, newColor);
            return newColor;
        }} // end of getColor

        releaseColor(variableName) {{
            const color = this.variableColorMap.get(variableName);
            if (color) {{
                this.usedColors.delete(color);
                this.variableColorMap.delete(variableName);
            }}
        }} // end of releaseColor

    }}; // end of class definition

    window.colorManager = new window.ColorManager();
}}

// Get the toggle that triggered the callback
const toggle = cb_obj;
const var_type = toggle.tags[0].var_type;
const index = toggle.tags[0].index;
const index_checkbox = index - 1; // objective does not have checkbox
const checkbox = bounds_off_on_checkboxes[index_checkbox];
const varname = cb_obj.label;

// index value of 0 is for the objective variable whose axis
// is on the left. The index variable refers to the list of toggle buttons.
// The axes list variable only is for desvars and cons, whose axes are on the right.
// The lines list variables includes all vars

// Set line visibility
lines[index].visible = toggle.active;

// Set axis visibility if it exists (all except first line)
if (var_type == "desvars" || var_type == "cons") {{
    axes[index-1].visible = toggle.active;
}}

// if turning on, get a color and set the line, axis label, and toggle button to that color
if (toggle.active) {{
    let color = window.colorManager.getColor(varname);
    if (var_type == "desvars" || var_type == "cons") {{
        axes[index-1].axis_label_text_color = color
    }}

    // using set_value is a workaround because of a bug in Bokeh.
    // see https://github.com/bokeh/bokeh/issues/14364 for more info
    if (lines[index].glyph.type == "VArea"){{
        lines[index].glyph.properties.fill_color.set_value(color);
    }}
    if (lines[index].glyph.type == "Line"){{
        lines[index].glyph.properties.line_color.set_value(color);
    }}

    // enable bounds_off_on_checkbox
    checkbox.disabled = false;

    // turn on bounds display if bounds checkbox is on and var is a desvar
    if (var_type == "desvars" && checkbox.active) {{
        const index_desvar = index - 1;
        desvar_lower_bound_indicators[index_desvar].visible = true;
        desvar_upper_bound_indicators[index_desvar].visible = true;
        // TODO how to handle vars that are cons and desvars
        desvar_bound_indicator_source.data[varname] = [color];
        desvar_bound_indicator_source.change.emit();
    }}

    // turn on bounds display bounds checkbox is on and var is a cons
    if (var_type == "cons" && checkbox.active) {{
        const index_cons = index - num_desvars - 1;
        constraint_lower_bound_indicators[index_cons].visible = true;
        constraint_upper_bound_indicators[index_cons].visible = true;
        // TODO how to handle vars that are cons and desvars
        constraint_bound_indicator_source.data[varname] = [color];
        constraint_bound_indicator_source.change.emit();
    }}


    // make the button background color the same as the line, just slightly transparent
    toggle.stylesheets = [`
        .bk-btn.bk-active {{
            background-color: rgb(from ${{color}} R G B / 0.3);
            {_toggle_styles}
        }}
    `];
}} else {{
    // if turning off a variable, return the color to the pool
    window.colorManager.releaseColor(varname);
    if (var_type == "desvars") {{
        const index_desvar = index - 1;
        desvar_lower_bound_indicators[index_desvar].visible = false;
        desvar_upper_bound_indicators[index_desvar].visible = false;
    }}
    if (var_type == "cons") {{
        const index_cons = index - num_desvars - 1;
        constraint_lower_bound_indicators[index_cons].visible = false;
        constraint_upper_bound_indicators[index_cons].visible = false;
    }}
    // if the variable is not being displayed, do not let the user turn on bounds display
    checkbox.disabled = true;
    toggle.stylesheets = [`
        .bk-btn {{
            {_toggle_styles}
        }}
    `];

}}
"""
bounds_off_on_callback_code = """
// Get the checkbox that triggered the callback
const checkbox = cb_obj;
// the checkbox was created with the tags set to a list
//  with the single item of a dict with some key values in it
const var_type = checkbox.tags[0].var_type;
const index = checkbox.tags[0].index;
const varname = checkbox.tags[0].varname;
let plot_line_color;

// Set bounds violation indicators visibility
if (var_type == "desvars") {{
    const index_desvar = index;
    desvar_lower_bound_indicators[index_desvar].properties.visible.set_value(checkbox.active);
    desvar_upper_bound_indicators[index_desvar].properties.visible.set_value(checkbox.active);

    if (checkbox.active){{
        // set bounds indicator color to plot line color
        const line_index = index + 1 ; // the objective line is the first line

        if (lines[line_index].glyph.type == "VArea"){{
            plot_line_color = lines[line_index].glyph.properties.fill_color.get_value().value;
        }} else {{
            plot_line_color = lines[line_index].glyph.properties.line_color.get_value().value;
        }}

        desvar_bound_indicator_source.data[varname] = [plot_line_color];
        desvar_bound_indicator_source.change.emit();
    }}
}}
// Set bounds violation indicators visibility
if (var_type == "cons") {{
    const index_cons = index - num_desvars;
    constraint_lower_bound_indicators[index_cons].properties.visible.set_value(checkbox.active);
    constraint_upper_bound_indicators[index_cons].properties.visible.set_value(checkbox.active);

    if (checkbox.active){{
        // set bounds indicator color to plot line color
        const line_index = index + 1 ; // the objective line is the first line

        if (lines[line_index].glyph.type == "VArea"){{
            plot_line_color = lines[line_index].glyph.properties.fill_color.get_value().value;
        }} else {{
            plot_line_color = lines[line_index].glyph.properties.line_color.get_value().value;
        }}

        constraint_bound_indicator_source.data[varname] = [plot_line_color];
        constraint_bound_indicator_source.change.emit();
    }}
}}
"""


def _update_y_min_max(name, y, y_min, y_max):
    """
    Update the y_min and y_max dicts containing the min and max of the variables.

    Parameters
    ----------
    name : str
        Name of the variable.
    y : double
        Value of the variable.
    y_min : dict
        Dict of mins of each variable.
    y_max : dict
        Dict of maxs of each variable.

    Returns
    -------
    bool
        True if either min or max were updated.
    """
    min_max_changed = False
    if y < y_min[name]:
        y_min[name] = y
        min_max_changed = True
    if y > y_max[name]:
        y_max[name] = y
        min_max_changed = True
    return min_max_changed


def _get_value_for_plotting(value_from_recorder, var_type):
    """
    Return the double value to be used for plotting the variable.

    Handles variables that are vectors.

    Parameters
    ----------
    value_from_recorder : numpy array
        Value of the variable.
    var_type : str
        String indicating which type of variable it is: 'objs', 'desvars' or 'cons'.

    Returns
    -------
    double
        The value to be used for plotting the variable.
    """
    if value_from_recorder is None or value_from_recorder.size == 0:
        return 0.0
    if var_type == "cons":
        # plot the worst case value
        if value_from_recorder.size == 1:
            return value_from_recorder.item()
        else:
            return np.linalg.norm(value_from_recorder, ord=np.inf)
    elif var_type == "objs":
        return value_from_recorder.item()  # get as scalar
    else:  # for desvars, use L2 norm
        if value_from_recorder.size == 1:
            return value_from_recorder.item()
        else:
            return np.linalg.norm(value_from_recorder)


def _make_header_text_for_variable_chooser(header_text):
    """
    Return a Div to be used for the label for the type of variables in the variable list.

    Parameters
    ----------
    header_text : str
        Label string.

    Returns
    -------
    Div
        The header Div for the section of variables of that type.
    """
    header_text_div = Div(
        text=f"<b>{header_text}</b>",
        styles={"font-size": _variable_list_header_font_size},
    )
    return header_text_div


class _RealTimeOptimizerPlot(_RealTimePlot):
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
    script : str or None
        The name of the script used to create the case recorder file.
    """

    def __init__(
        self, case_tracker, callback_period, doc, pid_of_calling_script, script
    ):
        """
        Construct and initialize _RealTimeOptPlot instance.
        """
        # self._case_recorder_filename = case_recorder_filename
        super().__init__(
            case_tracker, callback_period, doc, pid_of_calling_script, script
        )

        self._lines = []
        self._toggles = []  # includes only the toggle buttons
        self._bounds_off_on_checkboxes = (
            []
        )  # includes only the checkboxes for bounds on/off
        self._column_items = (
            []
        )  # includes all items in the Column, including headers and toggles
        self._axes = []
        # flag to prevent updating label with units each time we get new data
        self._labels_updated_with_units = False

        self._source = None  # main source for data values for plots

        # used for showing which constraint points are out of bounds
        # For constraints
        self._constraint_bound_indicator_source = None
        self._constraint_lower_bound_indicators = []
        self._constraint_upper_bound_indicators = []

        # for design var/driver bounds
        self._desvar_bound_indicator_source = None
        self._desvar_lower_bound_indicators = []
        self._desvar_upper_bound_indicators = []

        self._num_desvars = 0

        self._setup_figure()

        # used to keep track of the y min and max of the data so that
        # the axes ranges can be adjusted as data comes in
        self._y_min = defaultdict(lambda: float("inf"))
        self._y_max = defaultdict(lambda: float("-inf"))

        doc.add_periodic_callback(self._update_wrapped_in_try, callback_period)
        doc.title = "OpenMDAO Optimization Progress Plot"

    def _setup_plotting(self, new_data):
        # when we first get some data, we can start building the
        # plot since we know about the variables and their metadata.
        self._setup_data_source()

        # Check to make sure we have one and only one objective before going farther
        obj_names = self._case_tracker._get_obj_names()
        if len(obj_names) != 1:
            raise ValueError(
                f"Plot requires there to be one and only one objective \
                    but {len(obj_names)} objectives found"
            )

        # setup sources for the colors of the bounds indicators
        # Changing values of the quads used to draw the bounds indicators
        #  is the only way to get this to work when changing via
        #  javascipt callbacks.
        desvar_bound_indicator_source_dict = {}
        desvar_names = self._case_tracker._get_desvar_names()
        for desvar_name in desvar_names:
            units = self._case_tracker._get_units(desvar_name)
            desvar_name_with_type = desvar_name
            if desvar_name in self._both_desvars_and_cons:
                desvar_name_with_type += " [dv]"
            desvar_button_label = f"{desvar_name_with_type} ({units})"
            desvar_bound_indicator_source_dict[desvar_button_label] = [
                "black"
            ]
        self._desvar_bound_indicator_source = ColumnDataSource(
            data=desvar_bound_indicator_source_dict
        )

        constraint_bound_indicator_source_dict = {}
        cons_names = self._case_tracker._get_cons_names()
        for cons_name in cons_names:
            units = self._case_tracker._get_units(cons_name)
            con_name_with_type = cons_name
            if cons_name in self._both_desvars_and_cons:
                con_name_with_type += " [cons]"
            cons_button_label = f"{con_name_with_type} ({units})"
            constraint_bound_indicator_source_dict[cons_button_label] = [
                "black"
            ]
        self._constraint_bound_indicator_source = ColumnDataSource(
            data=constraint_bound_indicator_source_dict
        )

        # Create CustomJS callback for toggle buttons.
        # Pass in the data from the Python side that the JavaScript side
        #   needs
        bounds_off_on_callback = CustomJS(
            args=dict(
                num_desvars=self._num_desvars,
                lines=self._lines,
                desvar_bound_indicator_source=self._desvar_bound_indicator_source,
                desvar_lower_bound_indicators=self._desvar_lower_bound_indicators,
                desvar_upper_bound_indicators=self._desvar_upper_bound_indicators,
                constraint_bound_indicator_source=self._constraint_bound_indicator_source,
                constraint_lower_bound_indicators=self._constraint_lower_bound_indicators,
                constraint_upper_bound_indicators=self._constraint_upper_bound_indicators,
            ),
            code=bounds_off_on_callback_code,
        )

        variable_button_callback = CustomJS(
            args=dict(
                num_desvars=self._num_desvars,
                lines=self._lines,
                axes=self._axes,
                bounds_off_on_checkboxes=self._bounds_off_on_checkboxes,
                desvar_bound_indicator_source=self._desvar_bound_indicator_source,
                desvar_lower_bound_indicators=self._desvar_lower_bound_indicators,
                desvar_upper_bound_indicators=self._desvar_upper_bound_indicators,
                constraint_bound_indicator_source=self._constraint_bound_indicator_source,
                constraint_lower_bound_indicators=self._constraint_lower_bound_indicators,
                constraint_upper_bound_indicators=self._constraint_upper_bound_indicators,
                colorPalette=_colorPalette,
            ),
            code=variable_button_callback_code,
        )

        # For the variables, make lines, axes, and the buttons to turn on and
        #   off the variable plot.
        # All the lines and axes for the desvars and cons are created in
        #   Python but initially are not visible. They are turned on and
        #   off on the JavaScript side.

        # objs
        obj_label = _make_header_text_for_variable_chooser("OBJECTIVE")
        self._column_items.append(obj_label)

        self.plot_figure.x_range = Range1d(1, 2)  # just to start

        for i, obj_name in enumerate(obj_names):
            units = self._case_tracker._get_units(obj_name)
            self.plot_figure.yaxis.axis_label = f"{obj_name} ({units})"
            self._make_variable_button(
                f"{obj_name} ({units})",
                "objs",
                True,
                variable_button_callback,
                bounds_off_on_callback,
            )
            self._make_line_and_hover_tool(
                "objs", obj_name, False, _obj_color, "solid", True
            )
            value = new_data["objs"][obj_name]
            float_value = _get_value_for_plotting(value, "objs")
            # just give it some non-zero initial range since we only have one point
            self.plot_figure.y_range = Range1d(float_value - 1, float_value + 1)

        # desvars
        desvars_label = _make_header_text_for_variable_chooser("DESIGN VARIABLES")
        self._column_items.append(desvars_label)
        desvar_names = self._case_tracker._get_desvar_names()
        for i, desvar_name in enumerate(desvar_names):
            units = self._case_tracker._get_units(desvar_name)

            desvar_name_with_type = desvar_name
            if desvar_name in self._both_desvars_and_cons:
                desvar_name_with_type += " [dv]"

            desvar_button_label = f"{desvar_name_with_type} ({units})"
            self._make_variable_button(
                desvar_button_label,
                "desvars",
                False,
                variable_button_callback,
                bounds_off_on_callback,
            )
            value = new_data["desvars"][desvar_name]
            # for desvars, if value is a vector, use Bokeh Varea glyph
            use_varea = value.size > 1
            self._make_line_and_hover_tool(
                "desvars",
                desvar_name_with_type,
                use_varea,
                _non_active_plot_color,
                "solid",
                False,
            )
            float_value = _get_value_for_plotting(value, "desvars")
            self._make_axis("desvars", desvar_name_with_type, float_value, units)

        # TODO include desvar driver constraints

        # cons
        cons_label = _make_header_text_for_variable_chooser("CONSTRAINTS")
        self._column_items.append(cons_label)
        cons_names = self._case_tracker._get_cons_names()
        for i, cons_name in enumerate(cons_names):
            units = self._case_tracker._get_units(cons_name)
            con_name_with_type = cons_name
            if cons_name in self._both_desvars_and_cons:
                con_name_with_type += " [cons]"

            cons_button_label = f"{con_name_with_type} ({units})"
            self._make_variable_button(
                cons_button_label,
                "cons",
                False,
                variable_button_callback,
                bounds_off_on_callback,
            )
            self._make_line_and_hover_tool(
                "cons",
                con_name_with_type,
                False,
                _non_active_plot_color,
                "dashed",
                False,
            )
            value = new_data["cons"][cons_name]
            float_value = _get_value_for_plotting(value, "cons")
            self._make_axis("cons", con_name_with_type, float_value, units)

        # Create a Column of the variable buttons and headers inside a scrolling window
        toggle_column = Column(
            children=self._column_items,
            sizing_mode="stretch_both",
            height_policy="fit",
            styles={
                "overflow-y": "auto",
                "border": "1px solid #ddd",
                "padding": "8px",
                "background-color": "#dddddd",
                "max-height": "100vh",  # Ensures it doesn't exceed viewport
            },
        )

        quit_button = Button(label="Quit Application", button_type="danger")

        # Define callback function for the quit button
        def quit_app():
            raise KeyboardInterrupt("Quit button pressed")

        # Attach the callback to the button
        quit_button.on_click(quit_app)

        # header for the variable list
        label = Div(
            text="Variables",
            width=200,
            styles={"font-size": "20px", "font-weight": "bold"},
        )
        label_and_toggle_column = Column(
            quit_button,
            label,
            toggle_column,
            sizing_mode="stretch_height",
            height_policy="fit",
            styles={"max-height": "100vh"},  # Ensures it doesn't exceed viewport
        )

        scroll_box = ScrollBox(
            child=label_and_toggle_column,
            sizing_mode="stretch_height",
            height_policy="max",
        )

        graph = Row(self.plot_figure, scroll_box, sizing_mode="stretch_both")
        self._doc.add_root(graph)

    def _update(self):
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

                # TODO need to do the same for the bounds source
                # self._constraint_bound_indicator_source

            return

        new_data = self._case_tracker._get_data_from_case(new_case)

        # See if Bokeh source object is defined yet. If not, set it up
        # since now we have data from the case recorder with info about the
        # variables to be plotted.
        if self._source is None:
            self._setup_plotting(new_data)

        # Do the actual update of the plot including updating the plot range and adding the new
        # data to the Bokeh plot stream
        counter = new_data["counter"]
        self._source_stream_dict = {"iteration": [counter]}
        self.plot_figure.x_range.end = counter

        iline = 0
        for obj_name, obj_value in new_data["objs"].items():
            float_obj_value = _get_value_for_plotting(obj_value, "objs")
            self._source_stream_dict[obj_name] = [float_obj_value]
            min_max_changed = _update_y_min_max(
                obj_name, float_obj_value, self._y_min, self._y_max
            )
            if min_max_changed:
                self.plot_figure.y_range.start = self._y_min[obj_name]
                self.plot_figure.y_range.end = self._y_max[obj_name]
            iline += 1

        for desvar_name, desvar_value in new_data["desvars"].items():
            desvar_name_with_type = desvar_name
            if desvar_name in self._both_desvars_and_cons:
                desvar_name_with_type += " [dv]"
            if not self._labels_updated_with_units and desvar_value.size > 1:
                units = self._case_tracker._get_units(desvar_name)
                desvar_button_label = (
                    desvar_name_with_type + f" ({units}) {desvar_value.shape}"
                )
                self._toggles[iline].label = desvar_button_label

            # handle non-scalar desvars
            min_max_changed = False
            min_max_changed = min_max_changed or _update_y_min_max(
                desvar_name, np.min(desvar_value), self._y_min, self._y_max
            )
            min_max_changed = min_max_changed or _update_y_min_max(
                desvar_name, np.max(desvar_value), self._y_min, self._y_max
            )
            if min_max_changed:
                self.plot_figure.extra_y_ranges[
                    f"extra_y_{desvar_name_with_type}_min"
                ].start = self._y_min[desvar_name]
                self.plot_figure.extra_y_ranges[
                    f"extra_y_{desvar_name_with_type}_min"
                ].end = self._y_max[desvar_name]
            # deal with when min and max are the same.
            # Otherwise the varea plot shows nothing, not even a line
            if np.min(desvar_value) == np.max(desvar_value):
                range = self._y_max[desvar_name] - self._y_min[desvar_name]
                min_thickness = range * _varea_min_width
                y1 = np.min(desvar_value) - min_thickness
                y2 = np.min(desvar_value) + min_thickness
            else:
                y1 = np.min(desvar_value)
                y2 = np.max(desvar_value)

            self._source_stream_dict[f"{desvar_name_with_type}_min"] = [y1]
            self._source_stream_dict[f"{desvar_name_with_type}_max"] = [y2]
            iline += 1

        for cons_name, cons_value in new_data["cons"].items():
            float_cons_value = _get_value_for_plotting(cons_value, "cons")
            con_name_with_type = cons_name
            if cons_name in self._both_desvars_and_cons:
                con_name_with_type += " [cons]"
            if not self._labels_updated_with_units and cons_value.size > 1:
                units = self._case_tracker._get_units(cons_name)
                cons_button_label = (
                    con_name_with_type + f" ({units}) {cons_value.shape}"
                )
                self._toggles[iline].label = cons_button_label
            self._source_stream_dict[con_name_with_type] = [float_cons_value]

            # handle non-scalar cons
            min_max_changed = False
            min_max_changed = min_max_changed or _update_y_min_max(
                con_name_with_type, np.min(cons_value), self._y_min, self._y_max
            )
            min_max_changed = min_max_changed or _update_y_min_max(
                con_name_with_type, np.max(cons_value), self._y_min, self._y_max
            )
            if min_max_changed:
                self.plot_figure.extra_y_ranges[
                    f"extra_y_{con_name_with_type}"
                ].start = self._y_min[con_name_with_type]
                self.plot_figure.extra_y_ranges[f"extra_y_{con_name_with_type}"].end = (
                    self._y_max[con_name_with_type]
                )
            iline += 1

        self._source.stream(self._source_stream_dict)

        self._labels_updated_with_units = True
        # end of _update method

    def _setup_data_source(self):
        self._source_dict = {"iteration": []}

        obj_names = self._case_tracker._get_obj_names()
        desvar_names = self._case_tracker._get_desvar_names()
        con_names = self._case_tracker._get_cons_names()
        self._both_desvars_and_cons = list(
            set(desvar_names).intersection(set(con_names))
        )

        # Obj
        for obj_name in obj_names:
            self._source_dict[obj_name] = []

        # Desvars
        for desvar_name in desvar_names:
            desvar_name_with_type = desvar_name
            if desvar_name in self._both_desvars_and_cons:
                desvar_name_with_type += " [dv]"
            self._source_dict[f"{desvar_name_with_type}_min"] = []
            self._source_dict[f"{desvar_name_with_type}_max"] = []
            self._num_desvars += 1

        # Cons

        for con_name in con_names:
            con_name_with_type = con_name
            if con_name in self._both_desvars_and_cons:
                con_name_with_type += " [cons]"
            self._source_dict[con_name_with_type] = []
        self._source = ColumnDataSource(self._source_dict)

    def _make_variable_button(
        self, varname, var_type, active, callback, bounds_off_on_callback
    ):
        index = len(self._toggles)
        toggle = Toggle(
            label=varname,
            active=active,
            margin=(0, 0, 8, 0),
            tags=[{"var_type": var_type, "index": index}],
        )
        toggle.js_on_change("active", callback)
        self._toggles.append(toggle)

        if var_type != "objs":
            index = len(self._bounds_off_on_checkboxes)
            checkbox = Checkbox(
                active=False,
                # visible=False,
                html_attributes={
                    "title": "Turn on/off plot of bounds for this variable. "
                    "Variable must be plotted to turn on."
                },
                disabled=True,
                margin=(12, 0, 8, 4),
                tags=[{"var_type": var_type, "index": index, "varname": varname}],
            )
            checkbox.js_on_change("active", bounds_off_on_callback)
            self._bounds_off_on_checkboxes.append(checkbox)
            self._column_items.append(Row(toggle, checkbox))
        else:
            self._column_items.append(toggle)

        # Add custom CSS styles for both active and inactive states
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
        return toggle

    def _make_bounds_display(self, top, bottom, source, varname):
        bounds_display = self.plot_figure.hstrip(
            y1=top,
            y0=bottom,
            source=source,
            fill_alpha=0.1,
            fill_color=varname,
            visible=False,
        )
        return bounds_display

    def _make_line_and_hover_tool(
        self, var_type, varname, use_varea, color, line_dash, visible
    ):
        if use_varea:
            line = self.plot_figure.varea(
                x="iteration",
                y1=f"{varname}_min",
                y2=f"{varname}_max",
                source=self._source,
                color=color,
                alpha=_varea_alpha,
                visible=visible,
            )
        else:
            if var_type == "desvars":
                y_name = f"{varname}_min"
            else:
                y_name = varname
            line = self.plot_figure.line(
                x="iteration",
                y=y_name,
                line_width=_plot_line_width,
                line_dash=line_dash,
                source=self._source,
                color=color,
                visible=visible,
            )

        # make graphics showing bounds
        if var_type == "desvars":
            varname_minus_type = re.sub(r"\s*\[.*?\]$", "", varname)
            lower_bound, upper_bound = self._case_tracker._get_desvar_bounds(
                varname_minus_type
            )

            units = self._case_tracker._get_units(varname_minus_type)
            desvars_button_label = f"{varname} ({units})"

            if upper_bound != INF_BOUND:
                upper_bound_indicator = self._make_bounds_display(
                    _bounds_infinity,
                    upper_bound,
                    self._desvar_bound_indicator_source,
                    desvars_button_label,
                )
            else:
                upper_bound_indicator = self._make_bounds_display(
                    _bounds_infinity,
                    upper_bound,
                    self._desvar_bound_indicator_source,
                    desvars_button_label,
                )
                upper_bound_indicator = self.plot_figure.quad(
                    right=0.0,
                    left=0.0,
                    top=0.0,
                    bottom=0.0,
                    visible=False,
                )

            if lower_bound != -INF_BOUND:
                lower_bound_indicator = self._make_bounds_display(
                    lower_bound,
                    -_bounds_infinity,
                    self._desvar_bound_indicator_source,
                    desvars_button_label,
                )

            else:
                lower_bound_indicator = self.plot_figure.quad(
                    right=0.0,
                    left=0.0,
                    top=0.0,
                    bottom=0.0,
                    visible=False,
                )

            self._desvar_lower_bound_indicators.append(
                lower_bound_indicator
            )
            self._desvar_upper_bound_indicators.append(
                upper_bound_indicator
            )

        if var_type == "cons":
            # Match optional whitespace followed by [anything] at the end of string
            varname_minus_type = re.sub(r"\s*\[.*?\]$", "", varname)

            lower_bound, upper_bound = self._case_tracker._get_constraint_bounds(
                varname_minus_type
            )

            # varname includes if cons, but not units so need to add it here
            units = self._case_tracker._get_units(varname_minus_type)
            cons_button_label = f"{varname} ({units})"

            upper_bound_indicator = self._make_bounds_display(
                _bounds_infinity,
                upper_bound,
                self._constraint_bound_indicator_source,
                cons_button_label,
            )

            lower_bound_indicator = self._make_bounds_display(
                lower_bound,
                -_bounds_infinity,
                self._constraint_bound_indicator_source,
                cons_button_label,
            )

            self._constraint_lower_bound_indicators.append(
                lower_bound_indicator
            )
            self._constraint_upper_bound_indicators.append(
                upper_bound_indicator
            )

        if var_type == "desvars":
            line.y_range_name = f"extra_y_{varname}_min"
            lower_bound_indicator.y_range_name = f"extra_y_{varname}_min"
            upper_bound_indicator.y_range_name = f"extra_y_{varname}_min"
        elif var_type == "cons":
            line.y_range_name = f"extra_y_{varname}"
            lower_bound_indicator.y_range_name = f"extra_y_{varname}"
            upper_bound_indicator.y_range_name = f"extra_y_{varname}"

        self._lines.append(line)
        if not use_varea:  # hover tool does not work with Varea
            if var_type == "desvars":
                datasource_name = f"{varname}_min"
            else:
                datasource_name = varname
            hover = HoverTool(
                renderers=[line],
                # list of tuples. Each is label followed by value
                tooltips=[
                    ("Iteration", "@iteration"),
                    (varname, "@{%s}" % datasource_name + "{0.00}"),
                ],
                mode="vline",
                visible=visible,
            )
            self.plot_figure.add_tools(hover)

    def _make_axis(self, var_type, varname, plot_value, units):
        # Make axis for this variable on the right of the plot
        if var_type == "desvars":
            y_range_name = f"extra_y_{varname}_min"
        else:
            y_range_name = f"extra_y_{varname}"

        extra_y_axis = LinearAxis(
            y_range_name=y_range_name,
            axis_label=f"{varname} ({units})",
            axis_label_text_font_size="20px",
            visible=False,
        )
        self._axes.append(extra_y_axis)
        self.plot_figure.add_layout(extra_y_axis, "right")

        self.plot_figure.extra_y_ranges[y_range_name] = Range1d(
            plot_value - 1, plot_value + 1
        )

    def _setup_figure(self):
        # Make the figure and all the settings for it
        if self._script:
            title = f"Optimization Progress Plot for: {self._script}"
        else:
            title = (
                "Optimization Progress Plot for: "
                f"{self._case_tracker.get_case_recorder_filename()}"
            )

        # need to do these separate since we want to make them the default
        pan_tool = PanTool()
        wheel_zoom_both = WheelZoomTool()
        self.plot_figure = figure(
            tools=[
                pan_tool,
                WheelZoomTool(dimensions="width"),
                wheel_zoom_both,
                ZoomInTool(),
                ZoomOutTool(),
                BoxZoomTool(),
                ResetTool(),
                SaveTool(),
            ],
            width_policy="max",
            height_policy="max",
            sizing_mode="stretch_both",
            title=title,
            active_drag=pan_tool,
            active_scroll=wheel_zoom_both,
            active_tap=None,
            output_backend="webgl",
        )

        self.plot_figure.title.text_font_size = "14px"
        self.plot_figure.title.text_color = "black"
        self.plot_figure.title.text_font = "arial"
        self.plot_figure.title.align = "left"
        self.plot_figure.title.standoff = 40  # Adds 40 pixels of space below the title

        self.plot_figure.xaxis.axis_label = "Driver iterations"
        self.plot_figure.xaxis.minor_tick_line_color = None
        self.plot_figure.xaxis.ticker = BasicTicker(
            desired_num_ticks=10, min_interval=1
        )

        self.plot_figure.axis.axis_label_text_font_style = "bold"
        self.plot_figure.axis.axis_label_text_font_size = "20pt"
