"""A real-time plot monitoring the optimization process as an OpenMDAO script runs."""

from collections import defaultdict

from openmdao.utils.shell_proc import _is_process_running

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
# the CSS for the toggle buttons to let user choose what variables to plot
_variable_list_header_font_size = "14"
_toggle_styles = """
    font-size: 22px;
    box-shadow:
        0 4px 6px rgba(0, 0, 0, 0.1),    /* Distant shadow */
        0 1px 3px rgba(0, 0, 0, 0.08),   /* Close shadow */
        inset 0 2px 2px rgba(255, 255, 255, 0.2);  /* Top inner highlight */
"""
_bounds_hatch_alpha = 0.3
_bounds_hatch_pattern = 'cross'
_bounds_hatch_weight = 1
_bounds_line_width = 1
_bounds_alpha = 0.0
_bounds_infinity = 1e8
_bounds_left = -1

# colors used for the plot lines and associated buttons and axes labels
# start with color-blind friendly colors and then use others if needed
if bokeh_and_dependencies_available:
    _colorPalette = Colorblind[8] + Category20[20]

# This is the JavaScript code that gets run when a user clicks on
#   one of the toggle buttons that change what variables are plotted
callback_code = f"""
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
const checkbox = bounds_off_on_checkboxes[index - 1];
const variable_name = cb_obj.label;

// index value of 0 is for the objective variable whose axis
// is on the left. The index variable really refers to the list of toggle buttons.
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
    let color = window.colorManager.getColor(variable_name);
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
    
    // enable bounds checkbox
    checkbox.disabled = false;
    
    if (var_type == "cons" && checkbox.active) {{
        const index_cons = index - num_desvars - 1;
        lower_bound_violation_indicators[index_cons].active = true;
        upper_bound_violation_indicators[index_cons].active = true;
        bound_violation_indicator_source.data[variable_name] = [color];
        bound_violation_indicator_source.change.emit();
        // bound_violation_indicator_source.data["hatch_color"] = color;
        //lower_bound_violation_indicators[index_cons].glyph.properties.fill_color=color;
        //upper_bound_violation_indicators[index_cons].glyph.properties.fill_color=color;
        //lower_bound_violation_indicators[index_cons].glyph.properties.fill_color.set_value(color);
        //upper_bound_violation_indicators[index_cons].glyph.properties.fill_color.set_value(color);
    }} 
  

    // make the button background color the same as the line, just slightly transparent
    toggle.stylesheets = [`
        .bk-btn.bk-active {{
            background-color: rgb(from ${{color}} R G B / 0.3);
            {_toggle_styles}
        }}
    `];
// if turning off a variable, return the color to the pool
}} else {{
    window.colorManager.releaseColor(variable_name);
    if (var_type == "cons") {{
        const index_cons = index - num_desvars - 1;
        lower_bound_violation_indicators[index_cons].active = false;
        upper_bound_violation_indicators[index_cons].active = false;
    }}
    checkbox.disabled = true;
    toggle.stylesheets = [`
        .bk-btn {{
            {_toggle_styles}
        }}
    `];

}}
"""
bounds_off_on_callback_code = f"""
// Get the toggle that triggered the callback
const checkbox = cb_obj;
// the checkbox was created with the name set to the index of 
//  the checkbox in the list of checkboxes, so we can 
//  determine which checkbox was checked in this callback and 
//  use that index to know which bounds graphics to turn off and on
const var_type = checkbox.tags[0].var_type;
const index = checkbox.tags[0].index;
const variable_name = checkbox.tags[0].variable_name;

// index value of 0 is for the objective variable whose axis
// is on the left. The index variable really refers to the list of toggle buttons.
// The axes list variable only is for desvars and cons, whose axes are on the right.
// The lines list variables includes all vars
 
// Set cons_violation_indicators visibility
if (var_type == "cons") {{
    debugger;
    const index_cons = index - num_desvars;
    
    console.log("lower_bound_violation_indicators = " + lower_bound_violation_indicators);
    console.log("index_cons = " + index_cons);
    lower_bound_violation_indicators[index_cons].properties.visible.set_value(checkbox.active);
    upper_bound_violation_indicators[index_cons].properties.visible.set_value(checkbox.active);

    if (checkbox.active){{
        // set bounds indicator color to plot line color
        const line_index = index + 1 ; // the objective line is the first line
        
        // need to handle this here since it has fill color, not line_color
//  if (lines[index].glyph.type == "VArea"){{

        const plot_line_color = lines[line_index].glyph.properties.line_color.get_value().value;
    
    
        // TODO problem is here!
        bound_violation_indicator_source.data[variable_name] = [plot_line_color];
        bound_violation_indicator_source.change.emit();
        //lower_bound_violation_indicators[index_cons].glyph.properties.fill_color=plot_line_color;
        //upper_bound_violation_indicators[index_cons].glyph.properties.fill_color=plot_line_color;
        //lower_bound_violation_indicators[index_cons].glyph.properties.fill_color.set_value(plot_line_color);
        //upper_bound_violation_indicators[index_cons].glyph.properties.fill_color.set_value(plot_line_color);

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
        self._bounds_off_on_checkboxes = []  # includes only the checkboxes for bounds on/off
        
        
        
        
        self._column_items = []  # includes all items in the Column, including headers and toggles
        self._axes = []
        # flag to prevent updating label with units each time we get new data
        self._labels_updated_with_units = False

        # user for showing which constraint points are out of bounds
        self._lower_bounds_cons_source = None
        self._upper_bounds_cons_source = None
        self._constraint_bounds = {}
        self._lower_bound_violation_indicators = []
        self._upper_bound_violation_indicators = []
        self._num_desvars = 0
        self._up_arrow_image_path = "./images/up_arrow_small.png"
        self._down_arrow_image_path = "./images/down_arrow_small.png"

        self._upper_bounds_region_source = None

        self._setup_figure()

        # used to keep track of the y min and max of the data so that
        # the axes ranges can be adjusted as data comes in
        self._y_min = defaultdict(lambda: float("inf"))
        self._y_max = defaultdict(lambda: float("-inf"))

        doc.add_periodic_callback(self._update_wrapped_in_try, callback_period)
        doc.title = "OpenMDAO Optimization Progress Plot"

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
                # self._lower_bounds_cons_source.stream(
                #     self._lower_bounds_cons_source_stream_dict
                # )
                # self._upper_bounds_cons_source.stream(
                #     self._upper_bounds_cons_source_stream_dict
                # )

                self._upper_bounds_region_source.stream(self._upper_bounds_region_source_stream_dict)
            return

        new_data = self._case_tracker._get_data_from_case(new_case)

        # See if Bokeh source object is defined yet. If not, set it up
        # since now we have data from the case recorder with info about the
        # variables to be plotted.
        if self._source is None:
            self._setup_data_source()

            # Check to make sure we have one and only one objective before going farther
            obj_names = self._case_tracker._get_obj_names()
            if len(obj_names) != 1:
                raise ValueError(
                    f"Plot requires there to be one and only one objective \
                        but {len(obj_names)} objectives found"
                )

            # Create CustomJS callback for toggle buttons.
            # Pass in the data from the Python side that the JavaScript side
            #   needs

            bound_violation_indicator_source_dict = {}
            cons_names = self._case_tracker._get_cons_names()
            for i, cons_name in enumerate(cons_names):
                units = self._case_tracker._get_units(cons_name)
                con_name_with_type = cons_name
                if cons_name in self._both_desvars_and_cons:
                    con_name_with_type += " [cons]"

                cons_button_label = f"{con_name_with_type} ({units})"
                bound_violation_indicator_source_dict[cons_button_label] = ["black"]

            self._bound_violation_indicator_source = ColumnDataSource(
                    data=bound_violation_indicator_source_dict
                )

            import copy

            gleep = self._bounds_off_on_checkboxes.copy() # TODO figure out why I need to do this
            
            gleep = copy.deepcopy(self._bounds_off_on_checkboxes)
            
            print("before customjs")
            print(f"{self._toggles=}")
            print(f"{self._bounds_off_on_checkboxes=}")

            bounds_off_on_callback = CustomJS(
                args=dict(
                    lines=self._lines,
                    lower_bound_violation_indicators=self._lower_bound_violation_indicators,
                    upper_bound_violation_indicators=self._upper_bound_violation_indicators,
                    num_desvars=self._num_desvars,
                    # bounds_off_on_checkboxes=self._bounds_off_on_checkboxes,
                    # bounds_off_on_checkboxes=gleep,
                    plot=self.plot_figure,
                    bound_violation_indicator_source = self._bound_violation_indicator_source,
                ),
                code=bounds_off_on_callback_code,
            )
            
            legend_item_callback = CustomJS(
                args=dict(
                    lines=self._lines,
                    axes=self._axes,
                    lower_bound_violation_indicators=self._lower_bound_violation_indicators,
                    upper_bound_violation_indicators=self._upper_bound_violation_indicators,
                    num_desvars=self._num_desvars,
                    # toggles=self._toggles,

                    bounds_off_on_checkboxes=self._bounds_off_on_checkboxes,
                    # bounds_off_on_checkboxes=gleep,
                    colorPalette=_colorPalette,
                    plot=self.plot_figure,
                    bound_violation_indicator_source = self._bound_violation_indicator_source,
                ),
                code=callback_code,
            )



            # For the variables, make lines, axes, and the buttons to turn on and
            #   off the variable plot.
            # All the lines and axes for the desvars and cons are created in
            #   Python but initially are not visible. They are turned on and
            #   off on the JavaScript side.

            # objs
            obj_label = _make_header_text_for_variable_chooser("OBJECTIVE")
            self._column_items.append(obj_label)

            for i, obj_name in enumerate(obj_names):
                units = self._case_tracker._get_units(obj_name)
                self.plot_figure.yaxis.axis_label = f"{obj_name} ({units})"
                self._make_variable_button(
                    f"{obj_name} ({units})", 'objs', _obj_color, True, legend_item_callback,
                    bounds_off_on_callback
                )
                self._make_line_and_hover_tool(
                    "objs", obj_name, False, _obj_color, "solid", True
                )
                value = new_data["objs"][obj_name]
                float_value = _get_value_for_plotting(value, "objs")
                # just give it some non-zero initial range since we only have one point
                self.plot_figure.y_range = Range1d(float_value - 1, float_value + 1)

            # desvars
            desvars_label = _make_header_text_for_variable_chooser("DESIGN VARS")
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
                    'desvars',  # TODO should make this and the other types a const
                    _non_active_plot_color,
                    False,
                    legend_item_callback,
                    bounds_off_on_callback
                )
                value = new_data["desvars"][desvar_name]
                # for desvars, if value is a vector, use Bokeh Varea glyph
                use_varea = value.size > 1
                self._make_line_and_hover_tool(
                    "desvars",
                    # desvar_name,
                    desvar_name_with_type,
                    use_varea,
                    _non_active_plot_color,
                    "solid",
                    False,
                )
                float_value = _get_value_for_plotting(value, "desvars")
                # self._make_axis("desvars", desvar_name, float_value, units)
                self._make_axis("desvars", desvar_name_with_type, float_value, units)

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
                    _non_active_plot_color,
                    False,
                    legend_item_callback,
                    bounds_off_on_callback
                )
                self._make_line_and_hover_tool(
                    "cons",
                    # cons_name,
                    con_name_with_type,
                    False,
                    _non_active_plot_color,
                    "dashed",
                    False,
                )
                value = new_data["cons"][cons_name]
                float_value = _get_value_for_plotting(value, "cons")
                self._make_axis("cons", con_name_with_type, float_value, units)
                self._constraint_bounds[cons_name] = (
                    self._case_tracker._get_constraint_bounds(cons_name)
                )

            # for toggle in self._toggles:
            #     toggle.js_on_change("active", legend_item_callback)

            # for checkbox in self._bounds_off_on_checkboxes:
            #     checkbox.js_on_change("active", bounds_off_on_callback)



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
            # end of self._source is None - plotting is setup

        # Do the actual update of the plot including updating the plot range and adding the new
        # data to the Bokeh plot stream
        counter = new_data["counter"]

        self._source_stream_dict = {"iteration": [counter]}

        self.plot_figure.x_range = Range1d(1, counter)

        # need separate sources to be able to plot the icons indicating the cons
        # are out of bounds
        self._lower_bounds_cons_source_stream_dict = {
            "iteration": [counter],
            "urls": [self._up_arrow_image_path],
        }
        self._upper_bounds_cons_source_stream_dict = {
            "iteration": [counter],
            "urls": [self._down_arrow_image_path],
        }

        self._upper_bounds_region_source_stream_dict = {
            "left": [1],
            "right": [counter],
            "top": [1e6],
            "bottom": [0.0]  # TODO need to have a separate entry for each cons upper bound
        }

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
                desvar_name_with_type += ' [dv]'
            if not self._labels_updated_with_units and desvar_value.size > 1:
                units = self._case_tracker._get_units(desvar_name)
                desvar_button_label = desvar_name_with_type + f' ({units}) {desvar_value.shape}'
                # if desvar_name in self._both_desvars_and_cons:
                #     varname = f"{desvar_name} [dv] ({units}) {desvar_value.shape}"
                # else:
                #     varname = f"{desvar_name} ({units}) {desvar_value.shape}"
                # self._toggles[iline].label = varname
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
                range = Range1d(self._y_min[desvar_name], self._y_max[desvar_name])
                # self.plot_figure.extra_y_ranges[f"extra_y_{desvar_name}_min"] = range
                self.plot_figure.extra_y_ranges[f"extra_y_{desvar_name_with_type}_min"] = range
            # deal with when min and max are the same.
            # Otherwise the varea plot shows nothing, not even a line
            if np.min(desvar_value) == np.max(desvar_value):
                range = self._y_max[desvar_name] - self._y_min[desvar_name]
                # 0.001 seems to be a good fraction, not too thick but still visible
                min_thickness = range * .001
                y1 = np.min(desvar_value) - min_thickness
                y2 = np.min(desvar_value) + min_thickness
            else:
                y1 = np.min(desvar_value)
                y2 = np.max(desvar_value)

            # self._source_stream_dict[f"{desvar_name}_min"] = [y1]
            # self._source_stream_dict[f"{desvar_name}_max"] = [y2]
            self._source_stream_dict[f"{desvar_name_with_type}_min"] = [y1]
            self._source_stream_dict[f"{desvar_name_with_type}_max"] = [y2]
            iline += 1

        for cons_name, cons_value in new_data["cons"].items():
            float_cons_value = _get_value_for_plotting(cons_value, "cons")
            con_name_with_type = cons_name
            if cons_name in self._both_desvars_and_cons:
                con_name_with_type += ' [cons]'
            if not self._labels_updated_with_units and cons_value.size > 1:
                units = self._case_tracker._get_units(cons_name)
                cons_button_label = con_name_with_type + f' ({units}) {cons_value.shape}'
                # if cons_name in self._both_desvars_and_cons:
                #     varname = f"{cons_name} [cons] ({units}) {cons_value.shape}"
                # else:
                #     varname = f"{cons_name} ({units}) {cons_value.shape}"
                self._toggles[iline].label = cons_button_label
                # self._toggles[iline].label = varname
            # self._source_stream_dict[cons_name] = [float_cons_value]
            self._source_stream_dict[con_name_with_type] = [float_cons_value]

            lower_bound, upper_bound = self._constraint_bounds[cons_name]
            # if given np.nan, nothing will be plotted. Only plot arrows when out of bounds
            # lower_value = float_cons_value if float_cons_value < lower_bound else np.nan
            # upper_value = float_cons_value if float_cons_value > upper_bound else np.nan
            # self._lower_bounds_cons_source_stream_dict[con_name_with_type] = [lower_value]
            # self._upper_bounds_cons_source_stream_dict[con_name_with_type] = [upper_value]
            # self._lower_bounds_cons_source_stream_dict[cons_name] = [lower_value]
            # self._upper_bounds_cons_source_stream_dict[cons_name] = [upper_value]

            # handle non-scalar cons
            min_max_changed = False
            min_max_changed = min_max_changed or _update_y_min_max(
                con_name_with_type, np.min(cons_value), self._y_min, self._y_max
            )
            min_max_changed = min_max_changed or _update_y_min_max(
                con_name_with_type, np.max(cons_value), self._y_min, self._y_max
            )
            if min_max_changed:
                range = Range1d(self._y_min[con_name_with_type], self._y_max[con_name_with_type])
                # self.plot_figure.extra_y_ranges[f"extra_y_{cons_name}"] = range
                self.plot_figure.extra_y_ranges[f"extra_y_{con_name_with_type}"] = range
            iline += 1

        self._source.stream(self._source_stream_dict)
        # self._lower_bounds_cons_source.stream(self._lower_bounds_cons_source_stream_dict)
        # self._upper_bounds_cons_source.stream(self._upper_bounds_cons_source_stream_dict)

        self._upper_bounds_region_source.stream(self._upper_bounds_region_source_stream_dict)

        self._labels_updated_with_units = True
        # end of _update method

    def _setup_data_source(self):
        self._source_dict = {"iteration": []}

        obj_names = self._case_tracker._get_obj_names()
        desvar_names = self._case_tracker._get_desvar_names()
        con_names = self._case_tracker._get_cons_names()
        self._both_desvars_and_cons = list(set(desvar_names).intersection(set(con_names)))

        # Obj
        for obj_name in obj_names:
            self._source_dict[obj_name] = []

        # Desvars
        for desvar_name in desvar_names:
            desvar_name_with_type = desvar_name
            if desvar_name in self._both_desvars_and_cons:
                desvar_name_with_type += ' [dv]'
            self._source_dict[f"{desvar_name_with_type}_min"] = []
            self._source_dict[f"{desvar_name_with_type}_max"] = []
            # self._source_dict[f"{desvar_name}_min"] = []
            # self._source_dict[f"{desvar_name}_max"] = []
            self._num_desvars += 1

        # Cons

        for con_name in con_names:
            con_name_with_type = con_name
            if con_name in self._both_desvars_and_cons:
                con_name_with_type += ' [cons]'
            # self._source_dict[con_name] = []
            self._source_dict[con_name_with_type] = []
        self._source = ColumnDataSource(self._source_dict)

        # Cons - lower bound
        self._lower_bounds_cons_source_dict = {
            "iteration": [],
            "urls": [],
        }
        for con_name in con_names:
            con_name_with_type = con_name
            if con_name in self._both_desvars_and_cons:
                con_name_with_type += ' [cons]'
            self._lower_bounds_cons_source_dict[con_name_with_type] = []
            # self._lower_bounds_cons_source_dict[con_name] = []
        self._lower_bounds_cons_source = ColumnDataSource(self._lower_bounds_cons_source_dict)

        # Cons - upper bound
        self._upper_bounds_cons_source_dict = {
            "iteration": [],
            "urls": [],
        }
        for con_name in con_names:
            con_name_with_type = con_name
            if con_name in self._both_desvars_and_cons:
                con_name_with_type += ' [cons]'
            self._upper_bounds_cons_source_dict[con_name_with_type] = []
            # self._upper_bounds_cons_source_dict[con_name] = []
        self._upper_bounds_cons_source = ColumnDataSource(self._upper_bounds_cons_source_dict)

        self._upper_bounds_region_source_stream_dict = {
            "left": [],
            "right": [],
            "top": [],
            "bottom": [],
        }
        self._upper_bounds_region_source = ColumnDataSource(self._upper_bounds_region_source_stream_dict)

    def _make_variable_button(self, varname, var_type, color, active, callback, bounds_off_on_callback):
        index = len(self._toggles)
        toggle = Toggle(
            label=varname,
            active=active,
            margin=(0, 0, 8, 0),
            tags=[{'var_type': var_type, 'index': index}])
        toggle.js_on_change("active", callback)
        self._toggles.append(toggle)

        if var_type != 'objs':
            index = len(self._bounds_off_on_checkboxes)
            checkbox = Checkbox(active=False, disabled=True, margin=(12, 0, 8, 4),
                                tags=[{'var_type': var_type, 'index': index, 'variable_name':varname}])
            # TODO consistently use varname, cons,...
            # can be accessed on the javascript side using checkbox.tags[0].name
            checkbox.js_on_change("active", bounds_off_on_callback)
            self._bounds_off_on_checkboxes.append(checkbox)
            self._column_items.append(Row(toggle,checkbox))
        else:
            self._column_items.append(toggle)
            
        # self._column_items.append(toggle)

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
            if var_type == "cons":

                import re
                # Match optional whitespace followed by [anything] at the end of string
                varname_minus_type = re.sub(r'\s*\[.*?\]$', '', varname)

                lower_bound, upper_bound = self._case_tracker._get_constraint_bounds(varname_minus_type)

                # varname includes if cons, but not units so need to add it here
                units = self._case_tracker._get_units(varname_minus_type)
                cons_button_label = f"{varname} ({units})"

                upper_bound_violation_indicator = self.plot_figure.quad(
                    top=_bounds_infinity,
                    bottom=upper_bound,
                    left=_bounds_left,
                    right=_bounds_infinity,
                    source=self._bound_violation_indicator_source,
                    hatch_pattern=_bounds_hatch_pattern,  # Diagonal hatch pattern
                    hatch_color=cons_button_label,  # Use color from data source
                    hatch_alpha=_bounds_hatch_alpha,  # Hatch opacity
                    hatch_weight=_bounds_hatch_weight,  # Hatch line thickness
                    line_color=cons_button_label,  # Quad outline
                    line_width=_bounds_line_width,
                    alpha=_bounds_alpha,
                    color=cons_button_label,
                    visible=False,
                )

                lower_bound_violation_indicator = self.plot_figure.quad(
                    top=lower_bound,
                    bottom=-_bounds_infinity,
                    left=_bounds_left,
                    right=_bounds_infinity,
                    source=self._bound_violation_indicator_source,
                    hatch_pattern=_bounds_hatch_pattern,  # Diagonal hatch pattern
                    hatch_color=cons_button_label,  # Use color from data source
                    hatch_alpha=_bounds_hatch_alpha,  # Hatch opacity
                    hatch_weight=_bounds_hatch_weight,  # Hatch line thickness
                    line_color=cons_button_label,  # Quad outline
                    line_width=_bounds_line_width,
                    alpha=_bounds_alpha,
                    color=cons_button_label,
                    visible=False,
                )

                self._lower_bound_violation_indicators.append(lower_bound_violation_indicator)
                self._upper_bound_violation_indicators.append(upper_bound_violation_indicator)

        if var_type == "desvars":
            line.y_range_name = f"extra_y_{varname}_min"
        elif var_type == "cons":
            line.y_range_name = f"extra_y_{varname}"

            lower_bound_violation_indicator.y_range_name = f"extra_y_{varname}"
            upper_bound_violation_indicator.y_range_name = f"extra_y_{varname}"

        self._lines.append(line)
        if not use_varea:  # hover tool does not work with Varea
            hover = HoverTool(
                renderers=[line],
                tooltips=[
                    ("Iteration", "@iteration"),
                    (varname, "@{%s}" % varname + "{0.00}"),
                ],
                mode="vline",
                visible=visible,
            )
            self.plot_figure.add_tools(hover)

    def _make_axis(self, var_type, varname, plot_value, units):
        # Make axis for this variable on the right of the plot

        # print(f"make axis for {varname=}")
        if var_type == "desvars":
            y_range_name = f"extra_y_{varname}_min"
        else:
            y_range_name = f"extra_y_{varname}"
        extra_y_axis = LinearAxis(
            y_range_name=y_range_name,
            axis_label=f"{varname} ({units})",
            axis_label_text_font_size="20px",
            visible=False,
            # visible=True,
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
            title = "Optimization Progress Plot for: " \
                f"{self._case_tracker.get_case_recorder_filename()}"
        self.plot_figure = figure(
            tools=[
                PanTool(),
                WheelZoomTool(dimensions='width'),
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
            active_drag=None,
            active_scroll="auto",
            active_tap=None,
            output_backend="webgl",
        )
        # self.plot_figure.x_range.start = 1
        # self.plot_figure.x_range.follow = "start"

        self.plot_figure.title.text_font_size = "14px"
        self.plot_figure.title.text_color = "black"
        self.plot_figure.title.text_font = "arial"
        self.plot_figure.title.align = "left"
        self.plot_figure.title.standoff = 40  # Adds 40 pixels of space below the title

        self.plot_figure.xaxis.axis_label = "Driver iterations"
        self.plot_figure.xaxis.minor_tick_line_color = None
        self.plot_figure.xaxis.ticker = BasicTicker(desired_num_ticks=10, min_interval=1)

        self.plot_figure.axis.axis_label_text_font_style = "bold"
        self.plot_figure.axis.axis_label_text_font_size = "20pt"
