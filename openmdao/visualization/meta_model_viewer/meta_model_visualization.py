"""Define output of Meta Models and visualize the results."""

import warnings
from collections import OrderedDict
import math
from itertools import product

from scipy.spatial import cKDTree
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.plotting import figure
from bokeh.models import Slider, ColumnDataSource
from bokeh.models import ColorBar, BasicTicker, LinearColorMapper, Range1d
from bokeh.models.widgets import TextInput, Select
from bokeh.models.ranges import DataRange1d
from bokeh.server.server import Server

from openmdao.components.meta_model_unstructured_comp import MetaModelUnStructuredComp
from openmdao.components.meta_model_structured_comp import MetaModelStructuredComp
from openmdao.core.problem import Problem


def stack_outputs(outputs_dict):
    """
    Stack the values of a dictionary.

    Parameters
    ----------
    outputs_dict : dict
        Dictionary of outputs

    Returns
    -------
    array
        np.stack of values
    """
    output_lists_to_stack = []
    for values in outputs_dict.values():
        output_lists_to_stack.append(np.asarray(values))

    return np.stack(output_lists_to_stack, axis=-1)


class MetaModelVisualization(object):
    """
    Top-level container for the Meta Model Visualization.

    Attributes
    ----------
    prob : Problem
        Name of variable corresponding to Problem Component
    surrogate_ref : MetaModel
        Name of Meta Model Component object reference
    model_ref : MetaModel
        Name of empty Meta Model Component object reference
    resolution : int
        Number used to calculate width and height of contour plot
    is_structured_meta_model : Bool
        Boolean used to signal whether the meta model is structured or unstructured
    slider_source : ColumnDataSource
        Data source containing dictionary of sliders
    contour_training_data_source : ColumnDataSource
        Data source containing dictionary of training data points
    bot_plot_source : ColumnDataSource
        Data source containing data for the bottom subplot
    bot_plot_scatter_source : ColumnDataSource
        Data source containing scatter point data for the bottom subplot
    right_plot_source : ColumnDataSource
        Data source containing data for the right subplot
    right_plot_scatter_source : ColumnDataSource
        Data source containing scatter point data for the right subplot
    source : ColumnDataSource
        Data source containing data for the contour plot
    input_list : list
        List of input data titles as strings
    output_list : list
        List of output data titles as strings
    input_data : dict
        Dictionary of input training data
    x_input : Select
        Bokeh Select object containing a list of inputs for the x axis
    y_input : Select
        Bokeh Select object containing a list of inputs for the y axis
    output_select : Select
        Bokeh Select object containing a list of inputs for the outputs
    x_input_slider : Slider
        Bokeh Slider object containing a list of input values for the x axis
    y_input_slider : Slider
        Bokeh Slider object containing a list of input values for the y axis
    slider_dict : dict
        Dictionary of slider names and their respective slider objects
    input_data_dict : OrderedDict
        Dictionary containing training data points to predict at.
    num_of_inputs : int
        Number of inputs
    num_of_outputs : int
        Number of outputs
    scatter_distance : TextInput
        Text input for user to enter custom value to calculate distance of training points around
        slice line
    dist_range : float
        Value taken from scatter_distance used for calculating distance of training points around
        slice line
    x_index : int
        Value of x axis column
    y_index : int
        Value of y axis column
    output_variable : int
        Value of output axis column
    sliders_and_selects : layout
        Layout containing the sliders and select elements
    layout : layout
        Contains first row of plots
    layout2 : layout
        Contains second row of plots
    z : array
        A 2D array containing contour plot data
    """

    def __init__(self, surrogate_ref, resolution=50, doc=None):
        """
        Initialize parameters.

        Parameters
        ----------
        surrogate_ref : MetaModelComponent
            Reference to meta model component
        resolution : int
            Value used to calculate the size of contour plot meshgrid
        doc : Document
            The bokeh document to build.
        """
        self.prob = Problem()
        self.surrogate_ref = surrogate_ref
        self.resolution = resolution

        # Create list of inputs
        if isinstance(self.surrogate_ref, MetaModelUnStructuredComp):
            self.is_structured_meta_model = False

            # self.resolution = resolution
            self.input_list = [name[0] for name in self.surrogate_ref._surrogate_input_names]

            if len(self.input_list) < 2:
                raise ValueError('Must have more than one input value')

            self.output_list = [name[0] for name in self.surrogate_ref._surrogate_output_names]

            self.model_ref = MetaModelUnStructuredComp(
                default_surrogate=self.surrogate_ref.options['default_surrogate'])

        elif isinstance(self.surrogate_ref, MetaModelStructuredComp):
            self.is_structured_meta_model = True

            self.input_list = [name for name in self.surrogate_ref._var_rel_names['input']]

            if len(self.input_list) < 2:
                raise ValueError('Must have more than one input value')

            self.output_list = [name for name in self.surrogate_ref._var_rel_names['output']]

            self.model_ref = MetaModelStructuredComp(
                distributed=self.surrogate_ref.options['distributed'],
                extrapolate=self.surrogate_ref.options['extrapolate'],
                method=self.surrogate_ref.options['method'],
                training_data_gradients=self.surrogate_ref.options['training_data_gradients'],
                vec_size=1)

        # Pair input list names with their respective data
        self.input_data = {}

        self._setup_empty_prob_comp()

        # Setup dropdown menus for x/y inputs and the output value
        self.x_input = Select(title="X Input:", value=[x for x in self.input_list][0],
                              options=[x for x in self.input_list])
        self.x_input.on_change('value', self._x_input_update)

        self.y_input = Select(title="Y Input:", value=[x for x in self.input_list][1],
                              options=[x for x in self.input_list])
        self.y_input.on_change('value', self._y_input_update)

        self.output_select = Select(title="Output:", value=[x for x in self.output_list][0],
                                    options=[x for x in self.output_list])
        self.output_select.on_change('value', self._output_value_update)

        # Create sliders in a loop
        self.slider_dict = {}
        self.input_data_dict = OrderedDict()
        for title, values in self.input_data.items():
            slider_data = np.linspace(min(values), max(values), self.resolution)
            self.input_data_dict[title] = slider_data
            # Calculates the distance between slider ticks
            slider_step = slider_data[1] - slider_data[0]
            slider_object = Slider(start=min(values), end=max(values), value=min(values),
                                   step=slider_step, title=str(title))
            self.slider_dict[title] = slider_object

        self._slider_attrs()

        # Length of inputs and outputs
        self.num_of_inputs = len(self.input_list)
        self.num_of_outputs = len(self.output_list)

        # Precalculate the problem bounds.
        bounds = [[min(i), max(i)] for i in self.input_data.values()]
        limits = np.array(bounds)
        self.limit_range = limits[:, 1] - limits[:, 0]

        # Positional indicies
        self.x_index = 0
        self.y_index = 1
        self.output_variable = self.output_list.index(self.output_select.value)

        # Most data sources are filled with initial values
        self.slider_source = ColumnDataSource(data=self.input_data_dict)
        self.source = ColumnDataSource(data=dict(
            z=np.random.rand(self.resolution, self.resolution)))
        self.contour_training_data_source = ColumnDataSource(
            data=dict(x=np.repeat(0, self.resolution), y=np.repeat(0, self.resolution)))

        self.bot_plot_source = ColumnDataSource(data=dict(
            x=np.repeat(0, self.resolution), y=np.repeat(0, self.resolution)))
        self.bot_plot_scatter_source = ColumnDataSource(data=dict(
            bot_slice_x=np.repeat(0, self.resolution), bot_slice_y=np.repeat(0, self.resolution)))

        self.right_plot_source = ColumnDataSource(data=dict(
            x=np.repeat(0, self.resolution), y=np.repeat(0, self.resolution)))
        self.right_plot_scatter_source = ColumnDataSource(data=dict(
            left_slice_x=np.repeat(0, self.resolution), left_slice_y=np.repeat(0, self.resolution)))

        # Text input to change the distance of reach when searching for nearest data points
        self.scatter_distance = TextInput(value="0.1", title="Scatter Distance")
        self.scatter_distance.on_change('value', self._scatter_input)
        self.dist_range = float(self.scatter_distance.value)

        # Grouping all of the sliders and dropdowns into one column
        sliders = [i for i in self.slider_dict.values()]
        sliders.extend([self.x_input, self.y_input, self.output_select, self.scatter_distance])
        self.sliders_and_selects = row(
            column(*sliders))

        # Layout creation
        self.layout = row(self._contour_data(), self._right_plot(), self.sliders_and_selects)
        self.layout2 = row(self._bot_plot())

        if doc is None:
            doc = curdoc()

        doc.add_root(self.layout)
        doc.add_root(self.layout2)
        doc.title = 'Meta Model Visualization'

    def _setup_empty_prob_comp(self):
        """
        Take data from surrogate ref and pass it into new surrogate model with empty Problem model.

        Parameters
        ----------
        None

        """
        if self.is_structured_meta_model:
            for idx, name in enumerate(self.input_list):
                try:
                    self.input_data[name] = self.surrogate_ref.params[idx]
                    self.model_ref.add_input(
                        name, 0.,
                        training_data=self.surrogate_ref.params[idx])
                except TypeError:
                    msg = "No training data present for one or more parameters"
                    raise TypeError(msg)

            for idx, name in enumerate(self.output_list):
                self.model_ref.add_output(
                    name, 0.,
                    training_data=self.surrogate_ref.training_outputs[name])

        else:
            for name in self.input_list:
                try:
                    self.input_data[name] = {
                        i for i in self.surrogate_ref.options['train:' + str(name)]}
                    self.model_ref.add_input(
                        name, 0.,
                        training_data=[i for i in self.surrogate_ref.options['train:' + str(name)]])
                except TypeError:
                    msg = "No training data present for one or more parameters"
                    raise TypeError(msg)

            for name in self.output_list:
                self.model_ref.add_output(
                    name, 0.,
                    training_data=[i for i in self.surrogate_ref.options['train:' + str(name)]])


        self.prob.model.add_subsystem('interp', self.model_ref)
        self.prob.setup()

    def _slider_attrs(self):
        """
        Assign slider objects and callback functions.

        Parameters
        ----------
        None

        """
        for name, slider_object in self.slider_dict.items():
            # Checks if there is a callback previously assigned and then clears it
            if len(slider_object._callbacks) == 1:
                slider_object._callbacks.clear()
            if name == self.x_input.value:
                self.x_input_slider = slider_object
                self.x_input_slider.on_change('value', self._scatter_plots_update)
            elif name == self.y_input.value:
                self.y_input_slider = slider_object
                self.y_input_slider.on_change('value', self._scatter_plots_update)
            else:
                slider_object.on_change('value', self._update)

    def _make_predictions(self, data):
        """
        Run the data parameter through the surrogate model which is given in prob.

        Parameters
        ----------
        data : dict
            Dictionary containing Ordered Dict of training points.

        Returns
        -------
        array
            np.stack of predicted points.
        """
        outputs = {i: [] for i in self.output_list}
        print("Making Predictions")

        # Parse dict into shape [n**2, number of inputs] list
        inputs = np.empty([self.resolution**2, self.num_of_inputs])
        for idx, values in enumerate(data.values()):
            inputs[:, idx] = values.flatten()

        if self.is_structured_meta_model:
            for idx, tup in enumerate(inputs):
                for name, val in zip(data.keys(), tup):
                    self.prob[str.format(self.model_ref.name + '.' + name)] = val
                self.prob.run_model()
                for i in self.output_list:
                    outputs[i].append(
                        np.array(self.prob[str.format(self.model_ref.name + '.' + i)]))
        else:
            # Pair data points with their respective prob name. Loop to make predictions
            for idx, tup in enumerate(inputs):
                for name, val in zip(data.keys(), tup):
                    self.prob[str.format(self.model_ref.name + '.' + name)] = val
                self.prob.run_model()
                for i in self.output_list:
                    outputs[i].append(float(self.prob[str.format(self.model_ref.name + '.' + i)]))

        return stack_outputs(outputs)

    def _cont_data_calcs(self):
        resolution = self.resolution
        x_data = np.zeros((resolution, resolution, self.num_of_inputs))

        self._slider_attrs()

        self.input_point_list = [i.value for i in self.slider_dict.values()]
        x_data[:, :, :] = np.array(self.input_point_list)

        for idx, (title, values) in enumerate(self.slider_source.data.items()):
            if title == self.x_input.value:
                self.xlins_mesh = values
                x_index_position = idx
            if title == self.y_input.value:
                self.ylins_mesh = values
                y_index_position = idx

        X, Y = np.meshgrid(self.xlins_mesh, self.ylins_mesh)
        x_data[:, :, x_index_position] = X
        x_data[:, :, y_index_position] = Y

        pred_dict = {}
        for idx, title in enumerate(self.slider_source.data):
            pred_dict.update({title: x_data[:, :, idx]})

        return pred_dict

    def _contour_data(self):
        """
        Create a contour plot.

        Parameters
        ----------
        None

        Returns
        -------
        Bokeh Image Plot
        """
        resolution = self.resolution
        y_data = np.zeros((resolution, resolution, self.num_of_outputs))

        # Pass the dict to make predictions and then reshape the output to (n, n, number of outputs)
        y_data[:, :, :] = self._make_predictions(self._cont_data_calcs()).reshape(
            (resolution, resolution, self.num_of_outputs))
        self.Z = y_data[:, :, self.output_variable]
        self.Z = self.Z.reshape(resolution, resolution)

        self.source.data = dict(z=[self.Z])

        self.contour_x_range = xlins = self.xlins_mesh
        self.contour_y_range = ylins = self.ylins_mesh

        # Color bar formatting
        color_mapper = LinearColorMapper(
            palette="Viridis11", low=np.amin(self.Z), high=np.amax(self.Z))
        color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(), label_standoff=12,
                             location=(0, 0))

        # Contour Plot
        self.contour_plot = contour_plot = figure(
            match_aspect=False,
            tooltips=[(self.x_input.value, "$x"), (self.y_input.value, "$y"),
                      (self.output_select.value, "@z")], tools='')
        contour_plot.x_range.range_padding = 0
        contour_plot.y_range.range_padding = 0
        contour_plot.plot_width = 600
        contour_plot.plot_height = 500
        contour_plot.xaxis.axis_label = self.x_input.value
        contour_plot.yaxis.axis_label = self.y_input.value
        contour_plot.min_border_left = 0
        contour_plot.add_layout(color_bar, 'right')
        contour_plot.x_range = Range1d(min(xlins), max(xlins))
        contour_plot.y_range = Range1d(min(ylins), max(ylins))
        contour_plot.image(image='z', source=self.source, x=min(xlins), y=min(ylins),
                           dh=(max(ylins) - min(ylins)), dw=(max(xlins) - min(xlins)),
                           palette="Viridis11")

        # Adding training data points overlay to contour plot
        if self.is_structured_meta_model:
            data = self._structured_training_points()
        else:
            data = self._unstructured_training_points()

        if len(data):
            data = np.array(data)
            self.contour_training_data_source.data = dict(x=data[:, 0], y=data[:, 1])
            self.contour_plot.circle(
                x='x', y='y', source=self.contour_training_data_source,
                size=5, color='white', alpha=0.50)

        return self.contour_plot

    def _right_plot(self):
        """
        Create the right side subplot to view the projected slice.

        Parameters
        ----------
        None

        Returns
        -------
        Bokeh figure
        """
        # Sets data for x/y inputs

        y_data = self.input_data_dict[self.y_input.value]
        x_value = self.x_input_slider.value
        # Rounds the x_data to match the input_data_dict value
        subplot_value_index = np.where(
            np.around(self.input_data_dict[self.x_input.value], 5) == np.around(x_value, 5))[0]

        # Make slice in Z data at the point calculated before and add it to the data source
        z_data = self.Z[:, subplot_value_index].flatten()

        x = z_data
        y = self.slider_source.data[self.y_input.value]

        self.right_plot_source.data = dict(x=x, y=y)

        # Create and format figure
        self.right_plot_fig = right_plot_fig = figure(
            plot_width=250, plot_height=500,
            x_range=(min(x), max(x)),
            y_range=(min(self.contour_y_range), max(self.contour_y_range)),
            title="{} vs {}".format(self.y_input.value, self.output_select.value), tools="pan")
        right_plot_fig.xaxis.axis_label = self.output_select.value
        right_plot_fig.yaxis.axis_label = self.y_input.value
        right_plot_fig.xaxis.major_label_orientation = math.pi / 9
        right_plot_fig.line(x='x', y='y', source=self.right_plot_source)

        # Determine distance and alpha opacity of training points
        if self.is_structured_meta_model:
            data = self._structured_training_points(compute_distance=True)
        else:
            data = self._unstructured_training_points(compute_distance=True)

        alphas = np.zeros((len(data), ))
        for i, info in enumerate(data):
            alpha = np.abs(info[0] - x_value) / self.limit_range[self.x_index]
            if alpha < self.dist_range:
                alphas[i] = 1
                # (1 - alpha / self.dist_range) * info[-1]

        right_plot_fig.scatter(x=data[:, 3], y=data[:, 1], line_color=None, fill_color='#000000',
                               fill_alpha=alphas)

        # Set the right_plot data source to new values
        self.right_plot_scatter_source.data = dict(
            left_slice_x=np.repeat(x_value, self.resolution), left_slice_y=y_data)

        self.contour_plot.line(
            'left_slice_x', 'left_slice_y', source=self.right_plot_scatter_source,
            color='black', line_width=2)

        return self.right_plot_fig

    def _bot_plot(self):
        """
        Create the bottom subplot to view the projected slice.

        Parameters
        ----------
        None

        Returns
        -------
        Bokeh figure
        """
        x_data = self.input_data_dict[self.x_input.value]
        y_value = self.y_input_slider.value
        subplot_value_index = np.where(
            np.around(self.input_data_dict[self.y_input.value], 5) == np.around(y_value, 5))[0]

        z_data = self.Z[subplot_value_index, :].flatten()

        x = self.slider_source.data[self.x_input.value]
        y = z_data
        self.bot_plot_source.data = dict(x=x, y=y)

        self.bot_plot_fig = bot_plot_fig = figure(
            plot_width=550, plot_height=250,
            x_range=(min(self.contour_x_range), max(self.contour_x_range)),
            y_range=(min(y), max(y)),
            title="{} vs {}".format(self.x_input.value, self.output_select.value), tools="")
        bot_plot_fig.xaxis.axis_label = self.x_input.value
        bot_plot_fig.yaxis.axis_label = self.output_select.value
        bot_plot_fig.line(x='x', y='y', source=self.bot_plot_source)

        if self.is_structured_meta_model:
            data = self._structured_training_points(compute_distance=True)
        else:
            data = self._unstructured_training_points(compute_distance=True)

        alphas = np.zeros((len(data), ))
        for i, info in enumerate(data):
            alpha = np.abs(info[1] - y_value) / self.limit_range[self.y_index]
            if alpha < self.dist_range:
                alphas[i] = 1
                # (1 - alpha / self.dist_range) * info[-1]

        bot_plot_fig.scatter(x=data[:, 0], y=data[:, 3], line_color=None, fill_color='#000000',
                             fill_alpha=alphas)

        self.bot_plot_scatter_source.data = dict(
            bot_slice_x=x_data,
            bot_slice_y=np.repeat(y_value, self.resolution))

        self.contour_plot.line(
            'bot_slice_x', 'bot_slice_y', source=self.bot_plot_scatter_source, color='black',
            line_width=2)

        return self.bot_plot_fig

    def _update_all_plots(self):
        self.layout.children[0] = self._contour_data()
        self.layout.children[1] = self._right_plot()
        self.layout2.children[0] = self._bot_plot()

    def _update_subplots(self):
        self.layout.children[1] = self._right_plot()
        self.layout2.children[0] = self._bot_plot()

    # Event handler functions
    def _update(self, attr, old, new):
        # self._contour_data()
        # self._right_plot()
        # self._bot_plot()
        self._update_all_plots()

    def _scatter_plots_update(self, attr, old, new):
        # self._right_plot()
        # self._bot_plot()
        self._update_subplots()

    def _scatter_input(self, attr, old, new):
        self.dist_range = float(new)
        self._update_all_plots()

    def _input_dropdown_checks(self, x, y):
        # Checks to see if x and y inputs are equal to each other
        if x == y:
            return False
        else:
            return True

    def _x_input_update(self, attr, old, new):
        if not self._input_dropdown_checks(new, self.y_input.value):
            raise ValueError("Inputs should not equal each other")
        else:
            self.x_input.value = new
            self._update_all_plots()

    def _y_input_update(self, attr, old, new):
        if not self._input_dropdown_checks(self.x_input.value, new):
            raise ValueError("Inputs should not equal each other")
        else:
            self.y_input.value = new
            self._update_all_plots()

    def _output_value_update(self, attr, old, new):
        self.output_variable = self.output_list.index(new)
        self._update_all_plots()

    def _unstructured_training_points(self, compute_distance=False):
        """
        Calculate the training points and returns and array containing the position and alpha.

        Parameters
        ----------
        None

        Returns
        -------
        array
            The array of training points and their alpha opacity with respect to the surrogate line
        """
        # x_training contains
        # [x1, x2, x3, x4]
        # Input Data
        # Output Data
        x_training = self.model_ref._training_input
        y_training = np.squeeze(stack_outputs(self.model_ref._training_output), axis=1)

        x_index = self.x_input.options.index(self.x_input.value)
        y_index = self.y_input.options.index(self.y_input.value)
        output_variable = self.output_list.index(self.output_select.value)

        # Vertically stack the x/y inputs and then transpose them
        infos = np.vstack((x_training[:, x_index], x_training[:, y_index])).transpose()
        if not compute_distance:
            return infos

        points = x_training.copy()
        # Set the first two columns of the points array to x/y inputs, respectively
        points[:, x_index] = self.input_point_list[x_index]
        points[:, y_index] = self.input_point_list[y_index]
        points = np.divide(points, self.limit_range)
        tree = cKDTree(points)
        dist_limit = np.linalg.norm(self.dist_range * self.limit_range)
        scaled_x0 = np.divide(self.input_point_list, self.limit_range)
        # Query the nearest neighbors tree for the closest points to the scaled x0 array
        dists, idx = tree.query(scaled_x0, k=len(x_training), distance_upper_bound=dist_limit)

        # info contains:
        # [x_value, y_value, ND-distance, func_value, alpha]

        data = np.zeros((len(idx), 5))
        for dist_index, i in enumerate(idx):
            info = np.ones((5))
            info[0:2] = infos[i, :]
            info[2] = dists[dist_index] / dist_limit
            info[3] = y_training[i, output_variable]
            info[4] = (1. - info[2] / self.dist_range) ** 0.5
            data[dist_index] = info

        return data

    def _structured_training_points(self, compute_distance=False):
        # reate tuple of the input parameters
        input_dimensions = tuple(self.surrogate_ref.params)

        # Input training data and output training data
        self.x_training = np.array([z for z in product(*input_dimensions)])
        self.y_training = self.surrogate_ref.training_outputs[self.output_select.value].flatten()

        # Index of input/output variables
        x_index = self.x_input.options.index(self.x_input.value)
        y_index = self.y_input.options.index(self.y_input.value)

        # Vertically stack the x/y inputs and then transpose them
        infos = np.vstack((self.x_training[:, x_index], self.x_training[:, y_index])).transpose()
        if not compute_distance:
            return infos

        points = self.x_training.copy()
        # Set the first two columns of the points array to x/y inputs, respectively
        points[:, x_index] = self.input_point_list[x_index]
        points[:, y_index] = self.input_point_list[y_index]
        points = np.divide(points, self.limit_range)
        self.dist_limit = np.linalg.norm(self.dist_range * self.limit_range)
        scaled_x0 = np.divide(self.input_point_list, self.limit_range)
        # Query the nearest neighbors tree for the closest points to the scaled x0 array
        # Nearest points to x slice

        if self.x_training.shape[1] < 3:
            x_tree, y_tree = self._two_dimension_input(scaled_x0, points)
        elif self.x_training.shape[1] > 2:
            x_tree, y_tree = self._multidimension_input(scaled_x0, points)

        # [x_value, y_value, ND-distance_X, func_value, x_alpha, ND-distance_Y, y_alpha]

        n = len(x_tree)
        data = np.zeros((n, 7))
        for i in range(n):
            info = np.ones((7))
            try:
                info[0:2] = infos[i, :]
            except IndexError:
                print("ERROR: Scatter distance value too low. Try: 0.1")
            info[2] = x_tree[i] / self.dist_limit
            info[3] = self.y_training[i]
            info[4] = (1. - info[2] / self.dist_range) ** 0.5
            info[5] = y_tree[i] / self.dist_limit
            info[6] = (1. - info[5] / self.dist_range) ** 0.5
            data[i] = info

        return data

    def _two_dimension_input(self, scaled_points, training_points):
        # array, index, axis
        x = np.delete(scaled_points, 0, 0)
        y = np.delete(scaled_points, 1, 0)

        x_tree = [abs(x - number) for number in training_points[:, 0]]
        y_tree = [abs(y - number) for number in training_points[:, 1]]

        return [x_tree, y_tree]

    def _multidimension_input(self, scaled_points, training_points):
        x = np.delete(scaled_points, self.x_input.options.index(self.x_input.value), 0)
        x_tree_training_points = np.delete(training_points, 0, axis=1)
        y = np.delete(scaled_points, self.y_input.options.index(self.y_input.value), 0)
        y_tree_training_points = np.delete(training_points, 1, axis=1)

        x_tree = cKDTree(x_tree_training_points)
        y_tree = cKDTree(y_tree_training_points)
        # Query the nearest neighbors tree for the closest points to the scaled x0 array

        x_dists, x_idx = x_tree.query(
            x, k=len(self.x_training), distance_upper_bound=self.dist_limit)
        y_dists, y_idx = y_tree.query(
            y, k=len(self.y_training), distance_upper_bound=self.dist_limit)

        # kdtree query always returns requested k even if there are not enough valid points
        idx_finite = np.where(np.isfinite(x_dists))
        x_dists = x_dists[idx_finite]
        idx_finite = np.where(np.isfinite(y_dists))
        y_dists = y_dists[idx_finite]

        return [x_dists, y_dists]


def view_metamodel(meta_model_comp, port_number):
    """
    Visualize a metamodel.

    Parameters
    ----------
    meta_model_comp : MetaModelStructuredComp or MetaModelUnStructuredComp
        The metamodel component.
    port_number : int
        Bokeh plot port number.
    """
    from bokeh.application.application import Application
    from bokeh.application.handlers import FunctionHandler

    def make_doc(doc):
        MetaModelVisualization(meta_model_comp, doc=doc)

    # print('Opening Bokeh application on http://localhost:5006/')
    server = Server({'/': Application(FunctionHandler(make_doc))}, port=int(port_number))
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
