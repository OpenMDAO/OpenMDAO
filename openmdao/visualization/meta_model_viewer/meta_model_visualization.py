"""Define output of Meta Models and visualize the results."""

import math
from itertools import product

from scipy.spatial import cKDTree
import numpy as np
import logging

from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.plotting import figure
from bokeh.models import Slider, ColumnDataSource, HoverTool
from bokeh.models import ColorBar, BasicTicker, LinearColorMapper, Range1d
from bokeh.models.widgets import TextInput, Select
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
    return np.stack([np.asarray(v) for v in outputs_dict.values()], axis=-1)


class MetaModelVisualization(object):
    """
    Top-level container for the Meta Model Visualization.

    Attributes
    ----------
    prob : Problem
        Name of variable corresponding to Problem Component
    meta_model : MetaModel
        Name of empty Meta Model Component object reference
    resolution : int
        Number used to calculate width and height of contour plot
    is_structured_meta_model : bool
        Boolean used to signal whether the meta model is structured or unstructured
    slider_source : ColumnDataSource
        Data source containing dictionary of sliders
    contour_training_data_source : ColumnDataSource
        Data source containing dictionary of training data points
    bottom_plot_source : ColumnDataSource
        Data source containing data for the bottom subplot
    bottom_plot_scatter_source : ColumnDataSource
        Data source containing scatter point data for the bottom subplot
    right_plot_source : ColumnDataSource
        Data source containing data for the right subplot
    right_plot_scatter_source : ColumnDataSource
        Data source containing scatter point data for the right subplot
    contour_plot_source : ColumnDataSource
        Data source containing data for the contour plot
    input_names : list
        List of input data titles as strings
    output_names : list
        List of output data titles as strings
    training_inputs : dict
        Dictionary of input training data
    x_input_select : Select
        Bokeh Select object containing a list of inputs for the x axis
    y_input_select : Select
        Bokeh Select object containing a list of inputs for the y axis
    output_select : Select
        Bokeh Select object containing a list of inputs for the outputs
    x_input_slider : Slider
        Bokeh Slider object containing a list of input values for the x axis
    y_input_slider : Slider
        Bokeh Slider object containing a list of input values for the y axis
    slider_dict : dict
        Dictionary of slider names and their respective slider objects
    predict_inputs : dict
        Dictionary containing training data points to predict at.
    num_inputs : int
        Number of inputs
    num_outputs : int
        Number of outputs
    limit_range : array
        Array containing the range of each input
    scatter_distance : TextInput
        Text input for user to enter custom value to calculate distance of training points around
        slice line
    right_alphas : array
        Array of points containing alpha values for right plot
    bottom_alphas : array
        Array of points containing alpha values for bottom plot
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
    doc_layout : layout
        Contains first row of plots
    doc_layout2 : layout
        Contains second row of plots
    Z : array
        A 2D array containing contour plot data
    """

    def __init__(self, model, resolution=50, doc=None):
        """
        Initialize parameters.

        Parameters
        ----------
        model : MetaModelComponent
            Reference to meta model component
        resolution : int
            Value used to calculate the size of contour plot meshgrid
        doc : Document
            The bokeh document to build.
        """
        self.prob = Problem()
        self.resolution = resolution
        logging.getLogger("bokeh").setLevel(logging.ERROR)

        # If the surrogate model coming in is structured
        if isinstance(model, MetaModelUnStructuredComp):
            self.is_structured_meta_model = False

            # Create list of input names, check if it has more than one input, then create list
            # of outputs
            self.input_names = [name[0] for name in model._surrogate_input_names]
            if len(self.input_names) < 2:
                raise ValueError('Must have more than one input value')
            self.output_names = [name[0] for name in model._surrogate_output_names]

            # Create reference for untructured component
            self.meta_model = MetaModelUnStructuredComp(
                default_surrogate=model.options['default_surrogate'])

        # If the surrogate model coming in is unstructured
        elif isinstance(model, MetaModelStructuredComp):
            self.is_structured_meta_model = True

            self.input_names = [name for name in model._var_rel_names['input']]

            if len(self.input_names) < 2:
                raise ValueError('Must have more than one input value')

            self.output_names = [name for name in model._var_rel_names['output']]

            self.meta_model = MetaModelStructuredComp(
                distributed=model.options['distributed'],
                extrapolate=model.options['extrapolate'],
                method=model.options['method'],
                training_data_gradients=model.options['training_data_gradients'],
                vec_size=1)

        # Pair input list names with their respective data
        self.training_inputs = {}

        self._setup_empty_prob_comp(model)

        # Setup dropdown menus for x/y inputs and the output value
        self.x_input_select = Select(title="X Input:", value=[x for x in self.input_names][0],
                                     options=[x for x in self.input_names])
        self.x_input_select.on_change('value', self._x_input_update)

        self.y_input_select = Select(title="Y Input:", value=[x for x in self.input_names][1],
                                     options=[x for x in self.input_names])
        self.y_input_select.on_change('value', self._y_input_update)

        self.output_select = Select(title="Output:", value=[x for x in self.output_names][0],
                                    options=[x for x in self.output_names])
        self.output_select.on_change('value', self._output_value_update)

        # Create sliders for each input
        self.slider_dict = {}
        self.predict_inputs = {}
        for title, values in self.training_inputs.items():
            slider_data = np.linspace(min(values), max(values), self.resolution)
            self.predict_inputs[title] = slider_data
            # Calculates the distance between slider ticks
            slider_step = slider_data[1] - slider_data[0]
            slider_object = Slider(start=min(values), end=max(values), value=min(values),
                                   step=slider_step, title=str(title))
            self.slider_dict[title] = slider_object

        self._slider_attrs()

        # Length of inputs and outputs
        self.num_inputs = len(self.input_names)
        self.num_outputs = len(self.output_names)

        # Precalculate the problem bounds.
        limits = np.array([[min(value), max(value)] for value in self.training_inputs.values()])
        self.limit_range = limits[:, 1] - limits[:, 0]

        # Positional indicies
        self.x_index = 0
        self.y_index = 1
        self.output_variable = self.output_names.index(self.output_select.value)

        # Data sources are filled with initial values
        # Slider Column Data Source
        self.slider_source = ColumnDataSource(data=self.predict_inputs)

        # Contour plot Column Data Source
        self.contour_plot_source = ColumnDataSource(data=dict(
            z=np.random.rand(self.resolution, self.resolution)))
        self.contour_training_data_source = ColumnDataSource(
            data=dict(x=np.repeat(0, self.resolution), y=np.repeat(0, self.resolution)))

        # Bottom plot Column Data Source
        self.bottom_plot_source = ColumnDataSource(data=dict(
            x=np.repeat(0, self.resolution), y=np.repeat(0, self.resolution)))
        self.bottom_plot_scatter_source = ColumnDataSource(data=dict(
            bot_slice_x=np.repeat(0, self.resolution), bot_slice_y=np.repeat(0, self.resolution)))

        # Right plot Column Data Source
        self.right_plot_source = ColumnDataSource(data=dict(
            x=np.repeat(0, self.resolution), y=np.repeat(0, self.resolution)))
        self.right_plot_scatter_source = ColumnDataSource(data=dict(
            right_slice_x=np.repeat(0, self.resolution),
            right_slice_y=np.repeat(0, self.resolution)))

        # Text input to change the distance of reach when searching for nearest data points
        self.scatter_distance = TextInput(value="0.1", title="Scatter Distance")
        self.scatter_distance.on_change('value', self._scatter_input)
        self.dist_range = float(self.scatter_distance.value)

        # Grouping all of the sliders and dropdowns into one column
        sliders = [value for value in self.slider_dict.values()]
        sliders.extend(
            [self.x_input_select, self.y_input_select, self.output_select, self.scatter_distance])
        self.sliders_and_selects = row(
            column(*sliders))

        # Layout creation
        self.doc_layout = row(self._contour_data(), self._right_plot(), self.sliders_and_selects)
        self.doc_layout2 = row(self._bottom_plot())

        if doc is None:
            doc = curdoc()

        doc.add_root(self.doc_layout)
        doc.add_root(self.doc_layout2)
        doc.title = 'Meta Model Visualization'

    def _setup_empty_prob_comp(self, metamodel):
        """
        Take data from surrogate ref and pass it into new surrogate model with empty Problem model.

        Parameters
        ----------
        metamodel : MetaModelComponent
            Reference to meta model component

        """
        # Check for structured or unstructured
        if self.is_structured_meta_model:
            # Loop through the input names
            for idx, name in enumerate(self.input_names):
                # Check for no training data
                try:
                    # Append the input data/titles to a dictionary
                    self.training_inputs[name] = metamodel.inputs[idx]
                    # Also, append the data as an 'add_input' to the model reference
                    self.meta_model.add_input(name, 0.,
                                              training_data=metamodel.inputs[idx])
                except TypeError:
                    msg = "No training data present for one or more parameters"
                    raise TypeError(msg)

            # Add the outputs to the model reference
            for idx, name in enumerate(self.output_names):
                self.meta_model.add_output(
                    name, 0.,
                    training_data=metamodel.training_outputs[name])

        else:
            for name in self.input_names:
                try:
                    self.training_inputs[name] = {
                        title for title in metamodel.options['train:' + str(name)]}
                    self.meta_model.add_input(
                        name, 0.,
                        training_data=[
                            title for title in metamodel.options['train:' + str(name)]])
                except TypeError:
                    msg = "No training data present for one or more parameters"
                    raise TypeError(msg)

            for name in self.output_names:
                self.meta_model.add_output(
                    name, 0.,
                    training_data=[
                        title for title in metamodel.options['train:' + str(name)]])

        # Add the subsystem and setup
        self.prob.model.add_subsystem('interp', self.meta_model)
        self.prob.setup()

    def _slider_attrs(self):
        """
        Assign data to slider objects and callback functions.

        Parameters
        ----------
        None

        """
        for name, slider_object in self.slider_dict.items():
            # Checks if there is a callback previously assigned and then clears it
            if len(slider_object._callbacks) == 1:
                slider_object._callbacks.clear()

            # Check if the name matches the 'x input' title
            if name == self.x_input_select.value:
                # Set the object and add an event handler
                self.x_input_slider = slider_object
                self.x_input_slider.on_change('value', self._scatter_plots_update)

            # Check if the name matches the 'y input' title
            elif name == self.y_input_select.value:
                # Set the object and add an event handler
                self.y_input_slider = slider_object
                self.y_input_slider.on_change('value', self._scatter_plots_update)
            else:
                # If it is not an x or y input then just assign it the event handler
                slider_object.on_change('value', self._update)

    def _make_predictions(self, data):
        """
        Run the data parameter through the surrogate model which is given in prob.

        Parameters
        ----------
        data : dict
            Dictionary containing training points.

        Returns
        -------
        array
            np.stack of predicted points.
        """
        # Create dictionary with an empty list
        outputs = {name: [] for name in self.output_names}

        # Parse dict into shape [n**2, number of inputs] list
        inputs = np.empty([self.resolution**2, self.num_inputs])
        for idx, values in enumerate(data.values()):
            inputs[:, idx] = values.flatten()

        # Check for structured or unstructured
        if self.is_structured_meta_model:
            # Assign each row of the data coming in to a tuple. Loop through the tuple, and append
            # the name of the input and value.
            for idx, tup in enumerate(inputs):
                for name, val in zip(data.keys(), tup):
                    self.prob[self.meta_model.name + '.' + name] = val
                self.prob.run_model()
                # Append the predicted value(s)
                for title in self.output_names:
                    outputs[title].append(
                        np.array(self.prob[self.meta_model.name + '.' + title]))

        else:
            for idx, tup in enumerate(inputs):
                for name, val in zip(data.keys(), tup):
                    self.prob[self.meta_model.name + '.' + name] = val
                self.prob.run_model()
                for title in self.output_names:
                    outputs[title].append(
                        float(self.prob[self.meta_model.name + '.' + title]))

        return stack_outputs(outputs)

    def _contour_data_calcs(self):
        """
        Parse input data into a dictionary to be predicted at.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dictionary of training data to be predicted at.
        """
        # Create initial data array of training points
        resolution = self.resolution
        x_data = np.zeros((resolution, resolution, self.num_inputs))

        self._slider_attrs()

        # Broadcast the inputs to every row of x_data array
        x_data[:, :, :] = np.array(self.input_point_list)

        # Find the x/y input titles and match their index positions
        for idx, (title, values) in enumerate(self.slider_source.data.items()):
            if title == self.x_input_select.value:
                self.xlins_mesh = values
                x_index_position = idx
            if title == self.y_input_select.value:
                self.ylins_mesh = values
                y_index_position = idx

        # Make meshgrid from the x/y inputs to be plotted
        X, Y = np.meshgrid(self.xlins_mesh, self.ylins_mesh)
        # Move the x/y inputs to their respective positions in x_data
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
        # Output data array initialization
        y_data = np.zeros((resolution, resolution, self.num_outputs))
        self.input_point_list = [point.value for point in self.slider_dict.values()]

        # Pass the dict to make predictions and then reshape the output to
        # (resolution, resolution, number of outputs)
        y_data[:, :, :] = self._make_predictions(self._contour_data_calcs()).reshape(
            (resolution, resolution, self.num_outputs))
        # Use the output variable to pull the correct column of data from the predicted
        # data (y_data)
        self.Z = y_data[:, :, self.output_variable]
        # Reshape it to be 2D
        self.Z = self.Z.reshape(resolution, resolution)

        # Update the data source with new data
        self.contour_plot_source.data = dict(z=[self.Z])

        # Min to max of training data
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
            tooltips=[(self.x_input_select.value, "$x"), (self.y_input_select.value, "$y"),
                      (self.output_select.value, "@z")], tools='')
        contour_plot.x_range.range_padding = 0
        contour_plot.y_range.range_padding = 0
        contour_plot.plot_width = 600
        contour_plot.plot_height = 500
        contour_plot.xaxis.axis_label = self.x_input_select.value
        contour_plot.yaxis.axis_label = self.y_input_select.value
        contour_plot.min_border_left = 0
        contour_plot.add_layout(color_bar, 'right')
        contour_plot.x_range = Range1d(min(xlins), max(xlins))
        contour_plot.y_range = Range1d(min(ylins), max(ylins))
        contour_plot.image(image='z', source=self.contour_plot_source, x=min(xlins), y=min(ylins),
                           dh=(max(ylins) - min(ylins)), dw=(max(xlins) - min(xlins)),
                           palette="Viridis11")

        # Adding training data points overlay to contour plot
        if self.is_structured_meta_model:
            data = self._structured_training_points()
        else:
            data = self._unstructured_training_points()

        if len(data):
            # Add training data points overlay to contour plot
            data = np.array(data)
            if self.is_structured_meta_model:
                self.contour_training_data_source.data = dict(x=data[:, 0], y=data[:, 1],
                                                              z=self.meta_model.training_outputs[
                                                              self.output_select.value].flatten())
            else:
                self.contour_training_data_source.data = dict(x=data[:, 0], y=data[:, 1],
                                                              z=self.meta_model._training_output[
                                                              self.output_select.value])

            training_data_renderer = self.contour_plot.circle(
                x='x', y='y', source=self.contour_training_data_source,
                size=5, color='white', alpha=0.50)

            self.contour_plot.add_tools(HoverTool(renderers=[training_data_renderer], tooltips=[
                (self.x_input_select.value + " (train)", '@x'),
                (self.y_input_select.value + " (train)", '@y'),
                (self.output_select.value + " (train)", '@z'), ]))

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
        # List of the current positions of the sliders
        self.input_point_list = [point.value for point in self.slider_dict.values()]

        # Find the title of the y input and match it with the data
        y_idx = self.y_input_select.value
        y_data = self.predict_inputs[y_idx]
        # Find the position of the x_input slider
        x_value = self.x_input_slider.value

        # Rounds the x_data to match the predict_inputs value
        subplot_value_index = np.where(
            np.around(self.predict_inputs[self.x_input_select.value], 5) ==
            np.around(x_value, 5))[0]

        # Make slice in Z data at the point calculated before and add it to the data source
        z_data = self.Z[:, subplot_value_index].flatten()

        x = z_data
        y = self.slider_source.data[y_idx]

        # Update the data source with new data
        self.right_plot_source.data = dict(x=x, y=y)

        # Create and format figure
        self.right_plot_fig = right_plot_fig = figure(
            plot_width=250, plot_height=500,
            title="{} vs {}".format(y_idx, self.output_select.value), tools="pan")
        right_plot_fig.xaxis.axis_label = self.output_select.value
        right_plot_fig.yaxis.axis_label = y_idx
        right_plot_fig.xaxis.major_label_orientation = math.pi / 9
        right_plot_fig.line(x='x', y='y', source=self.right_plot_source)
        right_plot_fig.x_range.range_padding = 0.1
        right_plot_fig.y_range.range_padding = 0.02

        # Determine distance and alpha opacity of training points
        if self.is_structured_meta_model:
            data = self._structured_training_points(compute_distance=True, source='right')
        else:
            data = self._unstructured_training_points(compute_distance=True, source='right')

        self.right_alphas = 1.0 - data[:, 2] / self.dist_range

        # Training data scatter plot
        scatter_renderer = right_plot_fig.scatter(x=data[:, 3], y=data[:, 1], line_color=None,
                                                  fill_color='#000000',
                                                  fill_alpha=self.right_alphas.tolist())

        right_plot_fig.add_tools(HoverTool(renderers=[scatter_renderer], tooltips=[
            (self.output_select.value + " (train)", '@x'),
            (y_idx + " (train)", '@y'),
        ]))
        right_plot_fig.scatter(x=data[:, 3], y=data[:, 1], line_color=None, fill_color='#000000',
                               fill_alpha=self.right_alphas.tolist())

        span_width = self.dist_range * (max(y_data) - min(y_data))

        # Set the right_plot data source to new values
        self.right_plot_scatter_source.data = dict(
            right_slice_x=np.repeat(x_value, self.resolution), right_slice_y=y_data,
            left_dashed=[i - span_width for i in np.repeat(x_value, self.resolution)],
            right_dashed=[i + span_width for i in np.repeat(x_value, self.resolution)])

        self.contour_plot.line(
            'right_slice_x', 'right_slice_y', source=self.right_plot_scatter_source,
            color='black', line_width=2)
        self.contour_plot.line(
            'left_dashed', 'right_slice_y', line_dash='dashed',
            source=self.right_plot_scatter_source, color='black', line_width=2)
        self.contour_plot.line(
            'right_dashed', 'right_slice_y', line_dash='dashed',
            source=self.right_plot_scatter_source, color='black', line_width=2)

        return self.right_plot_fig

    def _bottom_plot(self):
        """
        Create the bottom subplot to view the projected slice.

        Parameters
        ----------
        None

        Returns
        -------
        Bokeh figure
        """
        # List of the current positions of the sliders
        self.input_point_list = [point.value for point in self.slider_dict.values()]

        # Find the title of the x input and match it with the data
        x_idx = self.x_input_select.value
        x_data = self.predict_inputs[x_idx]
        # Find the position of the y_input slider
        y_value = self.y_input_slider.value

        # Rounds the y_data to match the predict_inputs value
        subplot_value_index = np.where(
            np.around(self.predict_inputs[self.y_input_select.value], 5) ==
            np.around(y_value, 5))[0]

        # Make slice in Z data at the point calculated before and add it to the data source
        z_data = self.Z[subplot_value_index, :].flatten()

        x = self.slider_source.data[x_idx]
        y = z_data

        # Update the data source with new data
        self.bottom_plot_source.data = dict(x=x, y=y)

        # Create and format figure
        self.bottom_plot_fig = bottom_plot_fig = figure(
            plot_width=550, plot_height=250,
            title="{} vs {}".format(x_idx, self.output_select.value), tools="")
        bottom_plot_fig.xaxis.axis_label = x_idx
        bottom_plot_fig.yaxis.axis_label = self.output_select.value
        bottom_plot_fig.line(x='x', y='y', source=self.bottom_plot_source)
        bottom_plot_fig.x_range.range_padding = 0.02
        bottom_plot_fig.y_range.range_padding = 0.1

        # Determine distance and alpha opacity of training points
        if self.is_structured_meta_model:
            data = self._structured_training_points(compute_distance=True)
        else:
            data = self._unstructured_training_points(compute_distance=True)

        self.bottom_alphas = 1.0 - data[:, 2] / self.dist_range

        # Training data scatter plot
        scatter_renderer = bottom_plot_fig.scatter(x=data[:, 0], y=data[:, 3], line_color=None,
                                                   fill_color='#000000',
                                                   fill_alpha=self.bottom_alphas.tolist())

        bottom_plot_fig.add_tools(HoverTool(renderers=[scatter_renderer], tooltips=[
            (x_idx + " (train)", '@x'),
            (self.output_select.value + " (train)", '@y'),
        ]))

        span_width = self.dist_range * (max(x_data) - min(x_data))

        # Set the right_plot data source to new values
        self.bottom_plot_scatter_source.data = dict(
            bot_slice_x=x_data, bot_slice_y=np.repeat(y_value, self.resolution),
            upper_dashed=[i + span_width for i in np.repeat(y_value, self.resolution)],
            lower_dashed=[i - span_width for i in np.repeat(y_value, self.resolution)])

        self.contour_plot.line(
            'bot_slice_x', 'bot_slice_y', source=self.bottom_plot_scatter_source, color='black',
            line_width=2)
        self.contour_plot.line(
            'bot_slice_x', 'upper_dashed', line_dash='dashed',
            source=self.bottom_plot_scatter_source, color='black', line_width=2)
        self.contour_plot.line(
            'bot_slice_x', 'lower_dashed', line_dash='dashed',
            source=self.bottom_plot_scatter_source, color='black', line_width=2)

        return self.bottom_plot_fig

    def _unstructured_training_points(self, compute_distance=False, source='bottom'):
        """
        Calculate the training points and returns and array containing the position and alpha.

        Parameters
        ----------
        compute_distance : bool
            If true, compute the distance of training points from surrogate line.
        source : str
            Which subplot the method is being called from.

        Returns
        -------
        array
            The array of training points and their alpha opacity with respect to the surrogate line
        """
        # Input training data and output training data
        x_training = self.meta_model._training_input
        training_output = np.squeeze(stack_outputs(self.meta_model._training_output), axis=1)

        # Index of input/output variables
        x_index = self.x_input_select.options.index(self.x_input_select.value)
        y_index = self.y_input_select.options.index(self.y_input_select.value)
        output_variable = self.output_names.index(self.output_select.value)

        # Vertically stack the x/y inputs and then transpose them
        infos = np.vstack((x_training[:, x_index], x_training[:, y_index])).transpose()
        if not compute_distance:
            return infos

        points = x_training.copy()

        # Normalize so each dimension spans [0, 1]
        points = np.divide(points, self.limit_range)
        dist_limit = np.linalg.norm(self.dist_range * self.limit_range)
        scaled_x0 = np.divide(self.input_point_list, self.limit_range)

        # Query the nearest neighbors tree for the closest points to the scaled x0 array
        # Nearest points to x slice
        if x_training.shape[1] < 3:

            tree = cKDTree(points)
            # Query the nearest neighbors tree for the closest points to the scaled x0 array
            dists, idxs = tree.query(
                scaled_x0, k=len(x_training), distance_upper_bound=self.dist_range)

            # kdtree query always returns requested k even if there are not enough valid points
            idx_finite = np.where(np.isfinite(dists))
            dists = dists[idx_finite]
            idxs = idxs[idx_finite]

        else:
            dists, idxs = self._multidimension_input(scaled_x0, points, source=source)

        # data contains:
        # [x_value, y_value, ND-distance, func_value]

        data = np.zeros((len(idxs), 4))
        for dist_index, j in enumerate(idxs):
            data[dist_index, 0:2] = infos[j, :]
            data[dist_index, 2] = dists[dist_index]
            data[dist_index, 3] = training_output[j, output_variable]

        return data

    def _structured_training_points(self, compute_distance=False, source='bottom'):
        """
        Calculate the training points and return an array containing the position and alpha.

        Parameters
        ----------
        compute_distance : bool
            If true, compute the distance of training points from surrogate line.
        source : str
            Which subplot the method is being called from.

        Returns
        -------
        array
            The array of training points and their alpha opacity with respect to the surrogate line
        """
        # Create tuple of the input parameters
        input_dimensions = tuple(self.meta_model.inputs)

        # Input training data and output training data
        x_training = np.array([z for z in product(*input_dimensions)])
        training_output = self.meta_model.training_outputs[self.output_select.value].flatten()

        # Index of input/output variables
        x_index = self.x_input_select.options.index(self.x_input_select.value)
        y_index = self.y_input_select.options.index(self.y_input_select.value)

        # Vertically stack the x/y inputs and then transpose them
        infos = np.vstack((x_training[:, x_index], x_training[:, y_index])).transpose()
        if not compute_distance:
            return infos

        points = x_training.copy()

        # Normalize so each dimension spans [0, 1]
        points = np.divide(points, self.limit_range)
        self.dist_limit = np.linalg.norm(self.dist_range * self.limit_range)
        scaled_x0 = np.divide(self.input_point_list, self.limit_range)
        # Query the nearest neighbors tree for the closest points to the scaled x0 array
        # Nearest points to x slice

        if x_training.shape[1] < 3:
            x_tree, x_idx = self._two_dimension_input(scaled_x0, points, source=source)
        else:
            x_tree, x_idx = self._multidimension_input(scaled_x0, points, source=source)

        # format for 'data'
        # [x_value, y_value, ND-distance_(x or y), func_value]

        n = len(x_tree)
        data = np.zeros((n, 4))
        for dist_index, j in enumerate(x_idx):
            data[dist_index, 0:2] = infos[j, :]
            data[dist_index, 2] = x_tree[dist_index]
            data[dist_index, 3] = training_output[j]

        return data

    def _two_dimension_input(self, scaled_points, training_points, source='bottom'):
        """
        Calculate the distance of training points to the surrogate line.

        Parameters
        ----------
        scaled_points : array
            Array of normalized slider positions.
        training_points : array
            Array of input training data.
        source : str
            Which subplot the method is being called from.

        Returns
        -------
        idxs : array
            Index of closest points that are within the dist range.
        x_tree : array
            One dimentional array of points that are within the dist range.
        """
        # Column of the input
        if source == 'right':
            col_idx = self.y_input_select.options.index(self.y_input_select.value)
        else:
            col_idx = self.x_input_select.options.index(self.x_input_select.value)

        # Delete the axis of input from source to predicted 1D distance
        x = np.delete(scaled_points, col_idx, axis=0)
        x_training_points = np.delete(training_points, col_idx, axis=1).flatten()

        # Tree of point distances
        x_tree = np.abs(x - x_training_points)

        # Only return points that are within our distance-viewing paramter.
        idx = np.where(x_tree <= self.dist_range)
        x_tree = x_tree[idx]
        return x_tree, idx[0]

    def _multidimension_input(self, scaled_points, training_points, source='bottom'):
        """
        Calculate the distance of training points to the surrogate line.

        Parameters
        ----------
        scaled_points : array
            Array of normalized slider positions.
        training_points : array
            Array of input training data.
        source : str
            Which subplot the method is being called from.

        Returns
        -------
        idxs : array
            Index of closest points that are within the dist range.
        x_tree : array
            Array of points that are within the dist range.
        """
        # Column of the input
        if source == 'right':
            col_idx = self.y_input_select.options.index(self.y_input_select.value)

        else:
            col_idx = self.x_input_select.options.index(self.x_input_select.value)

        # Delete the axis of input from source to predicted distance
        x = np.delete(scaled_points, col_idx, axis=0)
        x_training_points = np.delete(training_points, col_idx, axis=1)

        # Tree of point distances
        x_tree = cKDTree(x_training_points)

        # Query the nearest neighbors tree for the closest points to the scaled array
        dists, idx = x_tree.query(x, k=len(x_training_points),
                                  distance_upper_bound=self.dist_range)

        # kdtree query always returns requested k even if there are not enough valid points
        idx_finite = np.where(np.isfinite(dists))
        dists_finite = dists[idx_finite]
        idx = idx[idx_finite]
        return dists_finite, idx

    # Event handler functions
    def _update_all_plots(self):
        self.doc_layout.children[0] = self._contour_data()
        self.doc_layout.children[1] = self._right_plot()
        self.doc_layout2.children[0] = self._bottom_plot()

    def _update_subplots(self):
        self.doc_layout.children[1] = self._right_plot()
        self.doc_layout2.children[0] = self._bottom_plot()

    def _update(self, attr, old, new):
        self._update_all_plots()

    def _scatter_plots_update(self, attr, old, new):
        self._update_subplots()

    def _scatter_input(self, attr, old, new):
        # Text input update function of dist range value
        self.dist_range = float(new)
        self._update_all_plots()

    def _x_input_update(self, attr, old, new):
        # Checks that x and y inputs are not equal to each other
        if new == self.y_input_select.value:
            raise ValueError("Inputs should not equal each other")
        else:
            self.x_input_select.value = new
            self._update_all_plots()

    def _y_input_update(self, attr, old, new):
        # Checks that x and y inputs are not equal to each other
        if new == self.x_input_select.value:
            raise ValueError("Inputs should not equal each other")
        else:
            self.y_input_select.value = new
            self._update_all_plots()

    def _output_value_update(self, attr, old, new):
        self.output_variable = self.output_names.index(new)
        self._update_all_plots()


def view_metamodel(meta_model_comp, resolution, port_number, browser):
    """
    Visualize a metamodel.

    Parameters
    ----------
    meta_model_comp : MetaModelStructuredComp or MetaModelUnStructuredComp
        The metamodel component.
    resolution : int
        Number of points to control contour plot resolution.
    port_number : int
        Bokeh plot port number.
    browser : bool
        Boolean to show the browser
    """
    from bokeh.application.application import Application
    from bokeh.application.handlers import FunctionHandler

    def make_doc(doc):
        MetaModelVisualization(meta_model_comp, resolution, doc=doc)

    server = Server({'/': Application(FunctionHandler(make_doc))}, port=int(port_number))
    if browser:
        server.io_loop.add_callback(server.show, "/")
    else:
        print('Server started, to view go to http://localhost:{}/'.format(port_number))

    server.io_loop.start()
