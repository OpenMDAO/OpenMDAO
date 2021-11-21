import sys
import time

import numpy as np
from ipytree import Tree, Node
from ipywidgets import Label, HBox, VBox, Output, Button, Text

from openmdao.core.component import Component
from openmdao.core.group import Group
from openmdao.core.constants import _SetupStatus
from openmdao.core.indepvarcomp import IndepVarComp

# Because ipywidgets has a lot of backend<->frontend comm going on, for large
#  model hierarchies, a lot of widgets are created and you get buffer overruns. Need to slow it
#  down a bit. This seems like a good delay (in seconds) between creating a node in the tree
_DEFAULT_PAUSE_TIME = 0.02

# Using icons from https://fontawesome.com/v5.15/icons?d=gallery&p=2&m=free
_ICON_STYLE_VARIABLE_SELECTED = 'success'  # makes the icon green
_ICON_STYLE_VARIABLE_NOT_SELECTED = 'default'  # makes the icon grey
_ICON_COMPONENT = 'plug'
_ICON_GROUP = 'envelope-open'
_ICON_VARIABLE_NOT_SELECTED = 'eye-slash'
_ICON_VARIABLE_SELECTED = 'eye'

def set_values_gui(prob, vars_to_set=None, initial_depth = None, ivc_only = True,
                   pause_time=_DEFAULT_PAUSE_TIME, debug=False):
    """
    Create a Jupyter notebook-based GUI to let users set the values of input variables for models.

    Parameters
    ----------
    prob : Problem
        The Problem that will be used to populate the GUI.

    vars_to_set : None or list
        Used to set the initial set of input boxes visible. for If None, then no input boxes
        visible. If a list, a list of promoted

    initial_depth : None or int
        Used to set the initial set of depth of the hierarchy displayed. If None, the full
        hierarchy is displayed. This could be useful for very large hierarchies.

    ivc_only : bool
        If True, only inputs connected to IVCs are displayed in the hierarchy. If False,
        all inputs are displayed in the hierarchy

    pause_time : double
        Large hierarchies can cause the ipytree library to generate errors due to some messaging
        buffer overruns. To prevent that, a small pause is needed between ipytree widget
        creations. Usually the default should be fine, but if errors are being generated, the
        user can try setting this value to a value larger than the default

    debug : bool
        If True, a text box widget is displayed with some debugging info
    """

    ui = SetValuesUI(prob, vars_to_set=vars_to_set, initial_depth = initial_depth,
                     ivc_only = ivc_only, pause_time=pause_time, debug=debug)
    ui.setup()
    return ui.display()

class SetValuesUI(object):
    def __init__(self, prob, vars_to_set=None, initial_depth = None, ivc_only = True,
                 pause_time=_DEFAULT_PAUSE_TIME, debug= False):
        """
        Initialize attributes.
        """
        self._prob = prob
        self._vars_to_set = vars_to_set
        if initial_depth is None:
            self._initial_depth = sys.maxsize
        else:
            self._initial_depth = initial_depth
        self._ivc_only = ivc_only
        self._pause_time = pause_time
        self._debug = debug

        self._value_widget_box = None
        self._tree = None
        self.ui_widget = None
        self._refresh_in_progress = False

        self._inputs_connected_to_ivc = self._get_inputs_connected_to_ivc()
        self._abs2prom_inputs = self._prob.model._var_allprocs_abs2prom['input']

        # debugging
        self._output = Output(layout={'border': '1px solid black', 'width': '30%'})

    def setup(self):
        """
        Create the GUI widgets and define handlers.
        """
        if not (self._prob._metadata and self._prob._metadata['setup_status'] > _SetupStatus.PRE_SETUP):
            raise RuntimeError(f"The set values GUI cannot be created before Problem setup.")

        # make all the major Boxes and lay them out
        if self._debug:
            self._tree = Tree(stripes=True, multiple_selection=True, layout={'border': '1px solid black', 'width' :'30%'})
            self._value_widget_box = VBox([Label("Model Variables") ,], layout={'border': '1px solid black', 'width' :'30%'})
        else:  # more space for the tree and value widgets
            self._tree = Tree(stripes=True, multiple_selection=True, layout={'border': '1px solid black', 'width' :'40%'})
            self._value_widget_box = VBox([Label("Model Variables") ,], layout={'border': '1px solid black', 'width' :'50%'})

        refresh_button = Button(description="Refresh", layout={'border': '1px solid black',
                                                        'width' :'80px',
                                                        'display':'flex',
                                                        'justify_content':'center'})
        if self._debug:
            self._ui_widget = HBox([self._tree, self._value_widget_box, refresh_button,self._output])
        else:
            self._ui_widget = HBox([self._tree, self._value_widget_box, refresh_button])

        # populate the tree with model hierarchy and the variables
        self._set_vars_from_model_with_initial_list(self._prob.model, self._tree, 0,
                                                   vars_to_set=self._vars_to_set )

        # handlers
        self._tree.observe(self._on_selected_change, names='selected_nodes')
        refresh_button.on_click(self._refresh_ui_with_problem_data)

    def display(self):
        """
        Cause the GUI to be displayed in the notebook.
        """
        return self._ui_widget

    def _set_vars_from_model_with_initial_list(self, sys, node, depth, vars_to_set=None):
        """
        Populate the model tree selector widget. This method is called recursively

        Parameters
        ----------
        sys : an OpenMDAO System
            The System whose inputs will be added to the GUI hierarchy

        node : an ipytree Node object
            The node to which the variables will be added

        depth : int
            The depth in the hierarchy at which the vars are being added

        vars_to_set : None or list
            Used to set the initial set of input boxes visible. for If None, then no input boxes
            visible. If a list, a list of promoted
        """

        if sys.name == '_auto_ivc': #   _AutoIndepVarComp
            return

        # make the node for the System, either a Group or Component
        name = sys.name if sys.name else 'root'
        new_node = Node(name)
        node.add_node(new_node)

        # Don't expand this node if the user set a value for the initial displayed depth of
        #   the hierarchy and this method is processing a system below that.
        if depth > self._initial_depth - 1:
            new_node.opened = False

        # list of prom name at this level and abs name tuples
        var_names_at_this_level = self._get_var_names_at_this_level(sys)

        for input_varname, input_abs_name in var_names_at_this_level:
            # full_varname = f"{sys.pathname}.{input_varname}"   # sort of abs, sort of prom!
            prom_name = self._abs2prom_inputs[input_abs_name]

            # If user requested only inputs connected to IVCs, check to see if this var is one
            #   and skip if it is not
            if self._ivc_only and (prom_name not in self._inputs_connected_to_ivc):
                continue

            # Make the Node for the variable
            input_node = Node(input_varname)
            input_node.icon = _ICON_VARIABLE_NOT_SELECTED
            new_node.add_node(input_node)
            #  Need to know what component this var is associated. Used inside the handler for
            #  selecting this node
            input_node._sys = sys
            time.sleep(self._pause_time)  # To slow down the messages which cause issue with ipwidgets

            # make a Model Variables widget, depending on value of vars_to_set
            if vars_to_set is not None:
                if input_varname in vars_to_set:
                    self._add_value_widget(sys, input_varname, input_node)
            else:  # if vars_to_set is None, then add Model Variables widgets for all variables
                self._add_value_widget(sys, input_varname, input_node)

        # Set icon of System Node, and recurse if System is Group
        if isinstance(sys, Component):
            new_node.icon = _ICON_COMPONENT
        elif isinstance(sys, Group):  # must be Group
            new_node.icon = _ICON_GROUP
            for s in sys._subsystems_myproc:
                self._set_vars_from_model_with_initial_list(s, new_node, depth + 1, vars_to_set )

    def _get_inputs_connected_to_ivc(self):  # returns prom names
        """
        Get a list of all the model's inputs that are connected to IVCs.

        Returns
        -------
        list
            List of promoted names of variables that are connected to IVCs.
        """
        # Method used:
        #   1. loop over the top level var_allprocs_prom2abs_list['input'] entries
        #   2. do a lookup in the top level _conn_global_abs_in2out using the first entry of
        #       the 'abs list' from your var_allprocs_prom2abs_list lookup since you
        #       need an absolute input name to lookup the connected source
        #       from _conn_global_abs_in2out.
        #   3. determine if the parent of that source var is an IVC
        #   4. if yes, then put it in the list using the parent name of the promoted input var
        #   (the key from var_allprocs_prom2abs_list['input']).

        model = self._prob.model
        inputs_connected_to_ivc = []
        for prom, alist in model._var_allprocs_prom2abs_list['input'].items():
            var = alist[0]  # var is an abs name because that is what _conn_global_abs_in2out uses
            connected_source = model._conn_global_abs_in2out[var]
            compname = connected_source.rsplit('.', 1)[0]
            comp = model._get_subsystem(compname)
            # _AutoIndepVarComp is subclass of IndepVarComp so those caught too
            if isinstance(comp, IndepVarComp):
                inputs_connected_to_ivc.append(prom)

        return inputs_connected_to_ivc

    def _add_value_widget(self, sys, prom_name, tree_node):
        """
        Add a value widget to the Model Variables section of the GUI so the user
        can set the value and units of the variable

        Parameters
        ----------
        sys : an OpenMDAO System
            The System has this variable at its level

        prom_name : str
            The promoted name of the variable at the level of the System, sys

        tree_node : ipytree Node
            The node in the ipytree hierarchy that represents this variable
        """

        # check to see if the variable already has entry box
        prom2abs_list = sys._var_allprocs_prom2abs_list['input']
        abs_name = prom2abs_list[prom_name][0]
        promoted_name_from_top_level = self._abs2prom_inputs[abs_name]
        if promoted_name_from_top_level in self._get_widget_var_names():
            return # already there

        # get current value and units
        val = sys.get_val(prom_name)

        inputs_metadata = sys.get_io_metadata(('input',), ['units', ],
                                     get_remote=True, return_rel_names=False)

        metadata = inputs_metadata[abs_name]
        units = metadata['units']
        if not units:
            units = "None"

        if isinstance(val, np.ndarray):
            if val.size > 1:
                return # skip arrays for now TODO - need to raise warning or error
            val = val.item()

        # Create value widget
        val_widget = Text(
            value=str(val),
            description=promoted_name_from_top_level,
            disabled=False,
            continuous_update=False,
            style = {'description_width': 'initial'},
            step=None,
            layout = {'border': '1px solid black',
                      'width': '350px',
                      'display': 'flex',
                      'justify_content': 'center'}
        )
        val_widget.observe(self._update_prob_val, 'value')

        # create units widget
        units_widget = Text(units,
                            continuous_update=False,
                            description_tooltip=prom_name,
                            layout={'border': '1px solid black',  # TODO is center doing anything?
                                                        'width' :'60px',  # TODO dedup style stuff
                                                        'display':'flex',
                                                        'justify_content':'center'})
        units_widget._var_name = promoted_name_from_top_level # need to know which var it belongs to
        units_widget.observe(self._update_prob_unit, 'value')

        # create remove button widget
        remove_button = Button(description="X", layout={'border': '1px solid black',
                                                        'width' :'20px',
                                                        'display':'flex',
                                                        'justify_content':'center'})
        remove_button._var_name = promoted_name_from_top_level # which var it belongs to
        remove_button.on_click(self._remove_val_widget)

        # group all 3 widgets into an HBox
        val_and_units_remove_widget = HBox([val_widget, units_widget, remove_button])

        # Button needs to know which box to remove
        remove_button._val_and_units_remove_widget = val_and_units_remove_widget

        # need to know which node & var it is associated with so handlers can do all they need to do
        val_and_units_remove_widget._tree_node = tree_node
        val_and_units_remove_widget._var_name = promoted_name_from_top_level

        # set the display of the icon in the tree for this variable
        tree_node.icon = _ICON_VARIABLE_SELECTED
        tree_node.icon_style = _ICON_STYLE_VARIABLE_SELECTED

        # add to the list of value widgets
        self._value_widget_box.children += (val_and_units_remove_widget,)

    def _on_selected_change(self, change):
        """
        The handler for variable nodes in the hierarchy. This is called if the node is
        clicked on

        Parameters
        ----------
        change : dict
            A dictionary holding the information about the change

        """
        tree_node = change['new'][0]
        var_name = tree_node.name
        if var_name in self._get_widget_var_names():
            return

        self._add_value_widget(tree_node._sys, var_name, tree_node)

    def _remove_val_widget(self, button):
        """
        The handler for Button that lets users remove variable entry boxes from the Model
        Variable section of the GUI

        Parameters
        ----------
        button : ipywidget Button
            The remove Button that was clicked on

        """
        box_to_remove = button._val_and_units_remove_widget

        # # Cannot use remove on tuples, which children are so
        # if box_to_remove:
        #     # rebuild the children list minus the box to remove
        #     self._value_widget_box.children = tuple \
        #         (box for box in self._value_widget_box.children if box != box_to_remove)
        #     box_to_remove.close()
        #     associated_tree_node = box_to_remove._tree_node
        #     associated_tree_node.icon = _ICON_VARIABLE_NOT_SELECTED
        #     associated_tree_node.icon_style = _ICON_STYLE_VARIABLE_NOT_SELECTED

        # Cannot use remove on tuples, which children are so
        # rebuild the children list minus the box to remove
        self._value_widget_box.children = tuple \
            (box for box in self._value_widget_box.children if box != box_to_remove)
        box_to_remove.close()
        associated_tree_node = box_to_remove._tree_node
        associated_tree_node.icon = _ICON_VARIABLE_NOT_SELECTED
        associated_tree_node.icon_style = _ICON_STYLE_VARIABLE_NOT_SELECTED

    def _refresh_ui_with_problem_data(self, button):
        """
        The handler for Button that causes the values in the Model Variables section of the
        GUI to be updated with the current values of the model

        Parameters
        ----------
        button : ipywidget Button
            The refresh Button that was clicked on

        """
        # A kind of circular loop can happen. This code changes the value of the variable widgets
        #    which causes the handler for those widgets to fire. So use this flag to prevent that
        self._refresh_in_progress = True
        for box in self._value_widget_box.children[1:]:
            val_widget = box.children[0]
            val = self._prob[val_widget.description]
            if isinstance(val, np.ndarray):
                if val.size > 1:
                    return  # skip arrays for now
                val = val.item()

            val_widget.value = str(val)
        self._refresh_in_progress = False

    def _update_prob_val(self, change):
        """
        The handler for variable value text box. This is called if the value in the box changes

        Parameters
        ----------
        change : dict
            A dictionary holding the information about the change

        """
        # if self._refresh_in_progress,
        # no need to update problem since values from problem are being used to change the value
        # in the widget
        if not self._refresh_in_progress:
            val = float(change['new']) if change['new'] else 0.0

            # get the units also for the set_val call
            var_name = change['owner'].description
            corresponding_units_widget = self._get_unit_widget(var_name)  # TODO need a link here
            units = corresponding_units_widget.value
            if units == 'None':
                units = None

            self._prob.set_val(change['owner'].description, val, units=units)

    def _update_prob_unit(self, change):
        """
        The handler for variable units text box. This is called if the value in the box changes

        Parameters
        ----------
        change : dict
            A dictionary holding the information about the change

        """
        var_name = change['owner']._var_name
        units = change['new']

        corresponding_val_widget = self._get_val_widget(var_name) # TODO use links

        val = float(corresponding_val_widget.value)
        self._prob.set_val(var_name, val, units=units)

    def _get_unit_widget(self, var_name):   # TODO - still need it?
        corresponding_units_widget = None
        for box in self._value_widget_box.children[1:]:
            units_widget = box.children[1]
            if units_widget._var_name == var_name:
                corresponding_units_widget = units_widget
                break
        if not corresponding_units_widget:
            raise (f"Corresponding units widget for {var_name} not found")
        return corresponding_units_widget

    def _get_val_widget(self, var_name):   # TODO - still need it?
        corresponding_val_widget = None
        for box in self._value_widget_box.children[1:]:
            val_widget = box.children[0]
            if val_widget.description == var_name:
                corresponding_val_widget = val_widget
                break
        if not corresponding_val_widget:
            raise (f"Corresponding val widget for {var_name} not found")
        return corresponding_val_widget

    def _get_widget_var_names(self): # TODO - more efficient way?
        var_names = []
        for box in self._value_widget_box.children[1:]:
            float_text_widget = box.children[0]
            var_names.append(float_text_widget.description)
        self._output.append_stdout(f"result of _get_widget_var_names is {var_names}\n")
        return var_names

    def _get_var_names_at_this_level(self, sys):
        """
        Get the list of all the variables that have been promoted to this System, sys, and no
        higher.

        Parameters
        ----------
        sys : OpenMDAO System
            The OpenMDAO System that the caller wants the promoted variables from

        Returns
        -------
        list
            List of variables that have been promoted to this System, sys, and no higher.
        """
        var_names_at_this_level = []
        # current_group_absolute_path = sys.pathname
        if isinstance(sys, Component):
            for abs_name, prom_name in sys._var_allprocs_abs2prom['input'].items():
                current_group_var_promoted_name = prom_name
                # if they're different you would know that someone above the current group promoted it
                # current_group_pathname = f"{current_group_absolute_path}.{current_group_var_promoted_name}"
                # if promoted_name_from_top_level == current_group_pathname:
                #     var_names_at_this_level.add(prom_name)

                if "." not in current_group_var_promoted_name:  # this seems sufficient
                    # if promoted_name_from_top_level.endswith(f"{sys.name}.{prom_name}"):
                    if not self._does_parent_sys_have_this_var_promoted( sys, abs_name):
                            var_names_at_this_level.append((prom_name,abs_name))
        else: # Group
            for abs_name, prom_name in sys._var_allprocs_abs2prom['input'].items():
                # promoted_name_from_top_level = model_abs2prom[abs_name]
                current_group_var_promoted_name = prom_name
                # if they're different you would know that someone above the current group promoted it
                # current_group_pathname = f"{current_group_absolute_path}.{current_group_var_promoted_name}"

                if sys.pathname == '':  # root
                    if not "." in prom_name:
                        # don't want dups based on the prom_name, which is the first item in the
                        # tuple
                        if prom_name not in [a[0] for a in var_names_at_this_level]:
                            var_names_at_this_level.append((prom_name,abs_name))
                else:
                    if "." not in current_group_var_promoted_name:  # this seems sufficient
                        # WHAT IF a group higher up has the same name and the var is promoted to that level?
                        if not self._does_parent_sys_have_this_var_promoted(sys, abs_name):
                            # don't want dups based on the prom_name, which is the first item in
                            # the tuple
                            if prom_name not in [a[0] for a in var_names_at_this_level]:
                                    var_names_at_this_level.append((prom_name,abs_name))
                    # if promoted_name_from_top_level == current_group_pathname:   # DO I really need this?
                    #     var_names_at_this_level.add(prom_name)

        # need to sort them
        var_names_at_this_level.sort(key = lambda x: x[0].lower())
        return var_names_at_this_level

    def _does_parent_sys_have_this_var_promoted(self, sys, abs_name):
        #  TODO. It works for when sys is model, but should really handle that explicitly
        parent_pathname_and_child = sys.pathname.rsplit('.', 1)
        if len(parent_pathname_and_child) > 1:
            parent_pathname = parent_pathname_and_child[0]
        else:
            parent_pathname = ''
        parent = self._prob.model._get_subsystem(parent_pathname)
        parent_vars = parent._var_allprocs_abs2prom['input']
        if abs_name in parent_vars:
            if "." in parent_vars[abs_name]:
                return False # so it is at this level
            else:
                return True  # parent has same var and it's promoted name in that group has no .
                             # so it has it at that level at least. Definitely not at this sys level
        else:
            raise(f"shouldn't happen! {abs_name} not found in var for {parent_pathname}")
