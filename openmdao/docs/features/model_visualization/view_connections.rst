.. _view_connections:

***************************
View Connections of a Model
***************************

View Connections from Command Line
##################################

The :code:`openmdao view_connections` command generates a table of connection information for all input and
output variables in the model.  This is documented on :ref:`openmdao view_connections <om-command-view_connections>`.

View Connections from Script
############################

You can also generate a display of model connections from within a script by calling the function :code:`view_connections`.

.. autofunction:: openmdao.visualization.connection_viewer.viewconns.view_connections
   :noindex:

Here is an example of how this function can be used.

.. embed-code::
    openmdao.devtools.tests.test_viewconns.TestSellarFeature.test_feature_sellar
    :layout: interleave

In this example, an HTML file named `sellar_connections.html` is created. This file can then be opened using using your
browser. The page will look like this.

.. figure:: view_connections_sellar.png
   :align: center
   :alt: An example of a connection viewer

   An example of a connection viewer.

You can use the two menus at the top of the page to filter the connections that are displayed. One does filtering
based on the source `System` and one based on the target `System`.