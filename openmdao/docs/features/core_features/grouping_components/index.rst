.. _feature_grouping_components:

*******************
Working with Groups
*******************

It's often desirable to represent a complex model as a collection of components.
Using :code:`Group` objects, we can group components, as well as other :code:`Groups`,
together to form a tree structure.  These feature docs explain how to create that
tree structure, how to access subsystems and variables within the tree, and how
to connect those variables.  These docs will also explain how to *promote* a
variable from a subsystem up to its parent.


.. toctree::
    :maxdepth: 1

    add_subsystem.rst
    connect.rst
    src_indices.rst
    set_order.rst
    get_subsystem.rst
    parallel_group.rst
    configure_method.rst
    post_setup_config.rst
