*********************
Profiling and Tracing
*********************

.. toctree::
    :maxdepth: 1

    inst_profile
    inst_mem_profile
    inst_call_tracing


All of the tools mentioned above have a similar programmatic interface, even though most of the
time they will only be used in command-line mode.  However, if you really want to customize the
set of methods that are to be profiled or traced, see the following example.


.. testcode:: tool_programatic_interface

    # for performance profiling
    from openmdao.devtools import iprofile as tool
    # OR for memory profiling
    # from openmdao.devtools import iprof_mem as tool
    # OR for call tracing
    # from openmdao.devtools import itrace as tool

    # First, make sure that the classes I use to define my method set exist in this
    # namespace.
    from mystuff import MyClass, MyOtherClass

    # Let's say I only want to track methods with 'foo' or 'bar' in the name that belong to
    # MyClass and ones with 'baz' in the name that belong to either MyClass or MyOtherClass.
    # I use the following glob patterns and tuples of classes to specify this.
    methods = [
        ('*foo*', (MyClass,)),
        ('*bar*', (MyClass,)),
        ('*baz*', (MyClass, MyOtherClass))
    ]

    # set up the tool using my custom method set
    tool.setup(methods=methods)

    tool.start()

    # run the code I want to profile/trace...

    tool.stop()

    # do some other stuff that I don't want to profile/trace...
