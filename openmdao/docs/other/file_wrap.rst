.. index:: File Wrapping Tools

.. _filewrap_feature:

*************
File Wrapping
*************


Communicating with External Codes in OpenMDAO
=============================================


The :ref:`ExternalCodeComp <externalcodecomp_feature>` example describes how to
define a component that calls an external program to perform its computation,
passing input and output values via files.

The input and output files were very simple in that basic example, containing only
the values of interest.  In the general case however, you will probably need to
generate an input file (with a specific format of rows and fields), and you'll also need to parse a
similarly-formatted output file to get the output values. To facilitate working
with these more complex input and output files, OpenMDAO provides a couple of utility
classes: `InputFileGenerator` and `FileParser`.


Generating the Input File
-------------------------

You can generate an input file for an external application in a few different ways.
One way is to write the file completely from scratch using the new values that are
contained in the component's variables. Not much can be done to aid with this task, as
it requires knowledge of the file format and can be completed using Python's standard
formatted output.

A second way to generate an input file is by templating. A *template* file is
a sample input file which can be processed by a templating engine to insert
new values in the appropriate locations. Often the template file is a valid
input file before being processed, although other times it contains directives
or conditional logic to guide the generation. Obviously this method works well
for cases where only a small number of the possible variables and settings are
being manipulated.

OpenMDAO includes a basic templating capability that allows a template file to
be read, fields to be replaced with new values, and an input file to be
generated so that the external application can read it. Suppose you have an
input file that contains some integer, floating point, and string inputs:

::

    INPUT
    1 2 3
    INPUT
    10.1 20.2 30.3
    A B C

This is a valid input file for your application, and it can also be used as a
template file. The templating object is called `InputFileGenerator`, and it
includes methods that can replace specific fields as measured by their row
and field numbers.

To use the InputFileGenerator, first instantiate it and give it the name of
the template file and the name of the output file that you want to produce. (Note
that this code must be placed in the ``compute`` method of your component *before*
the external code is run.) The code will generally look like this:

::

    from openmdao.util.file_wrap import InputFileGenerator

    parser = InputFileGenerator()
    parser.set_template_file('mytemplate.txt')
    parser.set_generated_file('myinput.txt')

    # (Call functions to poke new values here)

    parser.generate()

When the template file is set, it is read into memory so that all subsequent
replacements are done without writing the intermediate file to the disk. Once
all replacements have been made, the ``generate`` method is called to create the
input file.  If you have not provided the name of an output file, then the
generated file data will be returned as a string.  We will use this feature in
the following examples.


Let's say you want to replace the second integer in the input file above
with a 7. The code would look like this.


.. embed-code::
    openmdao.utils.tests.test_file_wrap.FileGenFeature.test_transfer
    :layout: interleave


.. index:: mark_anchor

The method ``mark_anchor`` is used to define an anchor, which becomes the
starting point for the ``transfer_var`` method. Here you find the first line
down from the anchor, then the second field on that line and replace it with
the new value.

Now, if you want to replace the third value of the floating point numbers
after the second ``INPUT`` statement. An additional argument can be passed to the
``mark_anchor`` method to tell it to start at the second instance of the text
fragment ``"INPUT"``.

.. embed-code::
    openmdao.utils.tests.test_file_wrap.FileGenFeature.test_transfer_2
    :layout: interleave


Note that you are able to pass a floating point value to ``transfer_var`` and still
keep 15 digits of precision. See :ref:`A-Note-on-Precision` for a discussion of
why this is important.

Note also that we used the method ``reset_anchor`` to return the anchor to the
beginning of the file before marking our new anchor. Subsequent calls to
``mark_anchor`` start at the previous anchor and find the next instance of the
anchor text. It is a good practice to reset your anchor unless you are looking for
an instance of "B" that follows an instance of "A".

You can also count backwards from the bottom of the file by passing a negative
number. Here, the second instance of ``"INPUT"`` from the bottom brings you
back to the first one.

.. embed-code::
    openmdao.utils.tests.test_file_wrap.FileGenFeature.test_transfer_minus2
    :layout: interleave


There is also a method for replacing an entire array of values. Try
replacing the set of three integers as follows:


.. embed-code::
    openmdao.utils.tests.test_file_wrap.FileGenFeature.test_transfer_array
    :layout: interleave


.. index:: transfer_array

The method ``transfer_array`` takes four required inputs. The first is an array
of values that will become the new values in the file. The second is the
starting row after the anchor. The third is the starting field that will be
replaced, and the fourth is the ending field. The new array replaces the
block of fields spanned by the starting field and the ending field.

You can also use the ``transfer_array`` method to `stretch` an existing
array in a template to add more terms.

.. embed-code::
    openmdao.utils.tests.test_file_wrap.FileGenFeature.test_transfer_stretch
    :layout: interleave


The named argument ``sep`` defines which separator to include between the
additional terms of the array.

The input file templating capability that comes with OpenMDAO is basic, but quite
functional. If you need a more powerful templating engine, particularly one that
allows the inclusion of logic in your template files, then you may want to consider
one of the community-developed templating_ engines.

.. _templating: https://wiki.python.org/moin/Templating

.. todo:: Include some examples with one of the templating engines.


Parsing the Output File
-----------------------

When an external code is executed, it typically outputs the results into a
file. OpenMDAO includes a utility called `FileParser`, which contains functions
for parsing a file, extracting the fields you specify, and converting them to the
appropriate data type.

*Basic Extraction*
~~~~~~~~~~~~~~~~~~~

Consider an application that produces the following as part of its
text file output:

::

    LOAD CASE 1
    STRESS 1.3334e7 3.9342e7 NaN 2.654e5
    DISPLACEMENT 2.1 4.6 3.1 2.22234
    LOAD CASE 2
    STRESS 11 22 33 44 55 66
    DISPLACEMENT 1.0 2.0 3.0 4.0 5.0

As part of the file wrap, you need to extract the information from this file
that is needed by downstream components in the model. For this to
work, the file must have some general format that would allow you to locate the
piece of data you need relative to some constant feature in the file. In other
words, the main capability of the FileParser is to locate and extract a set of
characters that is some number of lines and some number of fields away from an
`anchor` point.

::

    from openmdao.util.file_wrap import FileParser

    parser = FileParser()
    parser.set_file('output.txt')

To use the FileParser object, first instantiate it and give it the name of the
output file. (Note that this code must be placed in your component's
``compute`` function *after* the external code has been run.

Say you want to extract the first ``STRESS`` value from each load case in the file
snippet shown above. The code would look like this.

.. embed-code::
    openmdao.utils.tests.test_file_wrap.FileParserFeature.test_parse_output
    :layout: interleave


The method ``mark_anchor`` is used to define an anchor, which becomes the
starting point for the ``transfer_var`` method. Here, you extract the value from the
second field in the first line down from the anchor. The parser is smart enough to
recognize the number as floating point and to create a Python float variable.

The third value of ``STRESS`` is `NaN`. Python has built-in values for `nan`
and `inf` that are valid for float variables. The parser recognizes them when it
encounters them in a file. This allows you to catch numerical overflows,
underflows, etc., and take action. NumPy includes the functions ``isnan`` and
``isinf`` to test for `nan` and `inf` respectively.  In the following example,
we extract that `nan` value:

.. embed-code::
    openmdao.utils.tests.test_file_wrap.FileParserFeature.test_parse_nan
    :layout: interleave


When the data is not a number, it is recognized as a string. For example, we can
extract the word ``DISPLACEMENT``.

.. embed-code::
    openmdao.utils.tests.test_file_wrap.FileParserFeature.test_parse_string
    :layout: interleave


Now, what if you want to extract the value of stress from the second load case? An
additional argument can be passed to the ``mark_anchor`` method telling it to
start at the second instance of the text fragment ``"LOAD CASE"``.

.. embed-code::
    openmdao.utils.tests.test_file_wrap.FileParserFeature.test_parse_output_2
    :layout: interleave


Note also that we used the method ``reset_anchor`` to return the anchor to the
beginning of the file before marking our new anchor. Subsequent calls to
``mark_anchor`` start at the previous anchor and find the next instance of the
anchor text. It is a good practice to reset your anchor unless you are looking for
an instance of "B" that follows an instance of "A".

You can also count backwards from the bottom of the file by passing a negative
number. Here, the second instance of ``"LOAD CASE"`` from the bottom brings us
back to the first one.

.. embed-code::
    openmdao.utils.tests.test_file_wrap.FileParserFeature.test_parse_output_minus2
    :layout: interleave


There is a shortcut for extracting data that is stored as ``Key Value`` or
``"Key Value Value ..."``. The method ``transfer_keyvar`` finds the first occurrence
of the *key* string after the anchor (in this case, the word ``DISPLACEMENT``), and
extracts the specified field value. This can be useful in cases where variables are
found on lines that are uniquely named, particularly where you don't always know how
many lines the key will occur past the anchor location. There are two optional
arguments to ``transfer_keyvar``. The first lets you specify the `nth` occurrence
of the key, and the second lets you specify a number of lines to offset from
the line where the key is found (negative numbers are allowed).

.. embed-code::
    openmdao.utils.tests.test_file_wrap.FileParserFeature.test_parse_keyvar
    :layout: interleave


*Array Extraction*
~~~~~~~~~~~~~~~~~~

Now consider the same application that produces the following as part of its
text file output:

::

    LOAD CASE 1
    STRESS 1.3334e7 3.9342e7 NaN 2.654e5
    DISPLACEMENT 2.1 4.6 3.1 2.22234
    LOAD CASE 2
    STRESS 11 22 33 44 55 66
    DISPLACEMENT 1.0 2.0 3.0 4.0 5.0

This time, extract all of the displacements in one read and store
them as an array. You can do this with the ``transfer_array`` method.

.. embed-code::
    openmdao.utils.tests.test_file_wrap.FileParserFeature.test_parse_array
    :layout: interleave


The ``transfer_array`` method takes four arguments: *starting row*, *starting field*,
*ending row*, and *ending field*. The parser extracts all values from the starting
row and field and continues until it hits the ending field in the ending row.
These values are all placed in a 1D array. When extracting multiple lines, if
a line break is hit, the parser continues reading from the next line until the
last line is hit. The following extraction illustrates this:

.. embed-code::
    openmdao.utils.tests.test_file_wrap.FileParserFeature.test_parse_array_multiline
    :layout: interleave


With the inclusion of ``'DISPLACEMENT'``, this is returned as an array of strings,
so you must be careful.

There is also a method to extract a 2-dimensional array from tabulated data.
Consider an output table that looks like this:

::

        FREQ  DELTA   A     B     C     D     E     F     G     H     I     J
         Hz
         50.   1.0   30.0  34.8  36.3  36.1  34.6  32.0  28.4  23.9  18.5  12.2
         63.   1.0   36.5  41.3  42.8  42.6  41.1  38.5  34.9  30.4  25.0  18.7
         80.   1.0   42.8  47.6  49.1  48.9  47.4  44.8  41.2  36.7  31.3  25.0
        100.   1.0   48.4  53.1  54.7  54.5  53.0  50.4  46.8  42.3  36.9  30.6

We would like to extract the relevant numerical data from this table, which
amounts to all values contained in columns labeled "A" through "J" and rows
labeled "50 Hz" through "100 Hz." We would like to save these values in a
two-dimensional numpy array. This can be accomplished using the
``transfer_2Darray`` method.

.. embed-code::
    openmdao.utils.tests.test_file_wrap.FileParser2dFeature.test_parse_array_2d
    :layout: interleave


The arguments to ``transfer_2Darray`` are the *starting row*, *starting field*,
*ending row*, and *ending field*. If the ending field is omitted, then all values
to the end of the line are extracted. In that case, care must be taken to make
sure that all lines have the same number of values.

Note that if the delimiter is set to ``'columns'``, then the column number should be
entered instead of the field number. Delimiters are discussed in the next section.

.. index:: delimiters

*Delimiters*
~~~~~~~~~~~~

When the parser counts fields in a line of output, it determines the field
boundaries by comparing against a set of delimiters. These delimiters can be
changed using the ``set_delimiters`` method. By default, the delimiters are the
general white space characters space (``" "``) and tab (``"\t"``). The newline characters
(``"\n"`` and ``"\r"``) are always removed regardless of the delimiter status.

One common case that will require a change in the default delimiter is comma
separated values (i.e. `csv`). Here's an example of such an output file:

.. embed-code::
    openmdao.utils.tests.test_file_wrap.FileParserDelimFeature.test_parse_default_delim
    :layout: interleave


What happened here is slightly confusing, but the main point is that the parser
did not handle this as expected because commas were not in the set of
delimiters. Now specify commas as your delimiter.

.. embed-code::
    openmdao.utils.tests.test_file_wrap.FileParserDelimFeature.test_parse_comma_delim
    :layout: interleave


With the correct delimiter set, you extract the second integer as expected.

While the ability to set the delimiters adds flexibility for parsing many
different types of input files, you may find cases that are too complex to
parse (e.g., a field with separator characters inside of quotes.) In such cases
you may need to read and extract the data manually.

*Special Case Delimiter - Columns*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One special-case value of the delimiter, ``'columns'``, is useful when the
data fields have defined column location, as is the case in certain formatted
output from Fortran or C. When the delimiter is set to ``'columns'``, the
behavior of some of the methods is slightly different. Consider the following
output file:

::

    CASE 1
    12345678901234567890
    TTF    3.7-9.4434967

The second line is a comment that helps the reader identify the column
number (particularly on a printout) and does not need to be parsed.

In the third line, the first three columns contain flags that are either ``'T'``
or ``'F'``. Columns 4-10 contain a floating point number, and columns 11
through 20 contain another floating point number. Note that there isn't
always a space between the two numbers in this format, particularly when the
second number has a negative sign. We can't parse this with a regular
separator, but we can use the special separator ``'columns'``.

Let's parse this file to extract the third boolean flag and the two numbers.

When the delimiters are in column mode, ``transfer_var`` takes the starting
field and the ending field as its second and third arguments. Since we just
want one column for the boolean flag, the starting field and ending field are
the same. For the floating point values, we provide the appropriate column ranges:

.. embed-code::
    openmdao.utils.tests.test_file_wrap.FileParserColumnsFeature.test_parse_columns
    :layout: interleave



The ``transfer_array`` method can also be used with columns, but it is used
differently than ``transfer_var``. Consider this output file:

::

    CASE 2
    123456789012345678901234567890
    NODE 11 22 33 COMMENT
    NODE 44 55 66 STUFF

In this example, we want to extract the six numerical values and place them in
an array. When the delimiter is set to columns, we can define a rectangular
box from which all elements are parsed into an array. Note that the numbers
inside of the box are parsed assuming standard separator characters (``" \t"``).

So here we call ``transfer_array`` with four arguments: *starting row*,
*starting column*, *ending row*, and *ending column*:

.. embed-code::
    openmdao.utils.tests.test_file_wrap.FileParserArrayColumnsFeature.test_parse_columns
    :layout: interleave


Note that, in this case, we exit column mode and return to normal delimiter
parsing by setting the delimiters back to the default after extracting the
desired values.


.. index:: Fortran namelists

A Special Case - Fortran Namelists
----------------------------------

Since legacy Fortran codes are expected to be frequent candidates for
file wrapping, you may also consider using the f90nml_ package for reading
and writing files to wrap those codes. This package enables the creation and
manipulation of namelist files using the common Python dictionary interface.

.. _f90nml: https://f90nml.readthedocs.io/en/latest/

.. todo:: Include an example with f90nml.


.. _A-Note-on-Precision:

A Note on Precision
---------------------

In a file-wrapped component, all key inputs for the external code come from an intermediate file
that must be written. When generating the input file, it is important to prevent the loss of
precision. Consider a variable with 15 digits of precision.

::

    >>> # Python 3 compatibility
    >>> from __future__ import print_function
    >>> val = 3.1415926535897932
    >>>
    >>> val
    3.141592653589793...
    >>>
    >>> print(val)
    3.14159265359
    >>>
    >>> print("%s" % str(val))
    3.14159265359
    >>>
    >>> print("%f" % val)
    3.141593
    >>>
    >>> print("%.16f" % val)
    3.141592653589793...

If the variable's value in the input file is created using the ``print``
statement, only 11 digits of precision are in the generated output. The same
is true if you convert the value to a string and use string output formatting.
Printing the variable as a floating point number with no format string gives
even less precision. To output the full precision of a variable, you must specify
decimal precision using formatted output (i.e., ``"%.16f"``).

Quibbling over the 11th--15th decimal place may sound unnecessary,
but some applications are sensitive to changes of this magnitude. Moreover, it
is important to consider how your component may be used during optimization. A
gradient optimizer will often use a finite difference scheme to calculate the
gradients for a model, and this means that some component params might be
subjected to small increments and decrements. A loss of precision here can
completely change the calculated gradient and prevent the optimizer from
reaching a correct minimum value.

The file-wrapping utilities in OpenMDAO use ``"%.16g"``. If you write your own
custom input-file generator for a new component, you should use this format
for the floating point variables.

Precision is also important when parsing the output, although the file-parsing
utilities always extract the entire number. However, some codes limit the number of
digits of precision in their output files for human readability. In such a case,
you should check your external application's manual to see if there is a flag for
telling the code to output the full precision.


.. tags:: ExternalCodeComp, FileWrapping
