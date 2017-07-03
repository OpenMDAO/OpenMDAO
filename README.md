# blue
This is the PRE-ALPHA version of OpenMDAO 2.0
(we have codenamed it 'blue').

Important Note:
---------------

While the API is MOSTLY stable, we reserve the right to change things as needed.
Production runs should still be done in 1.7.x for now.

We will be making very frequent updates to this code. If you’re going to try it,
make sure you pull these updates often

Installation Instructions:
--------------------------

Use git to clone the repository:

`git clone http://github.com/OpenMDAO/blue`

Use pip to install openmdao locally:

`cd blue/openmdao`
`pip install -e .`


Documentation Building Instructions:
------------------------------------

`cd openmdao/docs`
`make all`

This will build the docs into openmdao/docs/_build/html.
To view the docs in your browser:

`open _build/html/index.html`

or simply use your browser's File->Open File command.
