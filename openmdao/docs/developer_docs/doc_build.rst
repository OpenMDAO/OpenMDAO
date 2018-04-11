Local Building of OpenMDAO Documentation
----------------------------------------

When developing new documentation for OpenMDAO, it's necessary to build the documents yourself, to ensure things like code
embedding and formatting and links are all working the way the author intends them to work. There are a few options available
to a developer who is building the OpenMDAO documentation locally. First, though, you'll need to enter the docs directory:

:code:`cd openmdao/docs`

make clean
##########

When starting off, it's important to be sure that the docs you're building are your own, that there are no vestiges of
previous documentation builds holding you back.  That's where :code:`make clean` comes in. By invoking this command, everything is removed
from `_build`, `_srcdocs`, `tags`, and `tmp`.  :code:`make clean` is the go-to command for wiping things out and starting fresh.


make all / make html
####################

:code:`make all` (also aliased as :code:`make html`) is the usual command for a first build of the docs. This build
is what you want to do to make a baseline build of all the docs.  :code:`make all` will build the docs, and deposit the output into
:code:`openmdao/docs/_build/html.` Then, you need to just :code:`open openmdao/docs/_build/html/index.html`
to open your default browser and review what you've written.

For the real clean-slate documentation build, combine the two commands: :code:`make clean; make all`.

.. note::
    Running this command discards existing source docs directory (`docs/_srcdocs/`) and rebuilds them entirely, which takes time.

make html-update / make
#######################

:code:`make html-update` (also the default behavior you'd get in `OpenMDAO/docs` for just typing :code:`make`) is intended to help save build time
for a developer who is re-making one or two files over and over again. When there is no need to rebuild the entire document base, :code:`make html-update`
just builds the docs that have been changed between this build and the previous build.

make single <filename>
######################

Sometimes, when you're embedding code and tests into documents, your Python *code* changes, not the `rst` document.
You want to see what your document will look like, with that new code being embedded and/or running and embedding its output.
This is another situation where you could :code:`make clean; make all`, but that takes a lot of time.
That's where :code:`make single file=<filename>` comes in. :code:`make single file=<filename>` touches the `file` argument provided, to
make the rst file appear to have been updated, then runs :code:`html-update`, which means the single file and any others that
have changed will be rebuilt.

.. note:: The angle brackets aren't needed. An example: :code:`make single file=index.rst`