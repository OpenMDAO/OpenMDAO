Local Building of OpenMDAO Documentation
----------------------------------------

When developing new documentation for OpenMDAO, it's necessary to build the documents locally, to ensure things like code
embedding and formatting and links are all working the way the author intends them to work. There are a few options available
to a developer who is building the OpenMDAO documentation locally. First, though, you'll need to enter the docs directory:

:code:`cd openmdao/docs`

Once you're in the docs directory, you can try any of the commands listed below. If you have trouble remembering all the
commands, :code:`make help` will provide a brief command summary for you.

make clean
##########

When starting off, it's important to be sure that the docs you're building are your own, that there are no vestiges of
previous documentation builds holding you back.  That's where :code:`make clean` comes in. By invoking this command, everything is removed
from `_build`, `_srcdocs`, `tags`, and `tmp`.  :code:`make clean` is the go-to command for wiping things out and starting fresh.
(often used in conjunction with :code:`make all`.

make all / make html
####################

:code:`make all` (also aliased as :code:`make html`) is the usual command for a first build of the docs. This build
is what you want to do to make a baseline build of all the docs.  :code:`make all` will build the docs, and deposit the output into
:code:`openmdao/docs/_build/html.` When the build completes, you need to simply :code:`open openmdao/docs/_build/html/index.html`
to open your default browser and review what you've written.

For the real clean-slate documentation build, combine the two commands: :code:`make clean; make all`.

make html-update / make
#######################

:code:`make html-update` (also the default behavior you'd get in `OpenMDAO/docs` for just typing :code:`make`) is intended to help save build time
for a developer who is re-making one or two files over and over again. When there is no need to rebuild the entire document base, :code:`make html-update`
just builds the docs that have been changed between this build and the previous build. Once your document appears to be working the way you want, it's
good practice to go ahead and do a full :code:`make clean; make all`.

make single <filename>
######################

Sometimes, when you're embedding code and tests into documents, your Python *code* changes, not the `rst` document.
You want to see what your document will look like, with that new code being embedded and/or running and embedding its output.
This is another situation where you could :code:`make clean; make all`, but that takes a lot of time.
That's where :code:`make single file=<filename>` comes in. :code:`make single file=<filename>` touches the `file` argument provided, to
make the rst file appear to have been updated, then runs :code:`html-update`, which means the named file and any others that
have changed since the last build will be rebuilt.

.. note:: The angle brackets above aren't part of the syntax. An example: :code:`make single file=index.rst`