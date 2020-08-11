"""Define utils for use in testing."""


def _new_setup(self):
    import os
    import tempfile

    from openmdao.utils.mpi import MPI
    self.startdir = os.getcwd()
    if MPI is None:
        self.tempdir = tempfile.mkdtemp(prefix='testdir-')
    elif MPI.COMM_WORLD.rank == 0:
        self.tempdir = tempfile.mkdtemp(prefix='testdir-')
        MPI.COMM_WORLD.bcast(self.tempdir, root=0)
    else:
        self.tempdir = MPI.COMM_WORLD.bcast(None, root=0)

    os.chdir(self.tempdir)
    if hasattr(self, 'original_setUp'):
        self.original_setUp()


def _new_teardown(self):
    import os
    import shutil

    from openmdao.utils.mpi import MPI
    if hasattr(self, 'original_tearDown'):
        self.original_tearDown()

    os.chdir(self.startdir)

    if MPI is None:
        rank = 0
    else:
        # make sure everyone's out of that directory before rank 0 deletes it
        MPI.COMM_WORLD.barrier()
        rank = MPI.COMM_WORLD.rank

    if rank == 0:
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass


def use_tempdirs(cls):
    """
    Decorate each test in a unittest.TestCase so it runs in its own directory.

    TestCase methods setUp and tearDown are replaced with _new_setup and
    _new_teardown, above.  Method _new_setup creates a temporary directory
    in which to run the test, stores it in self.tempdir, and then calls
    the original setUp method.  Method _new_teardown first runs the original
    tearDown method, and then returns to the original starting directory
    and deletes the temporary directory.

    Parameters
    ----------
    cls : TestCase
        TestCase being decorated to use a tempdir for each test.

    Returns
    -------
    TestCase
        The decorated TestCase class.
    """
    if getattr(cls, 'setUp', None):
        setattr(cls, 'original_setUp', getattr(cls, 'setUp'))
    setattr(cls, 'setUp', _new_setup)

    if getattr(cls, 'tearDown', None):
        setattr(cls, 'original_tearDown', getattr(cls, 'tearDown'))
    setattr(cls, 'tearDown', _new_teardown)

    return cls
