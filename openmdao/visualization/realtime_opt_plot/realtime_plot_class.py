class _RealTimePlot(object):
    def __init__(
        self, case_tracker, callback_period, doc, pid_of_calling_script   # TODO why isn't callback_period used?
    ):
        """
        Construct and initialize _RealTimeOptPlot instance.
        """
        # self._case_recorder_filename = case_recorder_filename
        self._case_tracker = case_tracker
        self._pid_of_calling_script = pid_of_calling_script
        self._doc = doc

