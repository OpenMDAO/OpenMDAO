"""Test ShellProc functions."""
import unittest

import logging
import os.path
import shutil
import signal
import sys
import tempfile

from openmdao.utils.shell_proc import call, check_call, CalledProcessError, ShellProc


class TestCase(unittest.TestCase):
    """ Test ShellProc functions. """

    def setUp(self):
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='test_shellproc-')
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass

    def test_call(self):
        logging.debug('')
        logging.debug('test_call')

        cmd = 'dir' if sys.platform == 'win32' else 'ls'
        try:
            return_code, error_msg = call(cmd, stdout='stdout', stderr='stderr')
            self.assertEqual(os.path.exists('stdout'), True)
            self.assertEqual(os.path.exists('stderr'), True)
        finally:
            if os.path.exists('stdout'):
                os.remove('stdout')
            if os.path.exists('stderr'):
                os.remove('stderr')

    def test_check_call(self):
        logging.debug('')
        logging.debug('test_check_call')

        cmd = 'dir' if sys.platform == 'win32' else 'ls'
        try:
            check_call(cmd, stdout='stdout', stderr='stderr')
            self.assertEqual(os.path.exists('stdout'), True)
            self.assertEqual(os.path.exists('stderr'), True)
        finally:
            if os.path.exists('stdout'):
                os.remove('stdout')
            if os.path.exists('stderr'):
                os.remove('stderr')

        try:
            check_call('no-such-command', stdout='stdout', stderr='stderr')
            self.assertEqual(os.path.exists('stdout'), True)
            self.assertEqual(os.path.exists('stderr'), True)
        except CalledProcessError as exc:
            msg = "Command 'no-such-command' returned non-zero exit status"
            self.assertEqual(str(exc)[:len(msg)], msg)
        else:
            self.fail('Expected CalledProcessError')
        finally:
            if os.path.exists('stdout'):
                os.remove('stdout')
            if os.path.exists('stderr'):
                os.remove('stderr')

    def test_errormsg(self):
        logging.debug('')
        logging.debug('test_errormsg')

        cmd = 'dir' if sys.platform == 'win32' else 'ls'
        try:
            proc = ShellProc(cmd, stdout='stdout', stderr='stderr')
            proc.wait()
        finally:
            if os.path.exists('stdout'):
                os.remove('stdout')
            if os.path.exists('stderr'):
                os.remove('stderr')

        msg = proc.error_message(-signal.SIGTERM)
        if sys.platform == 'win32':
            self.assertEqual(msg, '')
        else:
            self.assertEqual(msg, ': SIGTERM')


if __name__ == '__main__':
    unittest.main()
