"""Test N2 GUI with multiple models using Node.js."""
import os
import shutil
import unittest
import subprocess

# set DEBUG to True if you want to view the generated HTML file
GUI_TEST_SUBDIR = 'gui_test_models'
GUI_TEST_EXE = 'test_gui.js'
DEBUG = True

@unittest.skipUnless(shutil.which('node') != None, "Node.js is required to test the N2 GUI.")
class N2GUITestCase(unittest.TestCase):
    def setUp(self):
        """
        Generate the N2 HTML files from all models in GUI_TEST_SUBDIR.
        """
        self.parentDir = os.path.dirname(os.path.realpath(__file__))
        self.modelDir = self.parentDir + '/' + GUI_TEST_SUBDIR
        models = filter(lambda x: x.endswith('.py'), os.listdir(self.modelDir))
        self.basenames = map(lambda x: x[:-3], models)
        self.n2files = []

        for n in self.basenames:
            n2file = self.modelDir + '/' + n + '_N2_TEST.html'
            pyfile = self.modelDir + '/' + n + '.py'
            self.n2files.append(n2file)
            # os.system('openmdao n2 -o ' + n2file + ' --no_browser ' + pyfile)
            subprocess.call(['openmdao', 'n2', '-o', n2file,  '--no_browser', pyfile])
        
    def test_n2_common(self):
        """
        Perform a set of generic tests against generated N2 files.
        """
        testCmd = self.parentDir + '/' + GUI_TEST_EXE
        # n2html_files = ','.join(map(str, self.n2files))

        for n2file in self.n2files:
        # The Node.js script will exit with a non-zero value if it
        # detects an error.
            self.assertEqual(os.system(testCmd + ' --n2files=' + n2file), 0,
                'N2 common test failed with ' + n2file)
        #    self.assertEqual(subprocess.call([testCmd, '--n2files=' + n2file]), 0,
        #        'N2 common test failed with ' + n2file)

    def tearDown(self):
        if not DEBUG:
            try:
                for n2html in self.n2files:
                    os.remove(n2html)
            except:
                # Don't want the test to fail if the test file is already removed
                pass

if __name__ == "__main__":
    unittest.main()
