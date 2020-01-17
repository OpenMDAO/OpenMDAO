"""Test N2 GUI with multiple models using Node.js."""
import os
import json
import unittest
import distutils.spawn
import subprocess
from pyppeteer import launch
import asyncio

# set DEBUG to True if you want to view the generated HTML file
GUI_TEST_SUBDIR = 'gui_test_models'
GUI_TEST_EXE = 'test_gui.js'
GUI_N2_SUFFIX = '_N2_TEST.html'
URL_PREFIX = 'file://'
DEBUG = True
currentTestDesc = ''
LINE_STR = '-' * 78

@unittest.skipUnless(distutils.spawn.find_executable('node') != None, "Node.js is required to test the N2 GUI.")
class N2GUITestCase(unittest.TestCase):

    async def handleConsoleErr(self, msg):
        if msg.type == 'warning':
            print("Warning: " + currentTestDesc + "\n")
            for m in msg:
                print(msg + "\n")
        elif msg.type == 'error':
            self.fail(msg)

    async def setupErrorHandlers(self):
        self.page.on('console', lambda msg: self.handleConsoleErr(msg))
        self.page.on('pageerror', lambda msg: self.fail(msg))

    async def setupBrowser(self):
        self.browser = await launch({
            'defaultViewport': {
                'width': 1280,
                'height': 1024
            }
        })
        userAgentStr = await self.browser.userAgent()
        print("Browser: " + userAgentStr + "\n")

        self.page = await self.browser.newPage()
        await self.setupErrorHandlers()

    def logTest(self, msg):
        print ("  Test {:04}".format(self.current_test) + ": " + msg)
        self.current_test += 1

    def setUp(self):
        """
        Generate the N2 HTML files from all models in GUI_TEST_SUBDIR.
        """
        self.parentDir = os.path.dirname(os.path.realpath(__file__))
        self.modelDir = os.path.join(self.parentDir, GUI_TEST_SUBDIR)
        models = filter(lambda x: x.endswith('.py'), os.listdir(self.modelDir))
        self.basenames = map(lambda x: x[:-3], models)
        self.n2files = []
        self.current_test = 1

        # Load the scripts
        with open(os.path.join(self.parentDir, "gui_scripts.json"), "r") as read_file:
            self.scripts = json.load(read_file)

        self.known_model_names = self.scripts.keys()

        for n in self.basenames:
            n2file = os.path.join(self.modelDir, n + GUI_N2_SUFFIX)
            pyfile = os.path.join(self.modelDir, n + '.py')
            self.n2files.append(n2file)
            print("Creating " + n2file)
            subprocess.call(
                ['openmdao', 'n2', '-o', n2file,  '--no_browser', pyfile])

            # Create an N2 using declared partials
            if (n == 'circuit'):
                n2file = self.modelDir + '/udpi_' + n + GUI_N2_SUFFIX
                self.n2files.append(n2file)

                print("Creating " + n2file)

                subprocess.call(
                    ['openmdao', 'n2', '-o', n2file,  '--no_browser',
                     '--use_declare_partial_info', pyfile])

    async def load_test_page(self):
        print("\n" + LINE_STR + "\n" + os.path.basename(self.n2_filename) + "\n" + LINE_STR)
        url = URL_PREFIX + '/' + self.n2_filename

        # Without waitUntil: 'networkidle0', processing will begin before the page
        # is fully rendered
        await self.page.goto(url, waitUntil = 'domcontentloaded')
        
        # Milliseconds to allow for the last transition animation to finish.
        # Obtain value defined in N2 code.
        self.transitionWait = await self.page.evaluate("N2TransitionDefaults.durationSlow")
        self.transitionWait += 100
        print("  Transition wait set to " + str(self.transitionWait) + "ms")
        self.normalWait = 10
        await self.page.waitFor(self.transitionWait)

    # Click most of the toolbar buttons to see if an error occurs
    async def generic_toolbar_tests(self):
        for test in self.scripts['__toolbar']:
            self.logTest(test['desc'])
            btnHandle = await self.page.querySelector('#' + test['id'])
            await btnHandle.click(button='left', delay=5)
            waitTime = self.transitionWait if test['waitForTransition'] else self.normalWait
            await self.page.waitFor(waitTime)

    async def run_gui_tests(self):
        await self.setupBrowser()

        for self.n2_filename in self.n2files:
            await self.load_test_page()
            await self.generic_toolbar_tests()

        await self.browser.close()


    def test_n2_gui(self):
        # Get the async event loop going
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        coro = asyncio.coroutine(self.run_gui_tests)
        event_loop.run_until_complete(coro())
        event_loop.close()

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
