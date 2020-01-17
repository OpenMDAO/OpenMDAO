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
        print("  Test {:04}".format(self.current_test) + ": " + msg)
        self.current_test += 1

    def generate_n2_files(self):
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

    def setUp(self):
        self.generate_n2_files()

    async def load_test_page(self):
        print("\n" + LINE_STR + "\n" +
              os.path.basename(self.n2_filename) + "\n" + LINE_STR)
        url = URL_PREFIX + '/' + self.n2_filename

        # Without waitUntil: 'networkidle0', processing will begin before the page
        # is fully rendered
        await self.page.goto(url, waitUntil='domcontentloaded')

        # Milliseconds to allow for the last transition animation to finish.
        # Obtain value defined in N2 code.
        self.transitionWait = await self.page.evaluate("N2TransitionDefaults.durationSlow")
        self.transitionWait += 500
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

    # Count the number of elements located by the selector and make
    # sure it matches the supplied value.
    async def assert_element_count(self, selector, expected_found):
        hndl_list = await self.page.querySelectorAll(selector)

        self.assertIsNot(hndl_list, False, "Could not find any '" + selector + "' elements.")        
        self.assertEqual(len(hndl_list), expected_found,
            'Found ' + str(len(hndl_list)) + ' elements, expected ' + str(expected_found))

    # Count the number of path elements in the n2arrows < div > and make
    # sure it matches the specified value.
    async def assert_arrow_count(self, expected_arrows):
        await self.assert_element_count('g#n2arrows > path', expected_arrows)

    # Get a handle for the specific element and assert that it exists.
    async def get_handle(self, selector):
        handle = await self.page.querySelector(selector)

        self.assertIsNotNone(handle, "Could not find element with selector '" +
                        selector + "' in the N2 diagram.")

        return handle

    # Hover over a matrix cell, make sure the number of expected arrows are there,
    # then move off and make sure the arrows go away.
    async def hover_and_check_arrow_count(self, options):
        self.logTest(options['desc'] if options['desc'] else 
            "Hover over '" + options['selector'] + "' and checking arrow count")
        
        hndl = await self.get_handle(options['selector'])

        await hndl.hover()
        await self.page.waitFor(self.normalWait) # Give it a chance to draw the arrows

        await self.assert_arrow_count(options['arrowCount']) # Make sure there are enough arrows
        await self.page.mouse.move(0, 0) # Get the mouse off the element
        await self.page.waitFor(self.normalWait)
        await self.assert_arrow_count(0) # Make sure there are no arrows left

    # Perform a click of the type specified by options.button on the
    # element specified by options.selector.
    async def click(self, options):
        self.logTest(options['desc'] if options['desc'] else
            options['button'] + "-click on '" + options['selector'] + "'")
        hndl = await self.get_handle(options['selector'])
        await hndl.click( button = options['button'])
        await self.page.waitFor(self.transitionWait)

    # Left-click the home button and wait for the transition to complete.
    async def return_to_root():
        self.logTest("Return to root")
        hndl = await self.get_handle("button#returnToRootButtonId.myButton")
        await hndl.click()
        await self.page.waitFor(self.transitionWait)

    # Enter a string in the search textbox and check that the expected
    # number of elements are shown in the N2 matrix.
    async def search_and_check_result(self, options):
        self.logTest(options['desc'] if options['desc'] else
            "Searching for '" + options['searchString'] + "' and checking for " +
            str(options['n2ElementCount']) + " N2 elements after.")

        hndl = await self.get_handle("div#toolbarLoc input#awesompleteId")
        await hndl.type(options['searchString'])
        await hndl.press('Enter')
        await self.page.waitFor(self.transitionWait + 500)

        await self.assert_element_count("g#n2elements > g.n2cell", options['n2ElementCount'])

   # Iterate through the supplied script array and perform each
   # action/test.
    async def run_model_script(self, script):
        print("Performing diagram-specific tests...")
        await self.page.reload(waitUntil='domcontentloaded')
        await self.page.waitFor(self.transitionWait)

        for script_item in script:
            test_type = script_item['test']
            if test_type == 'hoverArrow':
                await self.hover_and_check_arrow_count(script_item)
            elif test_type == 'click':
                await self.click(script_item)
            elif test_type == 'root':
                await self.return_to_root()
            elif test_type == 'search':
                await self.search_and_check_result(script_item)
            elif test_type == 'count':
                self.logTest(scriptItem['desc'] if scriptItem['desc'] else
                        "Checking for " + str(scriptItem['count']) + "' instances of '" +
                        scriptItem['selector'] + "'")
                await assert_element_count(scriptItem['selector'], scriptItem['count'])

    async def run_gui_tests(self):
        await self.setupBrowser()

        for self.n2_filename in self.n2files:
            await self.load_test_page()
            await self.generic_toolbar_tests()

            bname = os.path.basename(self.n2_filename)[:-len(GUI_N2_SUFFIX)]
            
            if bname in self.known_model_names:
                await self.run_model_script(self.scripts[bname])

        await self.browser.close()

    def test_n2_gui(self):
        # Get the async event loop going
        event_loop=asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        coro=asyncio.coroutine(self.run_gui_tests)
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
