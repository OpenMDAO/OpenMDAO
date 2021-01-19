"""Test N2 GUI with multiple models using Pyppeteer."""
import asyncio
import pyppeteer
import subprocess
import unittest
import os

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized

# set DEBUG to True if you want to view the generated HTML file
GUI_TEST_SUBDIR = 'gui_test_models'
GUI_N2_SUFFIX = '_N2_TEST.html'
URL_PREFIX = 'file://'
DEBUG = False
LINE_STR = '-' * 78
current_test = 1

""" A set of toolbar tests that runs on each model. """
toolbar_script = [
    {
        "desc": "Uncollapse All button",
        "id": "expand-all",
        "waitForTransition": True
    },
    {
        "desc": "Collapse Outputs in View Only button",
        "id": "collapse-element-2",
        "waitForTransition": True
    },
    {
        "desc": "Uncollapse In View Only button",
        "id": "expand-element",
        "waitForTransition": True
    },
    {
        "desc": "Show Legend (off) button",
        "id": "legend-button",
        "waitForTransition": True
    },
    {
        "desc": "Show Legend (on) button",
        "id": "legend-button",
        "waitForTransition": False
    },
    {
        "desc": "Show Path (on) button",
        "id": "info-button",
        "waitForTransition": False
    },
    {
        "desc": "Show Path (off) button",
        "id": "info-button",
        "waitForTransition": False
    },
    {
        "desc": "Non-linear solver names button",
        "id": "non-linear-solver-button",
        "waitForTransition": False
    },
    {
        "desc": "Linear solver names button",
        "id": "linear-solver-button-2",
        "waitForTransition": True
    },
    {
        "desc": "Toggle solver visibility button (off)",
        "id": "no-solver-button",
        "waitForTransition": True
    },
    {
        "desc": "Toggle solver visibility button (on)",
        "id": "no-solver-button",
        "waitForTransition": True
    },
    {
        "desc": "Clear Arrows and Connection button",
        "id": "hide-connections",
        "waitForTransition": False
    },
    {
        "desc": "Help (on) button",
        "id": "question-button",
        "waitForTransition": False
    },
    {
        "desc": "Help (off) button",
        "id": "question-button",
        "waitForTransition": False
    },
    {
        "desc": "Collapse All Outputs button",
        "id": "collapse-all",
        "waitForTransition": True
    }
]

""" A dictionary of tests script with an array for each model."""
n2_gui_test_scripts = {
    "circuit": [
        {
            "desc": "Hover on N2 matrix element and check arrow count",
            "test": "hoverArrow",
            "selector": "g#n2elements rect#cellShape_node_22.vMid",
            "arrowCount": 4
        },
        {
            "desc": "Left-click on partition tree element to zoom",
            "test": "click",
            "selector": "g#tree rect#circuit_R2",
            "button": "left"
        },
        {
            "desc": "Hover on N2 matrix element and check arrow count",
            "test": "hoverArrow",
            "selector": "g#n2elements rect#cellShape_node_22.vMid",
            "arrowCount": 4
        },
        {
            "test": "root"
        },
        {
            "desc": "Right-click on partition tree element to collapse",
            "test": "click",
            "selector": "g#tree rect#circuit_n1",
            "button": "right"
        },
        {
            "desc": "Hover over collapsed N2 matrix element and check arrow count",
            "test": "hoverArrow",
            "selector": "g#n2elements rect#cellShape_node_6.gMid",
            "arrowCount": 5
        },
        {
            "desc": "Right-click on partition tree element to uncollapse",
            "test": "click",
            "selector": "g#tree rect#circuit_n1",
            "button": "right"
        },
        {
            "desc": "Left-click to zoom on solver element",
            "test": "click",
            "selector": "g#solver_tree rect#circuit_n1",
            "button": "left"
        },
        {
            "desc": "Hover over zoomed N2 cell and check arrow count",
            "test": "hoverArrow",
            "selector": "g#n2elements rect#cellShape_node_10.vMid",
            "arrowCount": 5
        },
        {
            "test": "root"
        },
        {
            "desc": "Right-click on solver element to collapse",
            "test": "click",
            "selector": "g#solver_tree rect#circuit_n1",
            "button": "right"
        },
        {
            "desc": "Hover over collapsed N2 cell and check arrow count",
            "test": "hoverArrow",
            "selector": "g#n2elements rect#cellShape_node_6.gMid",
            "arrowCount": 5
        },
        {
            "desc": "Right-click again on solver element to uncollapse",
            "test": "click",
            "selector": "g#solver_tree rect#circuit_n1",
            "button": "right"
        },
        {
            "test": "root"
        },
        {
            "desc": "Check the number of cells in the N2 Matrix",
            "test": "count",
            "selector": "g#n2elements > g.n2cell",
            "count": 40
        },
        {
            "desc": "Perform a search on V_out",
            "test": "search",
            "searchString": "V_out",
            "n2ElementCount": 11
        },
        {
            "test": "root"
        },
        {
            "desc": "Check that home button works after search",
            "test": "count",
            "selector": "g#n2elements > g.n2cell",
            "count": 40
        },
    ],
    "bug_arrow": [
        {
            "desc": "Check the number of cells in the N2 Matrix",
            "test": "count",
            "selector": "g#n2elements > g.n2cell",
            "count": 14
        },
        {
            "desc": "Hover on N2 matrix element and check arrow count",
            "test": "hoverArrow",
            "selector": "g#n2elements rect#cellShape_node_10.vMid",
            "arrowCount": 2
        },
        {
            "desc": "Left-click on partition tree element to zoom",
            "test": "click",
            "selector": "g#tree rect#design_fan_map_scalars",
            "button": "left"
        },
        {
            "desc": "Hover on N2 matrix element and check arrow count",
            "test": "hoverArrow",
            "selector": "g#n2elements rect#cellShape_node_13.vMid",
            "arrowCount": 2
        },
        {
            "test": "root"
        },
        {
            "desc": "Right-click on partition tree element to collapse",
            "test": "click",
            "selector": "g#tree rect#design_fan_map_scalars",
            "button": "right"
        },
        {
            "desc": "Hover over collapsed N2 matrix element and check arrow count",
            "test": "hoverArrow",
            "selector": "g#n2elements rect#cellShape_node_12.gMid",
            "arrowCount": 1
        },
        {
            "desc": "Right-click on partition tree element to uncollapse",
            "test": "click",
            "selector": "g#tree rect#design_fan_map_scalars",
            "button": "right"
        },
        {
            "desc": "Left-click to zoom on solver element",
            "test": "click",
            "selector": "g#solver_tree rect#design_fan_map_scalars",
            "button": "left"
        },
        {
            "desc": "Hover over zoomed N2 cell and check arrow count",
            "test": "hoverArrow",
            "selector": "g#n2elements rect#cellShape_node_13.vMid",
            "arrowCount": 2
        },
        {
            "test": "root"
        },
        {
            "desc": "Right-click on solver element to collapse",
            "test": "click",
            "selector": "g#solver_tree rect#design_fan_map_scalars",
            "button": "right"
        },
        {
            "desc": "Hover over collapsed N2 cell and check arrow count",
            "test": "hoverArrow",
            "selector": "g#n2elements rect#cellShape_node_12.gMid",
            "arrowCount": 1
        },
        {
            "desc": "Right-click again on solver element to uncollapse",
            "test": "click",
            "selector": "g#solver_tree rect#design_fan_map_scalars",
            "button": "right"
        }
    ],
    "double_sellar": [
        {
            "desc": "Hover on N2 matrix element and check arrow count",
            "test": "hoverArrow",
            "selector": "g#n2elements rect#cellShape_node_9.vMid",
            "arrowCount": 4
        },
        {
            "desc": "Left-click on partition tree element to zoom",
            "test": "click",
            "selector": "g#tree rect#g1_d2",
            "button": "left"
        },
        {
            "desc": "Hover on N2 matrix element and check arrow count",
            "test": "hoverArrow",
            "selector": "g#n2elements rect#cellShape_node_13.vMid",
            "arrowCount": 4
        },
        {
            "test": "root"
        },
        {
            "desc": "Right-click on partition tree element to collapse",
            "test": "click",
            "selector": "g#tree rect#g2_d1",
            "button": "right"
        },
        {
            "desc": "Hover over collapsed N2 matrix element and check arrow count",
            "test": "hoverArrow",
            "selector": "g#n2elements rect#cellShape_node_15.gMid",
            "arrowCount": 4
        },
        {
            "desc": "Right-click on partition tree element to uncollapse",
            "test": "click",
            "selector": "g#tree rect#g2_d1",
            "button": "right"
        },
        {
            "desc": "Left-click to zoom on solver element",
            "test": "click",
            "selector": "g#solver_tree rect#g2_d2",
            "button": "left"
        },
        {
            "desc": "Hover over zoomed N2 cell and check arrow count",
            "test": "hoverArrow",
            "selector": "g#n2elements rect#cellShape_node_23.vMid",
            "arrowCount": 4
        },
        {
            "test": "root"
        },
        {
            "desc": "Right-click on solver element to collapse",
            "test": "click",
            "selector": "g#solver_tree rect#g1_d1",
            "button": "right"
        },
        {
            "desc": "Hover over collapsed N2 cell and check arrow count",
            "test": "hoverArrow",
            "selector": "g#n2elements rect#cellShape_node_5.gMid",
            "arrowCount": 4
        },
        {
            "desc": "Right-click again on solver element to uncollapse",
            "test": "click",
            "selector": "g#solver_tree rect#g1_d1",
            "button": "right"
        },
        {
            "desc": "Right-click on partition tree element to collapse",
            "test": "click",
            "selector": "g#tree rect#g1",
            "button": "right"
        },
        {
            "desc": "Hover over N2 cell and check arrow count with collapsed group",
            "test": "hoverArrow",
            "selector": "g#n2elements rect#cellShape_node_17.vMid",
            "arrowCount": 2
        },
    ],
    "parabaloid": [
        {
            "desc": "Collapse the indeps view",
            "test": "click",
            "selector": "rect#indeps",
            "button": "right"
        },
        {
            "desc": "Hit back button to uncollapse the indeps view",
            "test": "click",
            "selector": "#undo-graph",
            "button": "left"
        },
        {
            "desc": "Collapse the indeps view",
            "test": "click",
            "selector": "rect#indeps",
            "button": "right"
        },
        {
            "desc": "Zoom into the indeps view",
            "test": "click",
            "selector": "rect#indeps",
            "button": "left"
        },
        {
            "desc": "Uncollapse the indeps view",
            "test": "click",
            "selector": "rect#indeps",
            "button": "right"
        },
        {
            "desc": "There should be two elements visible in indeps view",
            "test": "uncollapse_zoomed_element",
            "selector": "rect#indeps",
            "n2ElementCount": 2
        }
    ],
    "nan_value": []
}

n2_gui_test_models = n2_gui_test_scripts.keys()


class n2_gui_test_case(unittest.TestCase):

    async def handle_console_err(self, msg):
        """ Invoked any time that an error or warning appears in the log. """
        if msg.type == 'warning':
            self.console_warning = True
            print('    Console Warning: ' + msg.text)
        elif msg.type == 'error':
            self.console_error = True
            print('    Console Error: ' + msg.text)

    async def handle_page_err(self, msg):
        self.page_error = True
        print('    Error on page: ', msg)

    def setup_error_handlers(self):
        self.console_warning = False
        self.console_error = False
        self.page_error = False

        self.page.on('console', lambda msg: self.handle_console_err(msg))
        self.page.on('pageerror', lambda msg: self.handle_page_err(msg))

    async def setup_browser(self):
        """ Create a browser instance and print user agent info. """
        self.browser = await pyppeteer.launch({
            'defaultViewport': {
                'width': 1600,
                'height': 900
            },
            'args': ['--start-fullscreen'],
            'headless': True
        })
        userAgentStr = await self.browser.userAgent()
        print("Browser: " + userAgentStr + "\n")

        self.page = await self.browser.newPage()
        await self.page.bringToFront()
        self.setup_error_handlers()

    def log_test(self, msg):
        global current_test

        """ Print a description and index for the test about to run. """
        print("  Test {:04}".format(current_test) + ": " + msg)
        current_test += 1

    def generate_n2_file(self):
        """ Generate N2 HTML files from all models in GUI_TEST_SUBDIR. """
        self.parentDir = os.path.dirname(os.path.realpath(__file__))
        self.modelDir = os.path.join(self.parentDir, GUI_TEST_SUBDIR)
        self.n2files = {}

        self.scripts = n2_gui_test_scripts
        self.known_model_names = n2_gui_test_models

        n2file = os.path.join(
            self.modelDir, self.current_model + GUI_N2_SUFFIX)
        pyfile = os.path.join(self.modelDir, self.current_model + '.py')
        self.n2files[self.current_model] = n2file
        print("Creating " + n2file)

        subprocess.run(
            ['openmdao', 'n2', '-o', n2file,  '--no_browser', pyfile],
            stderr=subprocess.PIPE, stdout=subprocess.PIPE)

    async def load_test_page(self):
        """ Load the specified HTML file from the local filesystem. """
        url = URL_PREFIX + '/' + self.n2_filename

        # Without waitUntil: 'networkidle0', processing will begin before
        # the page is fully rendered
        await self.page.goto(url, waitUntil='networkidle0')

        # Milliseconds to allow for the last transition animation to finish.
        # Obtain value defined in N2 code.
        self.transition_wait = \
            await self.page.evaluate("N2TransitionDefaults.durationSlow")
        self.transition_wait += 100
        print("  Transition wait set to " + str(self.transition_wait) + "ms")
        self.normal_wait = 10
        await self.page.waitFor(self.transition_wait)

    async def generic_toolbar_tests(self):
        """ Click most of the toolbar buttons to see if an error occurs """
        for test in toolbar_script:
            with self.subTest(test['desc']):
                self.log_test(test['desc'])
                btnHandle = await self.page.querySelector('#' + test['id'])
                await btnHandle.click(button='left', delay=5)
                waitTime = self.transition_wait if test['waitForTransition'] \
                    else self.normal_wait
                await self.page.waitFor(waitTime)

    async def assert_element_count(self, selector, expected_found):
        """
        Count the number of elements located by the selector and make
        sure it matches the supplied value.
        """
        hndl_list = await self.page.querySelectorAll(selector)

        self.assertIsNot(hndl_list, False,
                         "Could not find any '" + selector + "' elements.")
        self.assertEqual(len(hndl_list), expected_found,
                         'Found ' + str(len(hndl_list)) +
                         ' elements, expected ' + str(expected_found))

    async def assert_arrow_count(self, expected_arrows):
        """
        Count the number of path elements in the n2arrows < div > and make
        sure it matches the specified value.
        """
        await self.assert_element_count('g#n2arrows > g', expected_arrows)

    async def get_handle(self, selector):
        """ Get handle for a specific element and assert that it exists. """
        handle = await self.page.querySelector(selector)

        self.assertIsNotNone(handle,
                             "Could not find element with selector '" +
                             selector + "' in the N2 diagram.")

        return handle

    async def hover_and_check_arrow_count(self, options):
        """
        Hover over a matrix cell, make sure the number of expected arrows
        are there, then move off and make sure the arrows go away.
        """
        self.log_test(options['desc'] if 'desc' in options else
                      "Hover over '" + options['selector'] +
                      "' and checking arrow count")

        hndl = await self.get_handle(options['selector'])

        await hndl.hover()
        # Give it a chance to draw the arrows
        await self.page.waitFor(self.normal_wait)

        # Make sure there are enough arrows
        await self.assert_arrow_count(options['arrowCount'])
        await self.page.mouse.move(0, 0)  # Get the mouse off the element
        await self.page.waitFor(self.normal_wait)
        await self.assert_arrow_count(0)  # Make sure no arrows left

    async def click(self, options):
        """
        Perform a click of the type specified by options.button on the
        element specified by options.selector.
        """
        self.log_test(options['desc'] if 'desc' in options else
                      options['button'] + "-click on '" +
                      options['selector'] + "'")
        hndl = await self.get_handle(options['selector'])
        await hndl.click(button=options['button'])
        await self.page.waitFor(self.transition_wait)

    async def return_to_root(self):
        """
        Left-click the home button and wait for the transition to complete.
        """

        self.log_test("Return to root")
        hndl = await self.get_handle("#reset-graph")
        await hndl.click()
        await self.page.waitFor(self.transition_wait * 2)

    async def search_and_check_result(self, options):
        """
        Enter a string in the search textbox and check that the expected
        number of elements are shown in the N2 matrix.
        """
        searchString = options['searchString']
        self.log_test(options['desc'] if 'desc' in options else
                      "Searching for '" + options['searchString'] +
                      "' and checking for " +
                      str(options['n2ElementCount']) + " N2 elements after.")

        # await self.page.hover(".searchbar-container")
        await self.page.click("#searchbar-container")
        await self.page.waitFor(500)

        searchbar = await self.page.querySelector('#awesompleteId')
        await searchbar.type(searchString + "\n")

        # await self.page.waitFor(500)

        # await self.page.keyboard.press('Backspace')
        # await self.page.keyboard.press("Enter")
        await self.page.waitFor(self.transition_wait + 500)
        await self.assert_element_count("g#n2elements > g.n2cell",
                                        options['n2ElementCount'])

    async def run_model_script(self, script):
        """
        Iterate through the supplied script array and perform each
        action/test.
        """

        print("Performing diagram-specific tests...")
        await self.page.reload(waitUntil='networkidle0')
        await self.page.waitFor(self.transition_wait)

        for script_item in script:
            if 'test' not in script_item:
                continue

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
                self.log_test(script_item['desc'] if 'desc' in script_item
                              else "Checking for " + str(script_item['count']) +
                              "' instances of '" + script_item['selector'] + "'")
                await self.assert_element_count(script_item['selector'],
                                                script_item['count'])

    async def run_gui_tests(self):
        """ Execute all of the tests in an async event loop. """
        await self.setup_browser()

        self.n2_filename = self.n2files[self.current_model]
        await self.load_test_page()
        await self.generic_toolbar_tests()

        bname = os.path.basename(self.n2_filename)[:-len(GUI_N2_SUFFIX)]

        if bname in self.known_model_names:
            await self.run_model_script(self.scripts[bname])

        await self.browser.close()

        if self.console_error:
            msg = "Console log contains errors."
            print(msg)
            self.fail(msg)

        if self.page_error:
            msg = "There were errors on the page."
            print(msg)
            self.fail(msg)

    @parameterized.expand(n2_gui_test_models)
    def test_n2_gui(self, basename):
        if (basename[:2] == "__"):
            return

        print("\n" + LINE_STR + "\n" + basename + "\n" + LINE_STR)

        self.current_test_desc = ''
        self.current_model = basename
        self.generate_n2_file()
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        coro = asyncio.coroutine(self.run_gui_tests)
        self.event_loop.run_until_complete(coro())
        self.event_loop.close()

        if not DEBUG:
            try:
                for n2html in self.n2files:
                    os.remove(n2html)
            except:
                # Don't want the test to fail if the test file is
                # already removed
                pass
