"""Test N2 GUI with multiple models using Pyppeteer."""
import asyncio
import pyppeteer
import subprocess
import unittest
from aiounittest import async_test
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

resize_dirs = {
    'top': [0, -1],
    'top-right': [1, -1],
    'right': [1, 0],
    'bottom-right': [1, 1],
    'bottom': [0, 1],
    'bottom-left': [-1, 1],
    'left': [-1, 0],
    'top-left': [-1, -1]
}

my_loop = asyncio.get_event_loop()

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
        {"test": "toolbar"},
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
        {"test": "root"},
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
        {"test": "root"},
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
        {"test": "root"},
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
        {"test": "root"},
        {
            "desc": "Check that home button works after search",
            "test": "count",
            "selector": "g#n2elements > g.n2cell",
            "count": 40
        },
        {
            "desc": "Expand toolbar connections menu",
            "test": "hover",
            "selector": ".group-3 > div.expandable:first-child"
        },
        {
            "desc": "Press toolbar show all connections button",
            "test": "click",
            "selector": "#show-all-connections",
            "button": "left"
        },
        {
            "desc": "Check number of arrows",
            "test": "count",
            "selector": "g#n2arrows > g",
            "count": 44
        },
        {
            "desc": "Expand toolbar connections menu",
            "test": "hover",
            "selector": ".group-3 > div.expandable:first-child"
        },
        {
            "desc": "Press toolbar hide all connections button",
            "test": "click",
            "selector": "#hide-connections-2",
            "button": "left"
        },
        {
            "desc": "Check number of arrows",
            "test": "count",
            "selector": "g#n2arrows > g",
            "count": 0
        }
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
        {"test": "root"},
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
        {"test": "root"},
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
        {"test": "root"},
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
        {"test": "root"},
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
    "nan_value": [
        {"test": "toolbar"}
    ],
    "valuewin2": [
        {
            "desc": "Turn on Node Info mode",
            "test": "click",
            "selector": "#info-button",
            "button": "left"
        },
        {
            "desc": "Hover to bring up Node Info window",
            "test": "hover",
            "selector": "#lingrp_lin_A",
        },
        {
            "desc": "Click variable to make Node Info window persistent",
            "test": "click",
            "selector": "#lingrp_lin_A",
            "button": "left"
        },
        {
            "desc": "Display Value Info window",
            "test": "click",
            "selector": '[id^="persistentNodeInfo"] button#value',
            "button": "left"
        },
        {
            "desc": "Drag Value Info window to new location",
            "test": "drag",
            "selector": '[id^="persistentNodeInfo"]  .window-draggable-header',
            "x": 700, "y": 700
        },
        {
            "desc": "Don't enlarge the Value Info window beyond its max width",
            "test": "resize",
            "selector": '[id^="valueInfo"]',
            "side": "right",
            "distance": 100,
            "expectChange": False
        },
        {
            "desc": "Resize the Value Info window smaller",
            "test": "resize",
            "selector": '[id^="valueInfo"]',
            "side": "right",
            "distance": -100,
            "expectChange": True
        },
        {
            "desc": "Close Value Info window",
            "test": "click",
            "selector": '[id^="valueInfo"] span.window-close-button',
            "button": "left"
        },
        {
            "desc": "Close Node Info window",
            "test": "click",
            "selector": '[id^="persistentNodeInfo"] span.window-close-button',
            "button": "left"
        },
        {
            "desc": "Turn off Node Info mode",
            "test": "click",
            "selector": "#info-button",
            "button": "left"
        },
        {
            "desc": "Display Legend",
            "test": "click",
            "selector": "#legend-button",
            "button": "left"
        },
        {
            "desc": "Drag Legend to new location",
            "test": "drag",
            "selector": "#n2win-legend .window-draggable-header",
            "x": 700, "y": 500
        },
        {
            "desc": "Hide Legend",
            "test": "click",
            "selector": "#n2win-legend span.window-close-button",
            "button": "left"
        }
    ]
}

n2_gui_test_models = n2_gui_test_scripts.keys()


class n2_gui_test_case(unittest.TestCase):

    def handle_console_err(self, msg):
        """ Invoked any time that an error or warning appears in the log. """
        if msg.type == 'warning':
            self.console_warning = True
            print('    Console Warning: ' + msg.text)
        elif msg.type == 'error':
            self.console_error = True
            print('    Console Error: ' + msg.text)

    def handle_page_err(self, msg):
        self.page_error = True
        print('    Error on page: ', msg)
        print(type(msg))

    def handle_request_err(self, msg):
        self.page_error = True
        print('    Request error: ', msg)

    def setup_error_handlers(self):
        self.console_warning = False
        self.console_error = False
        self.page_error = False

        self.page.on('console', lambda msg: self.handle_console_err(msg))
        self.page.on('pageerror', lambda msg: self.handle_page_err(msg))
        self.page.on('requestfailed', lambda msg: self.handle_request_err(msg))

    async def setup_browser(self):
        """ Create a browser instance and print user agent info. """
        self.browser = await pyppeteer.launch({
            'defaultViewport': {
                'width': 1600,
                'height': 900
            },
            'args': [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--start-fullscreen'
            ],
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
                self.log_test("[Toolbar] " + test['desc'])
                btnHandle = await self.page.querySelector('#' + test['id'])
                await btnHandle.click(button='left', delay=5)
                waitTime = self.transition_wait if test['waitForTransition'] \
                    else self.normal_wait
                await self.page.waitFor(waitTime)

        await self.page.reload(waitUntil='networkidle0')
        await self.page.waitFor(self.transition_wait)

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

    async def hover(self, options, log_test=True):
        """
        Hover over the specified element.
        """
        if log_test:
            self.log_test(options['desc'] if 'desc' in options else
                          "Hover over '" + options['selector'] + "'")

        hndl = await self.get_handle(options['selector'])

        await hndl.hover()

        # Give the browser a chance to do whatever
        await self.page.waitFor(self.normal_wait)

    async def hover_and_check_arrow_count(self, options):
        """
        Hover over a matrix cell, make sure the number of expected arrows
        are there, then move off and make sure the arrows go away.
        """
        await self.hover(options)

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

    async def drag(self, options):
        """
        Hover over the element, perform a mousedown event, move the mouse to the
        specified location, and perform a mouseup. Check to make sure the element
        moved in at least one direction.
        """
        self.log_test(options['desc'] if 'desc' in options else
                      "Dragging '" + options['selector'] + "' to " + options['x'] + "," + options['y'])

        hndl = await self.get_handle(options['selector'])

        pre_drag_bbox = await hndl.boundingBox()

        await hndl.hover()
        await self.page.mouse.down()
        await self.page.mouse.move(options['x'], options['y'])
        await self.page.mouse.up()

        await self.page.waitFor(self.normal_wait)
        post_drag_bbox = await hndl.boundingBox()

        moved = ((pre_drag_bbox['x'] != post_drag_bbox['x']) or
                 (pre_drag_bbox['y'] != post_drag_bbox['y']))

        self.assertIsNot(moved, False,
                         "The '" + options['selector'] + "' element did not move.")

    async def resize_window(self, options):
        """
        Drag an edge/corner of an N2WindowResizable and check that the size changed
        or didn't change as expected.
        """
        self.log_test(options['desc'] if 'desc' in options else
                      "Resizing '" + options['selector'] + "' window.")

        # await self.page.screenshot({'path': 'preresize.png'})

        win_hndl = await self.get_handle(options['selector'])
        pre_resize_bbox = await win_hndl.boundingBox()

        edge_hndl = await self.get_handle(options['selector'] + ' div.rsz-' + options['side'])
        edge_bbox = await edge_hndl.boundingBox()

        new_x = edge_bbox['x'] + \
            resize_dirs[options['side']][0] * options['distance']
        new_y = edge_bbox['y'] + \
            resize_dirs[options['side']][1] * options['distance']

        await edge_hndl.hover()
        await self.page.mouse.down()
        await self.page.mouse.move(new_x, new_y)
        await self.page.mouse.up()

        post_resize_bbox = await win_hndl.boundingBox()
        dw = post_resize_bbox['width'] - pre_resize_bbox['width']
        dh = post_resize_bbox['height'] - pre_resize_bbox['height']

        resized = ((dw != 0) or (dh != 0))
        if options['expectChange']:
            self.assertIsNot(resized, False,
                             "The '" + options['selector'] + "' element was NOT resized and should have been.")
        else:
            self.assertIsNot(resized, True,
                             "The '" + options['selector'] + "' element was resized and should NOT have been.")

        # await self.page.screenshot({'path': 'postresize.png'})

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

        print("Running tests from model script...")

        for script_item in script:
            if 'test' not in script_item:
                continue

            test_type = script_item['test']
            if test_type == 'hoverArrow':
                await self.hover_and_check_arrow_count(script_item)
            elif test_type == 'hover':
                await self.hover(script_item)
            elif test_type == 'click':
                await self.click(script_item)
            elif test_type == 'drag':
                await self.drag(script_item)
            elif test_type == 'resize':
                await self.resize_window(script_item)
            elif test_type == 'root':
                await self.return_to_root()
            elif test_type == 'search':
                await self.search_and_check_result(script_item)
            elif test_type == 'toolbar':
                await self.generic_toolbar_tests()
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
        await self.page.waitFor(2000)
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
    @async_test(loop=my_loop)
    async def test_n2_gui(self, basename):
        if (basename[:2] == "__"):
            return

        print("\n" + LINE_STR + "\n" + basename + "\n" + LINE_STR)

        self.current_test_desc = ''
        self.current_model = basename
        self.generate_n2_file()
        await self.run_gui_tests()

        if not DEBUG:
            try:
                for n2html in self.n2files:
                    os.remove(self.n2files[n2html])
            except:
                # Don't want the test to fail if the test file is
                # already removed
                pass
