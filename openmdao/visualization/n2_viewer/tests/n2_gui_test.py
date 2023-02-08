"""Test N2 GUI with multiple models using Playwright."""
import asyncio
from playwright.async_api import async_playwright
import subprocess
from aiounittest import async_test
import os
import sys

from openmdao.utils.gui_testing_utils import _GuiTestCase

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

if 'win32' in sys.platform:
    # Windows specific event-loop policy & cmd
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

my_loop = asyncio.get_event_loop()

""" A set of toolbar tests that runs on each model. """
toolbar_script = [
    {
        "desc": "Uncollapse All button",
        "id": "expand-all",
    },
    {
        "desc": "Collapse Outputs in View Only button",
        "id": "collapse-element-2",
    },
    {
        "desc": "Uncollapse In View Only button",
        "id": "expand-element",
    },
    {
        "desc": "Show Legend (off) button",
        "id": "legend-button",
    },
    {
        "desc": "Show Legend (on) button",
        "id": "legend-button",
    },
    {
        "desc": "Show Path (on) button",
        "id": "info-button",
    },
    {
        "desc": "Show Path (off) button",
        "id": "info-button",
    },
    {
        "desc": "Non-linear solver names button",
        "id": "non-linear-solver-button",
    },
    {
        "desc": "Linear solver names button",
        "id": "linear-solver-button-2",
    },
    {
        "desc": "Toggle solver visibility button (off)",
        "id": "no-solver-button",
    },
    {
        "desc": "Toggle solver visibility button (on)",
        "id": "no-solver-button",
    },
    {
        "desc": "Clear Arrows and Connection button",
        "id": "hide-connections",
    },
    {
        "desc": "Help (on) button",
        "id": "question-button",
    },
    {
        "desc": "Help (off) button",
        "id": "question-button",
    },
    {
        "desc": "Collapse All Outputs button",
        "id": "collapse-all",
    }
]

""" A dictionary of tests script with an array for each model."""
n2_gui_test_scripts = {
    "circuit": [
        {"test": "toolbar"},
        {
            "desc": "Hover on N2 matrix element and check arrow count",
            "test": "hoverArrow",
            "selector": "g#n2elements rect#cellShape_node_33.vMid",
            "arrowCount": 4
        },
        {
            "desc": "Hover on an N2 cell with cycle arrows and count",
            "test": "hoverArrow",
            "selector": "#cellShape_conn_33_to_13",
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
            "selector": "g#n2elements rect#cellShape_node_33.vMid",
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
            "selector": "g#n2elements rect#cellShape_node_10.gMid",
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
            "selector": "g#n2elements rect#cellShape_node_15.vMid",
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
            "selector": "g#n2elements rect#cellShape_node_10.gMid",
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
        },
        { "test": "root" },
        {
            "desc": "Alt-right-click the n1 component",
            "test": "click",
            "selector": "rect#circuit_n1",
            "button": "right",
            "modifiers": [ "Alt" ]
        },
        {
            "desc": "Check that variable selection dialog appears",
            "test": "count",
            "selector": "#childSelect-circuit_n1",
            "count": 1
        },
        {
            "desc": "Select a variable to hide",
            "test": "click",
            "selector": "input#circuit_n1_V-visible-check",
            "button": "left"
        },
        {
            "desc": "Click the Apply button",
            "test": "click",
            "selector": ".button-container button:last-child",
            "button": "left"
        },
        {
            "desc": "Check that the V variable is no longer displayed",
            "test": "count",
            "selector": "rect#circuit_n1_V",
            "count": 0
        },
        {
            "desc": "Alt-right-click the n1 component",
            "test": "click",
            "selector": "rect#circuit_n1",
            "button": "right",
            "modifiers": [ "Alt" ]
        },
        {
            "desc": "Click the Clear Search button",
            "test": "click",
            "selector": ".search-clear",
            "button": "left"
        },
        {
            "desc": "Perform a search in the variable selection dialog",
            "test": "var_select_search",
            "searchString": "in",
            "foundVariableCount": 1
        },
        {
            "desc": "Click the Apply button",
            "test": "click",
            "selector": ".button-container button:last-child",
            "button": "left"
        },
        {
            "desc": "Check the number of cells in the N2 Matrix",
            "test": "count",
            "selector": "g#n2elements > g.n2cell",
            "count": 36
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
            "selector": "g#n2elements rect#cellShape_node_14.vMid",
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
            "selector": "g#n2elements rect#cellShape_node_19.vMid",
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
            "selector": "g#n2elements rect#cellShape_node_18.gMid",
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
            "selector": "g#n2elements rect#cellShape_node_19.vMid",
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
            "selector": "g#n2elements rect#cellShape_node_18.gMid",
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
            "selector": "g#n2elements rect#cellShape_node_12.vMid",
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
            "selector": "g#n2elements rect#cellShape_node_18.vMid",
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
            "selector": "g#n2elements rect#cellShape_node_21.gMid",
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
            "selector": "g#n2elements rect#cellShape_node_32.vMid",
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
            "selector": "g#n2elements rect#cellShape_node_7.gMid",
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
            "selector": "g#n2elements rect#cellShape_node_23.vMid",
            "arrowCount": 2
        },
        {"test": "root"},
        {
            "desc": "Expand solver selector toolbar group",
            "test": "hover",
            "selector": "div.group-3 > div.expandable:nth-child(2)"
        },
        {
            "desc": "Click on non-linear solver toolbar button",
            "test": "click",
            "button": "left",
            "selector": "#non-linear-solver-button"
        },
        {
            "desc": "Check that solver names have been updated",
            "test": "hover",
            "selector": 'g.solver_nl_nlbgs:has-text("NL: NLBGS") > rect'
        }
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
            "desc": "There should be two elements visible in indeps view",
            "test": "count",
            "selector": "g#n2elements > g.n2cell",
            "count": 2
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
            "selector": "text=Show more",
            "button": "left"
        },
        {
            "desc": "Drag Value Info window to new location",
            "test": "drag",
            "selector": '[id^="persistentNodeInfo"]  .window-draggable-ribbon:first-child',
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
        # Begin multi-window close test
        {
            "desc": "Hover to bring up Node Info window for p1.A",
            "test": "hover",
            "selector": "#p1_A",
        },
        {
            "desc": "Hover to bring up Node Info window for lingrp.lin.x",
            "test": "hover",
            "selector": "#lingrp_lin_x",
        },
        {
            "desc": "Click lingrp.lin.x to make Node Info window persistent",
            "test": "click",
            "selector": "#lingrp_lin_x",
            "button": "left"
        },
        {
            "desc": "Hover to bring up Node Info window for lingrp.lin.A",
            "test": "hover",
            "selector": "#lingrp_lin_A",
        },
        {
            "desc": "Click lingrp.lin.A to make Node Info window persistent",
            "test": "click",
            "selector": "#lingrp_lin_A",
            "button": "left"
        },
        {
            "desc": "Click p1.A to make Node Info window persistent",
            "test": "click",
            "selector": "#p1_A",
            "button": "left"
        },
        {
            "desc": "Check for 3 open persistent Node Info windows",
            "test": "count",
            "selector": "[id^='persistentNodeInfo']",
            "count": 3
        },
        {
            "desc": "Close All Node Info windows",
            "test": "click",
            "selector": '[id^="persistentNodeInfo"] span.window-close-button',
            "button": "left",
            "modifiers": ["Shift"]
        },
        {
            "desc": "Check for 0 open persistent Node Info windows",
            "test": "count",
            "selector": "[id^='persistentNodeInfo']",
            "count": 0
        },
        # End multi-window close test
        {
            "desc": "Turn off Node Info mode",
            "test": "click",
            "selector": "#info-button",
            "button": "left"
        },
        {
            "desc": "Expand help group",
            "test": "hover",
            "selector": "#question-button"
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
            "selector": "#n2win-legend .window-draggable-ribbon:first-child",
            "x": 700, "y": 500
        },
        {
            "desc": "Hide Legend",
            "test": "click",
            "selector": "#n2win-legend span.window-close-button",
            "button": "left"
        },
        {
            "desc": "Expand target click group",
            "test": "hover",
            "selector": "#info-button"
        },
        {
            "desc": "Toggle expand/collapse mode on",
            "test": "click",
            "selector": "#collapse-target",
            "button": "left"
        },
        {
            "desc": "Collapse the lingrp group",
            "test": "click",
            "selector": "g#tree rect#lingrp",
            "button": "left"
        },
        {
            "desc": "Verify size of model tree",
            "test": "count",
            "selector": "g#tree > g.model_tree_grp",
            "count": 6
        },
        {
            "desc": "Expand the lingrp group",
            "test": "click",
            "selector": "g#tree rect#lingrp",
            "button": "left"
        },
        {
            "desc": "Verify size of model tree",
            "test": "count",
            "selector": "g#tree > g.model_tree_grp",
            "count": 10
        },
        {
            "desc": "Toggle expand/collapse mode off",
            "test": "click",
            "selector": "#collapse-target",
            "button": "left"
        },
        {
            "desc": "Expand target click group",
            "test": "hover",
            "selector": "#collapse-target"
        },
        {
            "desc": "Toggle filter mode on",
            "test": "click",
            "selector": "#filter-target",
            "button": "left"
        },
        {
            "desc": "Click the lin component, open variable dialog",
            "test": "click",
            "selector": "rect#lingrp_lin",
            "button": "left"
        },
        {
            "desc": "Close the lin variable dialog",
            "test": "click",
            "selector": "#childSelect-lingrp_lin span.window-close-button",
            "button": "left"
        },
        {
            "desc": "Toggle filter mode off",
            "test": "click",
            "selector": "#filter-target",
            "button": "left"
        },   
    ]
}

n2_gui_test_models = n2_gui_test_scripts.keys()


class n2_gui_test_case(_GuiTestCase):

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

        n2file = os.path.join(self.modelDir, self.current_model + GUI_N2_SUFFIX)
        pyfile = os.path.join(self.modelDir, self.current_model + '.py')
        self.n2files[self.current_model] = n2file
        print("Creating " + n2file)

        cmd = ['openmdao', 'n2', '-o', n2file,  '--no_browser', pyfile]
        subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)  # nosec: trusted input

    async def load_test_page(self):
        """ Load the specified HTML file from the local filesystem. """
        url = URL_PREFIX + '/' + self.n2_filename

        # Without wait_until: 'networkidle', processing will begin before
        # the page is fully rendered
        await self.page.goto(url, wait_until='networkidle')

    async def generic_toolbar_tests(self):
        """ Click most of the toolbar buttons to see if an error occurs """
        for test in toolbar_script:
            with self.subTest(test['desc']):
                self.log_test("[Toolbar] " + test['desc'])

                btnHandle = await self.get_handle('#' + test['id'])
                await btnHandle.click(button='left', timeout=3333, force=True)

        await self.page.reload(wait_until='networkidle')

    async def assert_element_count(self, selector, expected_found):
        """
        Count the number of elements located by the selector and make
        sure it exactly matches the supplied value. Try several times
        because sometimes transition animations throw things off.
        """
        max_tries = 3  # Max number of times to attempt to find a selector
        max_time = 2000  # The timeout in ms for each search

        if (expected_found > 0):
            num_tries = 0
            found = False
            while (not found and num_tries < max_tries):
                nth_selector = f':nth-match({selector}, {expected_found})'
                try:
                    await self.page.wait_for_selector(nth_selector, state='attached',
                                                      timeout=max_time)
                    found = True
                except:
                    num_tries += 1

            num_tries = 0
            found = False
            while (not found and num_tries < max_tries):
                nth_selector = f':nth-match({selector}, {expected_found + 1})'
                try:
                    await self.page.wait_for_selector(nth_selector, state='detached',
                                                      timeout=max_time)
                    found = True
                except:
                    num_tries += 1

        else:
            num_tries = 0
            found = False
            while (not found and num_tries < max_tries):
                nth_selector = f':nth-match({selector}, 1)'
                try:
                    await self.page.wait_for_selector(nth_selector, state='detached',
                                                      timeout=max_time)
                    found = True
                except:
                    num_tries += 1

        hndl_list = await self.page.query_selector_all(selector)
        if (len(hndl_list) > expected_found):
            global current_test
            await self.page.screenshot(path=f'shot_{current_test}.png')

        self.assertEqual(len(hndl_list), expected_found,
                         'Found ' + str(len(hndl_list)) +
                         ' elements, expected ' + str(expected_found))

    async def assert_arrow_count(self, expected_arrows):
        """
        Count the number of path elements in the n2arrows < div > and make
        sure it matches the specified value.
        """
        await self.assert_element_count('g#n2arrows > g', expected_arrows)

    async def hover(self, options, log_test=True):
        """
        Hover over the specified element.
        """
        if log_test:
            self.log_test(options['desc'] if 'desc' in options else
                          "Hover over '" + options['selector'] + "'")

        hndl = await self.get_handle(options['selector'])

        await hndl.hover(force=False)

    async def hover_and_check_arrow_count(self, options):
        """
        Hover over a matrix cell, make sure the number of expected arrows
        are there, then move off and make sure the arrows go away.
        """
        await self.hover(options)

        # Make sure there are enough arrows
        await self.assert_arrow_count(options['arrowCount'])
        await self.page.mouse.move(0, 0)  # Get the mouse off the element
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

        mod_keys = [] if 'modifiers' not in options else options['modifiers']
        await hndl.click(button=options['button'], modifiers=mod_keys)

    async def drag(self, options):
        """
        Hover over the element, perform a mousedown event, move the mouse to the
        specified location, and perform a mouseup. Check to make sure the element
        moved in at least one direction.
        """
        self.log_test(options['desc'] if 'desc' in options else
                      "Dragging '" + options['selector'] + "' to " + options['x'] + "," + options['y'])

        hndl = await self.get_handle(options['selector'])

        pre_drag_bbox = await hndl.bounding_box()

        await hndl.hover(force=True)
        await self.page.mouse.down()
        await self.page.mouse.move(options['x'], options['y'])
        await self.page.mouse.up()

        post_drag_bbox = await hndl.bounding_box()

        moved = ((pre_drag_bbox['x'] != post_drag_bbox['x']) or
                 (pre_drag_bbox['y'] != post_drag_bbox['y']))

        self.assertIsNot(moved, False,
                         "The '" + options['selector'] + "' element did not move.")

    async def resize_window(self, options):
        """
        Drag an edge/corner of a WindowResizable obj and check that the size changed
        or didn't change as expected.
        """
        self.log_test(options['desc'] if 'desc' in options else
                      "Resizing '" + options['selector'] + "' window.")

        win_hndl = await self.get_handle(options['selector'])
        pre_resize_bbox = await win_hndl.bounding_box()

        edge_hndl = await self.get_handle(options['selector'] + ' div.rsz-' + options['side'])
        edge_bbox = await edge_hndl.bounding_box()

        new_x = edge_bbox['x'] + \
            resize_dirs[options['side']][0] * options['distance']
        new_y = edge_bbox['y'] + \
            resize_dirs[options['side']][1] * options['distance']

        await edge_hndl.hover()
        await self.page.mouse.down()
        await self.page.mouse.move(new_x, new_y)
        await self.page.mouse.up()

        post_resize_bbox = await win_hndl.bounding_box()
        dw = post_resize_bbox['width'] - pre_resize_bbox['width']
        dh = post_resize_bbox['height'] - pre_resize_bbox['height']

        resized = ((dw != 0) or (dh != 0))
        if options['expectChange']:
            self.assertIsNot(resized, False,
                             "The '" + options['selector'] + "' element was NOT resized and should have been.")
        else:
            self.assertIsNot(resized, True,
                             "The '" + options['selector'] + "' element was resized and should NOT have been.")

    async def return_to_root(self):
        """
        Left-click the home button and wait for the transition to complete.
        """

        self.log_test("Return to root")
        hndl = await self.get_handle("#reset-graph")
        await hndl.click()

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

        await self.page.click("#searchbar-container")

        searchbar = await self.page.wait_for_selector('#awesompleteId', state='visible')
        await searchbar.type(searchString + "\n", delay=50)

        await self.assert_element_count("g.n2cell", options['n2ElementCount'])

    async def var_select_search_and_check_result(self, options):
        """
        Enter a string in the variable selection search textbox and check the result.
        """
        searchString = options['searchString']
        self.log_test(options['desc'] if 'desc' in options else
                      "Searching for '" + options['searchString'] +
                      "' and checking for " +
                      str(options['foundVariableCount']) + " table rows after.")

        searchbar = await self.page.wait_for_selector('.search-container input', state='visible')
        await searchbar.type(searchString + "\n", delay=50)

        await self.assert_element_count("td.varname", options['foundVariableCount'])

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
            elif test_type == 'var_select_search':
                await self.var_select_search_and_check_result(script_item)
            elif test_type == 'toolbar':
                await self.generic_toolbar_tests()
            elif test_type == 'count':
                self.log_test(script_item['desc'] if 'desc' in script_item
                              else "Checking for " + str(script_item['count']) +
                              "' instances of '" + script_item['selector'] + "'")
                await self.assert_element_count(script_item['selector'],
                                                script_item['count'])

    async def run_gui_tests(self, playwright):
        """ Execute all of the tests in an async event loop. """
        await self.setup_browser(playwright)

        self.n2_filename = self.n2files[self.current_model]
        await self.load_test_page()
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

        async with async_playwright() as playwright:
            await self.run_gui_tests(playwright)

        if not DEBUG:
            try:
                for n2html in self.n2files:
                    os.remove(self.n2files[n2html])
            except OSError:
                # Don't want the test to fail if the test file is
                # already removed
                pass
