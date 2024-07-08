"""Test N2 GUI with multiple models using Playwright."""
import asyncio
from playwright.async_api import async_playwright
import subprocess
from aiounittest import async_test
import os
import sys

from openmdao.utils.om_warnings import issue_warning

from openmdao.utils.gui_testing_utils import _GuiTestCase

# set DEBUG to True if you want to view the generated HTML file
GUI_DIAG_SUFFIX = '_GEN_TEST.html'
GUI_TEST_SUBDIR = 'gui_test_models'
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

my_loop = asyncio.get_event_loop_policy().get_event_loop()

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
gen_gui_test_scripts = {
    "gen_diag": [
        {"test": "toolbar"},
        {
            "desc": "Hover on matrix element and check arrow count",
            "test": "hoverArrow",
            "selector": "g#n2elements rect#cellShape_node_34.vMid",
            "arrowCount": 1
        },
        {
            "desc": "Left-click on model tree element to zoom",
            "test": "click",
            "selector": "g#tree rect#grp_two_child_one",
            "button": "left"
        },
        {
            "desc": "Hover on matrix element and check arrow count",
            "test": "hoverArrow",
            "selector": "g#n2elements rect#cellShape_node_34.vMid",
            "arrowCount": 1
        },
        {"test": "root"},
        {
            "desc": "Right-click on model tree element to collapse",
            "test": "click",
            "selector": "g#tree rect#grp_two_child_two",
            "button": "right"
        },
        {
            "desc": "Hover over collapsed matrix element and check arrow count",
            "test": "hoverArrow",
            "selector": "g#n2elements rect#cellShape_node_38.gMid",
            "arrowCount": 2
        },
        {
            "desc": "Right-click on model tree element to uncollapse",
            "test": "click",
            "selector": "g#tree rect#grp_two_child_two",
            "button": "right"
        },
        {"test": "root"},
        {
            "desc": "Check the number of cells in the Matrix",
            "test": "count",
            "selector": "g#n2elements > g.n2cell",
            "count": 53
        },
        {
            "desc": "Perform a search on output_2_3",
            "test": "search",
            "searchString": "output_2_3",
            "n2ElementCount": 9
        },
        {"test": "root"},
        {
            "desc": "Check that home button works after search",
            "test": "count",
            "selector": "g#n2elements > g.n2cell",
            "count": 53
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
            "count": 10
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
    ]
}

gen_gui_test_models = gen_gui_test_scripts.keys()


class gen_gui_test_case(_GuiTestCase):

    def log_test(self, msg):
        global current_test

        """ Print a description and index for the test about to run. """
        print("  Test {:04}".format(current_test) + ": " + msg)
        current_test += 1

    def generate_html_file(self):
        """ Generate HTML file for a generic model. """
        self.parentDir = os.path.dirname(os.path.realpath(__file__))
        self.outputDir = os.path.join(self.parentDir, GUI_TEST_SUBDIR)
        self.scripts = gen_gui_test_scripts

        self.diagram_file = os.path.join(self.outputDir, f"gen_diag{GUI_DIAG_SUFFIX}")
        pyfile = os.path.join(self.parentDir, 'create_generic_model.py')
        print("Creating " + self.diagram_file)

        cmd = ['python', pyfile, self.diagram_file]
        cp = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)  # nosec: trusted input

        if (cp.returncode != 0):
            raise RuntimeError(f"Failed to create HTML file with generic model ({cp.stderr}).")

    async def load_test_page(self):
        """ Load the specified HTML file from the local filesystem. """
        url = URL_PREFIX + self.diagram_file

        # Without wait_until: 'networkidle', processing will begin before
        # the page is fully rendered
        await self.page.goto(url, wait_until='networkidle')

    async def generic_toolbar_tests(self):
        """ Click most of the toolbar buttons to see if an error occurs """
        for test in toolbar_script:
            with self.subTest(test['desc']):
                self.log_test("[Toolbar] " + test['desc'])

                btnHandle = await self.get_handle('#' + test['id'])
                try:
                    await btnHandle.click(button='left', timeout=3333, force=True)
                except Exception as err:
                    if "Element is outside of the viewport" in str(err):
                        issue_warning(str(err))
                    else:
                        raise(err)

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
        number of elements are shown in the matrix.
        """
        searchString = options['searchString']
        self.log_test(options['desc'] if 'desc' in options else
                      "Searching for '" + options['searchString'] +
                      "' and checking for " +
                      str(options['n2ElementCount']) + " elements after.")

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

        await self.load_test_page()
        bname = os.path.basename(self.diagram_file)[:-len(GUI_DIAG_SUFFIX)]

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

    @async_test(loop=my_loop)
    async def test_gen_gui(self, basename = "create_generic_model.py"):
        if (basename[:2] == "__"):
            return

        print("\n" + LINE_STR + "\n" + basename + "\n" + LINE_STR)

        self.current_test_desc = ''
        self.current_model = basename

        self.generate_html_file()

        async with async_playwright() as playwright:
            await self.run_gui_tests(playwright)

        if not DEBUG:
            try:
                os.remove(self.diagram_file)
            except OSError:
                # Don't want the test to fail if the test file is
                # already removed
                pass
