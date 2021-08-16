"""Define utils for use in testing GUIs with Playwright."""
import unittest

import asyncio
# from playwright.async_api import async_playwright

current_test = 1


class gui_test_case(unittest.TestCase):

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

    async def setup_browser(self, playwright):
        """ Create a browser instance and print user agent info. """
        self.browser = await playwright.chromium.launch(args=['--start-fullscreen'])
        self.page = await self.browser.new_page()

        await self.page.bring_to_front()
        self.setup_error_handlers()

    def log_test(self, msg):
        global current_test

        """ Print a description and index for the test about to run. """
        print("  Test {:04}".format(current_test) + ": " + msg)
        current_test += 1

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

    async def get_handle(self, selector):
        """ Get handle for a specific element and assert that it exists. """
        handle = await self.page.wait_for_selector(selector, state='attached', timeout=3055)

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

        await hndl.hover(force=False)

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

    async def drag(self, options):
        """
        Hover over the element, perform a mousedown event, move the mouse to the
        specified location, and perform a mouseup. Check to make sure the element
        moved in at least one direction.
        """
        self.log_test(options['desc'] if 'desc' in options else
                      "Dragging '" + options['selector'] + "' to " + options['x'] + "," + options[
                          'y'])

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
