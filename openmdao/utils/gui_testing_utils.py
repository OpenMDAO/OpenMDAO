"""Define utils for use in testing GUIs with Playwright."""
import unittest

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
        print(msg)

    async def get_handle(self, selector):
        """ Get handle for a specific element and assert that it exists. """
        handle = await self.page.wait_for_selector(selector, state='attached', timeout=3055)

        self.assertIsNotNone(handle,
                             "Could not find element with selector '" +
                             selector + "' in the N2 diagram.")

        return handle

