"""Define utils for use in testing GUIs with Playwright."""

import contextlib
import socket
import unittest


class _GuiTestCase(unittest.TestCase):
    """
    Class that can be inherited from and used for GUI tests using Playwright.
    """

    def handle_console_err(self, msg):
        """
        Invoke any time that an error or warning appears in the log.

        Parameters
        ----------
        msg : string
            message string from console.
        """
        if msg.type == 'warning':
            self.console_warning = True
            print('    Console Warning: ' + msg.text)
        elif msg.type == 'error':
            self.console_error = True
            print('    Console Error: ' + msg.text)

    def handle_page_err(self, msg):
        """
        Invoke any time there is a page error.

        Parameters
        ----------
        msg : string
            error message string.
        """
        self.page_error = True
        print('    Error on page: ', msg)
        print(type(msg))

    def handle_request_err(self, msg):
        """
        Invoke any time there is a request error.

        Parameters
        ----------
        msg : string
            error message string.
        """
        self.page_error = True
        print('    Request error: ', msg)

    def setup_error_handlers(self):
        """
        Set up the error handlers.

        """
        self.console_warning = False
        self.console_error = False
        self.page_error = False

        self.page.on('console', lambda msg: self.handle_console_err(msg))
        self.page.on('pageerror', lambda msg: self.handle_page_err(msg))
        self.page.on('requestfailed', lambda msg: self.handle_request_err(msg))

    async def setup_browser(self, playwright):
        """
        Create a browser instance and print user agent info.

        Parameters
        ----------
        playwright : class playwright.async_api._generated.Playwright
            main playwright class.
        """
        self.browser = await playwright.chromium.launch(args=['--start-fullscreen'])
        self.page = await self.browser.new_page()

        await self.page.bring_to_front()
        self.setup_error_handlers()

    def log_test(self, msg):
        """
        Log a test message.

        Parameters
        ----------
        msg : string
            text to log.
        """
        print(msg)

    async def get_handle(self, selector):
        """
        Get handle for a specific element and assert that it exists.

        Parameters
        ----------
        selector : string
            selector used to locate the element.
        """
        handle = await self.page.wait_for_selector(selector, state='attached', timeout=3055)

        self.assertIsNotNone(handle,
                             "Could not find element with selector '" +
                             selector + "' in the N2 diagram.")

        return handle


def get_free_port():
    """
    Get a free port.

    Returns
    -------
    int
        A free port.
    """
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as _socket:
        _socket.bind(('', 0))
        _, port = _socket.getsockname()
        return port
