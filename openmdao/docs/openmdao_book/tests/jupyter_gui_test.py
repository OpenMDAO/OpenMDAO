"""Test Jupyter doc GUI mods specific to OpenMDAO using Playwright."""
import asyncio
from aiounittest import async_test
import http.server
import os
import pathlib
import sys
import threading
import unittest


from playwright.async_api import async_playwright

if 'win32' in sys.platform:
    # Windows specific event-loop policy & cmd
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from openmdao.utils.gui_testing_utils import _GuiTestCase, get_free_port

HEADLESS = True  # Set to False if you want to see the browser

# Only can test if the docs have been built
@unittest.skipUnless(pathlib.Path(__file__).parent.parent.joinpath("_build").exists(),
                     "Cannot test without docs being built")
class TestOpenMDAOJupyterBookDocs(_GuiTestCase):

    async def setup_browser(self, playwright):
        """ Create a browser instance and go to the home page."""
        self.browser = await playwright.chromium.launch(args=['--start-fullscreen'],
                                                        headless=HEADLESS)
        self.page = await self.browser.new_page()

        url = f'http://localhost:{self.port}/index.html'

        # Without wait_until: 'networkidle', processing will begin before
        # the page is fully rendered
        await self.page.goto(url, wait_until='networkidle')

        await self.page.bring_to_front()
        self.setup_error_handlers()

    async def setup_http_server(self):

        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            # to hide all the logging
            def log_message(self, format, *args):
                pass

        html_dir = pathlib.Path(__file__).parent.parent.joinpath("_build/html")
        os.chdir(html_dir)

        self.port = get_free_port()
        server_address = ("", self.port)
        self.server = http.server.HTTPServer(server_address, QuietHandler)

        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()

    async def close_http_server(self):
        self.server.server_close()  # closes the port

    async def cleanup(self):
        await self.browser.close()

        await self.close_http_server()

        if self.console_error:
            msg = "Console log contains errors."
            print(msg)
            self.fail(msg)

        if self.page_error:
            msg = "There were errors on the page."
            print(msg)
            self.fail(msg)

    @async_test
    async def test_include_source_docs_option_unchecked(self):
        # Check to make sure that the search does the right thing when Include Source
        #  Docs is not checked
        async with async_playwright() as playwright:

            await self.setup_http_server()

            await self.setup_browser(playwright)

            # Just use some example search string
            searchbar = await self.page.wait_for_selector('#search-input', state='visible')
            search_string = "Component"
            await searchbar.type(search_string + "\n", delay=50)

            # Have to wait until the search results are completely displayed
            await self.page.wait_for_selector('#search-results h2:has-text("Search Results")')

            # make sure there weren't any source docs in the search results
            search_results_all = await self.page.query_selector_all('ul.search li a')
            for a in search_results_all:
                href = await a.get_attribute("href")
                self.assertNotIn("_srcdocs", href,
                                 "Search did not include source docs but a source doc found")

            await self.cleanup()

    @async_test
    async def test_include_source_docs_option_checked(self):
        # Check to make sure that the search does the right thing when Include Source
        #  Docs is checked

        async with async_playwright() as playwright:

            await self.setup_http_server()

            await self.setup_browser(playwright)

            # Include source docs in the search
            search_source_checkbox = await self.page.wait_for_selector('#search-source',
                                                                       state='visible')
            await search_source_checkbox.click(delay=50)

            searchbar = await self.page.wait_for_selector('#search-input', state='visible')
            search_string = "Component"
            await searchbar.type(search_string + "\n", delay=50)

            # Have to wait until the search results are completely displayed
            await self.page.wait_for_selector('#search-results h2:has-text("Search Results")')

            # Look through all the search results
            search_results_all = await self.page.query_selector_all('ul.search li a')
            # Need to find at least one href which includes "_srcdocs"
            srcdocs_found = False
            for a in search_results_all:
                href = await a.get_attribute("href")
                if "_srcdocs" in href:
                    srcdocs_found = True
                    break

            self.assertTrue(srcdocs_found,
                            "Include source docs checkbox checked but no srcdocs "
                            "found in search results")

            await self.cleanup()
