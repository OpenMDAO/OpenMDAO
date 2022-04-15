"""
To be run when there's an update to the appearance of the N2 Toolbar.

It generates an N2 html file, uses Pyppeteer w/Chromium to load it up,
then takes a snapshot of the toolbar and saves it as a base64-encoded PNG.
"""

import os
import subprocess
import asyncio
from playwright.async_api import async_playwright
import base64

MODEL_FILE = '../tests/gui_test_models/circuit.py'
N2_FILE = 'n2_toolbar_screenshot.html'
TMP_PNG_FILE = 'n2_toolbar_tmp.png'
OUTPUT_FILE = 'n2toolbar_screenshot_png.b64'
URL_PREFIX = 'file://'
DEBUG = False
LINE_STR = '-' * 78

async def create_help(playwright):
    """Create a browser instance and print user agent info."""
    print("Opening browser")
    browser = await playwright.chromium.launch(args=['--start-fullscreen', '--headless'])

    page = await browser.new_page()
    await page.bring_to_front()

    curDir = os.path.dirname(os.path.realpath(__file__))
    url = f"{URL_PREFIX}/{curDir}/{N2_FILE}"

    print("Loading N2 HTML file")
    # Without waitUntil: 'networkidle0', processing will begin before
    # the page is fully rendered
    await page.goto(url, wait_until='networkidle')

    # Milliseconds to allow for the last transition animation to finish.
    # Obtain value defined in N2 code.
    transition_wait = await page.evaluate("transitionDefaults.durationSlow")
    transition_wait += 100

    await page.wait_for_selector("#cellShape_node_26")

    br_call_str = "d3.select('#toolbarLoc').node().getBoundingClientRect()"
    tb_height = await page.evaluate(f"{br_call_str}.height")
    tb_width = await page.evaluate(f"{br_call_str}.width")

    clipDims = {
        'x': 0,
        'y': 0,
        'width': tb_width - 8,
        'height': tb_height - 8
    }

    print(f"Taking {tb_width}x{tb_height} screenshot and saving to {TMP_PNG_FILE}")

    await page.screenshot(path = TMP_PNG_FILE, clip = clipDims, type = 'png')

    print(f"Converting to b64 and saving as {OUTPUT_FILE}")
    with open(TMP_PNG_FILE, "rb") as png_file:
        encoded_png = str(base64.b64encode(png_file.read()).decode("ascii"))
        png_file.close()
        print(f"Removing {TMP_PNG_FILE}")
        os.remove(TMP_PNG_FILE)

    with open(OUTPUT_FILE, "w") as b64_file:
        b64_file.write(encoded_png)
        b64_file.close()

    print(f"Removing {N2_FILE}")
    os.remove(N2_FILE)

    do_add = input(f"Perform 'git add {OUTPUT_FILE}' (y/n)? ")
    if do_add[0].lower() == 'y':
        cmd = f"git add -v {OUTPUT_FILE}"
        subprocess.run(cmd.split())  # nosec: trusted input

async def main():
    async with async_playwright() as playwright:
        await create_help(playwright)

if os.path.exists("./update_toolbar_help.py"):
    print(f"Generating N2 from {MODEL_FILE}")
    cmd = f"openmdao n2 -o {N2_FILE} --no_browser {MODEL_FILE}"
    subprocess.run(cmd.split())  # nosec: trusted input
    asyncio.get_event_loop().run_until_complete(main())
else:
    print("Only run from the directory where the script is located.")
