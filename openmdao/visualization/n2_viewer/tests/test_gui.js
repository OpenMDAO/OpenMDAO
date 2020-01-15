#!/usr/bin/env node

const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');
const run = require('child_process').execFileSync;
const scripts = require('./gui_scripts.json');

const urlPrefix = 'file://';

// Amount to wait when expecting a transition.
// Set later on by getting the value from the page:
let transitionWait = -1;

// The amount to wait when there's no transition:
const normalWait = 10;

// Updated at each test to describe the current action:
let currentTestDesc = '';

// A long line for visually separating console text areas
const lineStr = '-'.repeat(78);

// Track which test # we're on
let currentTest = 1;

const n2dir = path.join(__dirname, 'gui_test_models');
const n2suffix = '_N2_TEST.html';

/**
 * Add some zeroes to the front of a number if it's too short,
 * ala printf("%0d")
 * @param {Number} num The number to modify.
 * @param {Number} size The min number of digits.
 * @returns {String} Number with zeroes prepended, if necessary.
 */
function zeroPad(num, size) {
    let s = String(num);
    while (s.length < (size || 2)) { s = "0" + s; }
    return s;
}

/**
 * First set global currentTestDesc, which is printed if there's an error
 * on the page so we know where the error happened. Then print the message.
 * @param {String} msg The text to print to the console.
 */
function logTest(msg) {
    currentTestDesc = msg;
    console.log("  Test " + zeroPad(currentTest, 4) + ": " + msg);
    currentTest++;
}

/**
 * Find all the files in the test subdir that end with the specified suffix.
 * @param {String} suffix The filename suffix to search for. 
 * @returns {Array} The list of discovered filenames.
 */
function findFiles(suffix) {
    let n2HtmlRegex = new RegExp('^.+' + suffix + '$');

    files = fs.readdirSync(n2dir);

    let foundFiles = [];
    for (let filename of files) {
        if (n2HtmlRegex.test(filename)) {
            console.log('Found ' + filename);
            foundFiles.push(filename);
        }
    }

    return foundFiles;
}

/**
 * Run 'openmdao n2' on all of the .py files that were found.
 * @param {Array} pyFiles List of discovered models.
 */
function generateN2Files(pyFiles) {
    for (let pyFile of pyFiles) {
        let basename = path.basename(pyFile, '.py');
        let n2File = path.join(n2dir, basename + n2suffix);
        let pyPath = path.join(n2dir, pyFile);

        console.log("Generating N2 for " + basename + ".py");

        run('openmdao', ['n2', '-o', n2File, '--no_browser', pyPath]);

        // Create an N2 using declared partials
        if (basename == 'circuit') {
            console.log("Generating N2 with --use_declare_partial_info for " + basename + ".py");
            n2File = path.join(n2dir, 'udpi_' + basename + n2suffix);
            run('openmdao', ['n2', '-o', n2File, '--no_browser',
                '--use_declare_partial_info', pyPath]);

        }
    }
}

/**
 * Click most of the toolbar buttons to see if an error occurs.
 * @param {Page} page Reference to the page that's already been loaded.
 */
async function doGenericToolbarTests(page) {

    for (let test of scripts['__toolbar']) {
        logTest(test.desc);
        const btnHandle = await page.$('#' + test.id);
        await btnHandle.click({ 'button': 'left', 'delay': 5 });

        const waitTime = test.waitForTransition ? transitionWait : normalWait;
        await page.waitFor(waitTime);
        // await page.screenshot({ path: 'test_' + test.id + '.png' }, { 'fullPage': true });
    }
}

/**
 * Count the number of elements located by the selector and make
 * sure it matches the supplied value.
 * @param {Page} page Reference to the page that's already been loaded.
 * @param {String} selector CSS selector to find multiple elements.
 * @param {Number} expectedArrows The number to compare to.
 */
async function assertElementCount(page, selector, expectedFound) {
    const hndlArray = await page.$$(selector);
    if (!hndlArray) {
        console.log("Error: Could not find any '" + selector + "' elements.");
        process.exit(1);
    }
    else if (hndlArray.length != expectedFound) {
        console.log('Error: Found ' + hndlArray.length +
            ' elements, expected ' + expectedFound);
        process.exit(1);
    }
}

/**
 * Count the number of path elements in the n2arrows <div> and make
 * sure it matches the specified value.
 * @param {Page} page Reference to the page that's already been loaded.
 * @param {Number} expectedArrows The number to compare to.
 */
async function assertArrowCount(page, expectedArrows) {
    await assertElementCount(page, 'g#n2arrows > path', expectedArrows);
}

/**
 * Get a handle for the specific element and assert that it exists.
 * @param {Page} page Reference to the page that's already been loaded.
 * @param {String} selector Unique path to a page element.
 * @returns {elementHandle} Reference to the selected element.
 */
async function getHandle(page, selector) {
    let handle = await page.$(selector);

    if (!handle) {
        console.log("Error: Could not find element with selector '" + selector + "' in the N2 diagram.");
        process.exit(1);
    }

    return handle;
}

/**
 * Hover over a matrix cell, make sure the number of expected arrows are there,
 * then move off and make sure the arrows go away.
 * @param {Page} page Reference to the page that's already been loaded.
 * @param {String} selector Unique path to a page element.
 * @param {Number} expectedArrowCount The number of arrows to check.
 */
async function testHoverAndArrowCount(page, options) {
    logTest(options.desc ? options.desc :
        "Hover over '" + options.selector + "' and checking arrow count");
    let hndl = await getHandle(page, options.selector)

    await hndl.hover();
    await page.waitFor(normalWait); // Give it a chance to draw the arrows

    await assertArrowCount(page, options.arrowCount); // Make sure there are enough arrows
    await page.mouse.move(0, 0); // Get the mouse off the element
    await page.waitFor(normalWait);
    await assertArrowCount(page, 0); // Make sure there are no arrows left
}

/**
 * Perform a click of the type specified by options.button on the
 * element specified by options.selector.
 * @param {Object} options
 * @param {String} options.button Use the left or right mouse button
 * @param {String} options.selector The CSS selector for the element
 */
async function click(page, options) {
    logTest(options.desc ? options.desc :
        options.button + "-click on '" + options.selector + "'");
    const hndl = await getHandle(page, options.selector);
    await hndl.click({ 'button': options.button });
    await page.waitFor(transitionWait);
}

/**
 * Left-click the home button and wait for the transition to complete.
 * @param {Page} page Reference to the page that's already been loaded.
 */
async function returnToRoot(page) {
    logTest("Return to root")
    const hndl = await getHandle(page, "button#returnToRootButtonId.myButton");
    await hndl.click();
    await page.waitFor(transitionWait);
}

/**
 * Enter a string in the search textbox and check that the expected
 * number of elements are shown in the N2 matrix.
 * @param {Page} page Reference to the page that's already been loaded.
 * @param {Object} options
 * @param {String} options.desc Description of the current test.
 * @param {String} options.searchString The path name to search for.
 * @param {Number} options.n2ElementCount N2 elements to expect.
 */
async function testSearchAndResult(page, options) {
    logTest(options.desc ? options.desc :
        "Searching for '" + options.searchString + "' and checking for " +
        options.n2ElementCount + " N2 elements after.");

    const hndl = await getHandle(page, "div#toolbarLoc input#awesompleteId");
    await hndl.type(options.searchString);
    await hndl.press('Enter');
    await page.waitFor(transitionWait + 500);

    await assertElementCount(page, "g#n2elements > g.n2cell", options.n2ElementCount);
}

/**
 * Iterate through the supplied script array and perform each
 * action/test.
 * @param {Page} page Reference to the page that's already been loaded.
 * @param {Array} scriptArr Reference to the array of tests.
 */
async function runModelScript(page, scriptArr) {
    console.log("Performing diagram-specific tests...")
    await page.reload({ 'waitUntil': 'networkidle0' });
    await page.waitFor(transitionWait);

    for (let scriptItem of scriptArr) {
        switch (scriptItem.test) {
            case 'hoverArrow':
                await testHoverAndArrowCount(page, scriptItem);
                break;
            case 'click':
                await click(page, scriptItem);
                break;
            case 'root':
                await returnToRoot(page);
                break;
            case 'search':
                await testSearchAndResult(page, scriptItem);
                break;
            case 'count':
                logTest(scriptItem.desc ? scriptItem.desc :
                    "Checking for " + scriptItem.count + "' instances of '" +
                    scriptItem.selector + "'");
                await assertElementCount(page, scriptItem.selector,
                    scriptItem.count);
                break;
        }
    }
}

/**
 * Set up the event handlers that are critical for catching errors. When
 * an error is encountered, exit with status > 0 to indicate a failure.
 * @param {Page} page Reference to the page that's already been loaded.
 */
function setupErrorHandlers(page) {
    page.on('console', msg => {
        switch (msg.type()) {
            case 'warning':
                console.log("Warning: " + currentTestDesc)
                for (let i = 0; i < msg.args().length; ++i)
                    console.log(`${i}: ${msg.args()[i]}`);
                break;
            case 'error':
                console.log("Error: " + currentTestDesc + " failed")
                for (let i = 0; i < msg.args().length; ++i)
                    console.log(`${i}: ${msg.args()[i]}`);
                process.exit(1);
                break;
        }
    });

    page.on('pageerror', err => {
        console.log(err)
        process.exit(1);
    });
}

/**
 * Start the browser, load the page, and call all the testing functions.
 */
async function runTests() {

    const browser = await puppeteer.launch({
        'defaultViewport': {
            'width': 1280,
            'height': 1024
        }
    });

    const userAgentStr = await browser.userAgent();
    console.log("Browser: " + userAgentStr);

    const page = await browser.newPage();
    setupErrorHandlers(page);

    const knownModelNames = Object.keys(scripts);

    for (let n2Filename of n2Files) {
        console.log("\n" + lineStr + "\n" + n2Filename + "\n" + lineStr);

        // Without waitUntil: 'networkidle0', processing will begin before the page
        // is fully rendered
        await page.goto(urlPrefix + n2dir + '/' + n2Filename,
            { 'waitUntil': 'networkidle0' });

        // Milliseconds to allow for the last transition animation to finish.
        // Obtain value defined in N2 code.
        transitionWait = await page.evaluate(() => N2TransitionDefaults.durationSlow + 100)
        await doGenericToolbarTests(page);

        // If this model has an associated script, run it:
        n2Basename = n2Filename.replace(n2suffix, '');
        if (knownModelNames.includes(n2Basename)) {
            await runModelScript(page, scripts[n2Basename]);
        }
    }

    await browser.close(); // Don't forget this or the script will wait forever!
};

console.log("\n" + lineStr + "\n" + "PERFORMING N2 GUI TESTS" + "\n" + lineStr);

let pyFiles = findFiles('.py');
generateN2Files(pyFiles);

let n2Files = findFiles(n2suffix);
(async () => { await runTests(); })();