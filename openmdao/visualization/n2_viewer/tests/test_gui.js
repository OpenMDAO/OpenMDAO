#!/usr/bin/env node

const puppeteer = require('puppeteer');
const argv = require('yargs').argv;
// const path = require('path');
const fs = require('fs');
const urlPrefix = 'file://';

// Amount to wait when expecting a transition.
// Set later on by getting the value from the page:
let transitionWait = -1;

// The amount to wait when there's no transition:
const normalWait = 10;

const expectedModels = ['bug_arrow', 'circuit', 'double_sellar'];

// Updated at each test to describe the current action:
let currentTestDesc = '';

const lineStr = '-'.repeat(78);

function logTest(msg) {
    currentTestDesc = msg;
    console.log("  Test: " + msg);
}

/**
 * Click most of the toolbar buttons to see if an error occurs.
 * @param {Page} page Reference to the page that's already been loaded.
 */
async function doGenericToolbarTests(page) {

    const genericToolbarTests = [
        { "desc": "Collapse All Outputs button", "id": "collapseAllButtonId", "wait": transitionWait },
        { "desc": "Uncollapse All button", "id": "uncollapseAllButtonId", "wait": transitionWait },
        { "desc": "Collapse Outputs in View Only button", "id": "collapseInViewButtonId", "wait": transitionWait },
        { "desc": "Uncollapse In View Only button", "id": "uncollapseInViewButtonId", "wait": transitionWait },
        { "desc": "Show Legend (on) button", "id": "showLegendButtonId", "wait": normalWait },
        { "desc": "Show Legend (off) button", "id": "showLegendButtonId", "wait": normalWait },
        { "desc": "Show Path (on) button", "id": "showCurrentPathButtonId", "wait": normalWait },
        { "desc": "Show Path (off) button", "id": "showCurrentPathButtonId", "wait": normalWait },
        { "desc": "Toggle Solver Names (on) button", "id": "toggleSolverNamesButtonId", "wait": transitionWait },
        { "desc": "Toggle Solver Names (off) button", "id": "toggleSolverNamesButtonId", "wait": transitionWait },
        { "desc": "Clear Arrows and Connection button", "id": "clearArrowsAndConnectsButtonId", "wait": normalWait },
        { "desc": "Help (on) button", "id": "helpButtonId", "wait": normalWait },
        { "desc": "Help (off) button", "id": "helpButtonId", "wait": normalWait },
    ];

    for (let test of genericToolbarTests) {
        logTest(test.desc);
        const btnHandle = await page.$('#' + test.id);
        await btnHandle.click({ 'button': 'left', 'delay': 5 });
        await page.waitFor(test.wait);
        // await page.screenshot({ path: 'circuit_' + test.id + '.png' }, { 'fullPage': true });
    }
}

/**
 * Count the number of path elements in the n2arrows <div> and make
 * sure it matches the specified value.
 * @param {Page} page Reference to the page that's already been loaded.
 * @param {Number} expectedArrows The number to compare to.
 */
async function assertArrowCount(page, expectedArrows) {
    let arrows = await page.$$('g#n2arrows > path');
    if (!arrows) {
        console.log("Error: Could not find n2arrows <g> element.");
        process.exit(1);
    }
    else if ( arrows.length != expectedArrows ) {
        console.log('Error: Found ' + arrows.length + ' arrows, expected ' + expectedArrows);
        process.exit(1);
    }
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
async function hoverAndCheckArrowCount(page, selector, expectedArrowCount) {
    let hndl = await getHandle(page, selector)

    await hndl.hover();
    await page.waitFor(normalWait); // Give it a chance to draw the arrows

    await assertArrowCount(page, expectedArrowCount); // Make sure there are enough arrows
    await page.mouse.move(0,0); // Get the mouse off the element
    await page.waitFor(normalWait);
    await assertArrowCount(page, 0); // Make sure there are no arrows left
}

/**
 * Left-click the home button and wait for the transition to complete.
 * @param {Page} page Reference to the page that's already been loaded.
 */
async function returnToRoot(page) {
    logTest("Return to root")
    let hndl = await getHandle(page, "button#returnToRootButtonId.myButton");
    await hndl.click();
    await page.waitFor(transitionWait);
}

async function doCircuitModelTests(page) {
    console.log("Performing diagram-specific tests...")
    await page.reload({ 'waitUntil': 'networkidle0' });
    await page.waitFor(transitionWait);

    // Hover over a specific cell and make sure the number of arrows is correct.
    // When it was broken, this diagram would show an arrow going offscreen to
    // an element that didn't exist.
    logTest("Hover on circuit.R2.I and check arrow count");
    await hoverAndCheckArrowCount(page, "g#n2elements rect#cellShape_24_24.vMid", 4);

    logTest("Left-click on circuit.R2 to zoom");
    let hndl = await getHandle(page, "g#tree rect#circuit_R2");
    await hndl.click();
    await page.waitFor(transitionWait);

    logTest("Hover over zoomed circuit.R2.I and check arrow count");
    await hoverAndCheckArrowCount(page, "g#n2elements rect#cellShape_24_24.vMid", 4);

    await returnToRoot(page);

    logTest("Right-click on circuit.N1 to collapse")
    hndl = await getHandle(page, "g#tree rect#circuit_n1");
    await hndl.click({'button': 'right'});
    await page.waitFor(transitionWait);

    logTest("Hover over collapsed cell and check arrow count");
    await hoverAndCheckArrowCount(page, "g#n2elements rect#cellShape_7_7.gMid", 5);

    logTest("Right-click on circuit.N1 again to uncollapse");
    hndl = await getHandle(page, "g#tree rect#circuit_n1");
    await hndl.click({'button': 'right'});
    await page.waitFor(transitionWait);

    logTest("Left-click to zoom on solver element");
    hndl = await getHandle(page, "g#solver_tree rect#circuit_n1");
    await hndl.click();
    await page.waitFor(transitionWait);

    logTest("Hover over zoomed circuit.N1.V and check arrow count");
    await hoverAndCheckArrowCount(page, "g#n2elements rect#cellShape_12_12.vMid", 5);

    await returnToRoot(page);

    logTest("Right-click on solver element to collapse")
    hndl = await getHandle(page, "g#solver_tree rect#circuit_n1");
    await hndl.click({'button': 'right'});
    await page.waitFor(transitionWait);

    logTest("Hover over collapsed cell and check arrow count");
    await hoverAndCheckArrowCount(page, "g#n2elements rect#cellShape_7_7.gMid", 5);

    logTest("Right-click on solver element again to uncollapse")
    hndl = await getHandle(page, "g#solver_tree rect#circuit_n1");
    await hndl.click({'button': 'right'});
    await page.waitFor(transitionWait);
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
 * Using the value of the --n2dir command line arg and --suffix arg, find
 * all the files in that directory that end with that suffix.
 * @returns {Array} The list of discovered filenames.
 */
function findFiles() {
    let n2HtmlRegex = new RegExp('^.+' + argv.suffix + '$');
    files = fs.readdirSync(argv.n2dir);

    let n2Files = [];
    for (let filename of files) {
        if ( n2HtmlRegex.test(filename) ) {
            console.log('Found ' + filename);
            n2Files.push(filename);
        }
    }

    return n2Files;
}

/**
 * Run tests on arbitrary list of discovered N2 HTML files.
 * @param {Page} page Reference to a page that's been initialized, but not loaded.
 */
async function runGenericTests(page) {
    for (let n2Filename of n2Files) {
        
        console.log("\n" + lineStr + "\n" + n2Filename + "\n" + lineStr);

        // Without waitUntil: 'networkidle0', processing will begin before the page
        // is fully rendered
        await page.goto(urlPrefix + argv.n2dir + '/' + n2Filename, { 'waitUntil': 'networkidle0' });

        // Milliseconds to allow for the last transition animation to finish:
        transitionWait = await page.evaluate(() => N2TransitionDefaults.durationSlow + 100)
        await doGenericToolbarTests(page);

        n2Basename = n2Filename.replace(argv.suffix, '');
        switch (n2Basename) {
            case 'circuit':
                await doCircuitModelTests(page);
                break;
            default:
                break;
        }
    }
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

    await runGenericTests(page);
    await browser.close(); // Don't forget this or the script will wait forever!
};

console.log("\n" + lineStr + "\n" + "PERFORMING N2 GUI TESTS" + "\n" + lineStr);
let n2Files = findFiles();
(async () => { await runTests(); })();
