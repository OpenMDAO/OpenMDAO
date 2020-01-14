#!/usr/bin/env node

const puppeteer = require('puppeteer');
const argv = require('yargs').argv;
const fs = require('fs');
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

/**
 * First set global currentTestDesc, which is printed if there's an error
 * on the page so we know where the error happened. Then print the message.
 * @param {String} msg The text to print to the console.
 */
function logTest(msg) {
    currentTestDesc = msg;
    console.log("  Test: " + msg);
}

// These tests can be run on the toolbar of any model
const genericToolbarTests = [
    {
        "desc": "Collapse All Outputs button",
        "id": "collapseAllButtonId",
        "waitForTransition": true
    },
    {
        "desc": "Uncollapse All button",
        "id": "uncollapseAllButtonId",
        "waitForTransition": true
    },
    {
        "desc": "Collapse Outputs in View Only button",
        "id": "collapseInViewButtonId",
        "waitForTransition": true
    },
    {
        "desc": "Uncollapse In View Only button",
        "id": "uncollapseInViewButtonId",
        "waitForTransition": true
    },
    {
        "desc": "Show Legend (on) button",
        "id": "showLegendButtonId",
        "waitForTransition": false
    },
    {
        "desc": "Show Legend (off) button",
        "id": "showLegendButtonId",
        "waitForTransition": false
    },
    {
        "desc": "Show Path (on) button",
        "id": "showCurrentPathButtonId",
        "waitForTransition": false
    },
    {
        "desc": "Show Path (off) button",
        "id": "showCurrentPathButtonId",
        "waitForTransition": false
    },
    {
        "desc": "Toggle Solver Names (on) button",
        "id": "toggleSolverNamesButtonId",
        "waitForTransition": true
    },
    {
        "desc": "Toggle Solver Names (off) button",
        "id": "toggleSolverNamesButtonId",
        "waitForTransition": true
    },
    {
        "desc": "Clear Arrows and Connection button",
        "id": "clearArrowsAndConnectsButtonId",
        "waitForTransition": false
    },
    {
        "desc": "Help (on) button",
        "id": "helpButtonId",
        "waitForTransition": false
    },
    {
        "desc": "Help (off) button",
        "id": "helpButtonId",
        "waitForTransition": false
    },
];

const specificModelScripts = {
    'circuit': [
        {
            'desc': 'Hover on N2 matrix element and check arrow count',
            'test': 'hoverArrow',
            'selector': "g#n2elements rect#cellShape_24_24.vMid",
            'arrowCount': 4
        },
        {
            'desc': 'Left-click on partition tree element to zoom',
            'test': 'click',
            'selector': "g#tree rect#circuit_R2",
            'button': 'left'
        },
        {
            'desc': 'Hover on N2 matrix element and check arrow count',
            'test': 'hoverArrow',
            'selector': "g#n2elements rect#cellShape_24_24.vMid",
            'arrowCount': 4
        },
        {
            // Return to root diagram
            'test': 'root'
        },
        {
            'desc': 'Right-click on partition tree element to collapse',
            'test': 'click',
            'selector': "g#tree rect#circuit_n1",
            'button': 'right'
        },
        {
            'desc': 'Hover over collapsed N2 matrix element and check arrow count',
            'test': 'hoverArrow',
            'selector': "g#n2elements rect#cellShape_7_7.gMid",
            'arrowCount': 5
        },
        {
            'desc': 'Right-click on partition tree element to uncollapse',
            'test': 'click',
            'selector': "g#tree rect#circuit_n1",
            'button': 'right'
        },
        {
            'desc': 'Left-click to zoom on solver element',
            'test': 'click',
            'selector': "g#solver_tree rect#circuit_n1",
            'button': 'left'
        },
        {
            'desc': 'Hover over zoomed N2 cell and check arrow count',
            'test': 'hoverArrow',
            'selector': "g#n2elements rect#cellShape_12_12.vMid",
            'arrowCount': 5
        },
        {
            // Return to root diagram
            'test': 'root'
        },
        {
            'desc': 'Right-click on solver element to collapse',
            'test': 'click',
            'selector': "g#solver_tree rect#circuit_n1",
            'button': 'right'
        },
        {
            'desc': 'Hover over collapsed N2 cell and check arrow count',
            'test': 'hoverArrow',
            'selector': "g#n2elements rect#cellShape_7_7.gMid",
            'arrowCount': 5
        },
        {
            'desc': 'Right-click again on solver element to uncollapse',
            'test': 'click',
            'selector': "g#solver_tree rect#circuit_n1",
            'button': 'right'
        },
    ],
    'bug_arrow': [
        {
            'desc': 'Hover on N2 matrix element and check arrow count',
            'test': 'hoverArrow',
            'selector': "g#n2elements rect#cellShape_11_11.vMid",
            'arrowCount': 2
        },
        {
            'desc': 'Left-click on partition tree element to zoom',
            'test': 'click',
            'selector': "g#tree rect#design_fan_map_scalars",
            'button': 'left'
        },
        {
            'desc': 'Hover on N2 matrix element and check arrow count',
            'test': 'hoverArrow',
            'selector': "g#n2elements rect#cellShape_11_11.vMid",
            'arrowCount': 2
        },
        {
            // Return to root diagram
            'test': 'root'
        },
        {
            'desc': 'Right-click on partition tree element to collapse',
            'test': 'click',
            'selector': "g#tree rect#design_fan_map_scalars",
            'button': 'right'
        },
        {
            'desc': 'Hover over collapsed N2 matrix element and check arrow count',
            'test': 'hoverArrow',
            'selector': "g#n2elements rect#cellShape_10_10.gMid",
            'arrowCount': 1
        },
        {
            'desc': 'Right-click on partition tree element to uncollapse',
            'test': 'click',
            'selector': "g#tree rect#design_fan_map_scalars",
            'button': 'right'
        },
        {
            'desc': 'Left-click to zoom on solver element',
            'test': 'click',
            'selector': "g#solver_tree rect#design_fan_map_d1",
            'button': 'left'
        },
        {
            'desc': 'Hover over zoomed N2 cell and check arrow count',
            'test': 'hoverArrow',
            'selector': "g#n2elements rect#cellShape_9_9.vMid",
            'arrowCount': 1
        },
        {
            // Return to root diagram
            'test': 'root'
        },
        {
            'desc': 'Right-click on solver element to collapse',
            'test': 'click',
            'selector': "g#solver_tree rect#design_fan_map_scalars",
            'button': 'right'
        },
        {
            'desc': 'Hover over collapsed N2 cell and check arrow count',
            'test': 'hoverArrow',
            'selector': "g#n2elements rect#cellShape_10_10.gMid",
            'arrowCount': 1
        },
        {
            'desc': 'Right-click again on solver element to uncollapse',
            'test': 'click',
            'selector': "g#solver_tree rect#design_fan_map_scalars",
            'button': 'right'
        },
    ],
    'double_sellar': [
        {
            'desc': 'Hover on N2 matrix element and check arrow count',
            'test': 'hoverArrow',
            'selector': "g#n2elements rect#cellShape_11_11.vMid",
            'arrowCount': 4
        },
        {
            'desc': 'Left-click on partition tree element to zoom',
            'test': 'click',
            'selector': "g#tree rect#g1_d2_y2",
            'button': 'left'
        },
        {
            'desc': 'Hover on N2 matrix element and check arrow count',
            'test': 'hoverArrow',
            'selector': "g#n2elements rect#cellShape_11_11.vMid",
            'arrowCount': 4
        },
        {
            // Return to root diagram
            'test': 'root'
        },
        {
            'desc': 'Right-click on partition tree element to collapse',
            'test': 'click',
            'selector': "g#tree rect#g2_d1",
            'button': 'right'
        },
        {
            'desc': 'Hover over collapsed N2 matrix element and check arrow count',
            'test': 'hoverArrow',
            'selector': "g#n2elements rect#cellShape_13_13.gMid",
            'arrowCount': 3
        },
        {
            'desc': 'Right-click on partition tree element to uncollapse',
            'test': 'click',
            'selector': "g#tree rect#g2_d1",
            'button': 'right'
        },
        {
            'desc': 'Left-click to zoom on solver element',
            'test': 'click',
            'selector': "g#solver_tree rect#g2_d2",
            'button': 'left'
        },
        {
            'desc': 'Hover over zoomed N2 cell and check arrow count',
            'test': 'hoverArrow',
            'selector': "g#n2elements rect#cellShape_21_21.vMid",
            'arrowCount': 4
        },
        {
            // Return to root diagram
            'test': 'root'
        },
        {
            'desc': 'Right-click on solver element to collapse',
            'test': 'click',
            'selector': "g#solver_tree rect#g1_d1",
            'button': 'right'
        },
        {
            'desc': 'Hover over collapsed N2 cell and check arrow count',
            'test': 'hoverArrow',
            'selector': "g#n2elements rect#cellShape_3_3.gMid",
            'arrowCount': 3
        },
        {
            'desc': 'Right-click again on solver element to uncollapse',
            'test': 'click',
            'selector': "g#solver_tree rect#g1_d1",
            'button': 'right'
        },
    ]
}

/**
 * Click most of the toolbar buttons to see if an error occurs.
 * @param {Page} page Reference to the page that's already been loaded.
 */
async function doGenericToolbarTests(page) {

    for (let test of genericToolbarTests) {
        logTest(test.desc);
        const btnHandle = await page.$('#' + test.id);
        await btnHandle.click({ 'button': 'left', 'delay': 5 });

        const waitTime = test.waitForTransition? transitionWait : normalWait;
        await page.waitFor(waitTime);
        // await page.screenshot({ path: 'test_' + test.id + '.png' }, { 'fullPage': true });
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
    else if (arrows.length != expectedArrows) {
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
async function hoverAndCheckArrowCount(page, options) {
    logTest(options.desc? options.desc :
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
    logTest(options.desc? options.desc :
        options.button + "-click on '" + options.selector + "'");
    let hndl = await getHandle(page, options.selector);
    await hndl.click({ 'button': options.button });
    await page.waitFor(transitionWait);
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

async function runModelScript(page, scriptArr) {
    console.log("Performing diagram-specific tests...")
    await page.reload({ 'waitUntil': 'networkidle0' });
    await page.waitFor(transitionWait);

    for (let scriptItem of scriptArr) {
        switch (scriptItem.test) {
            case 'hoverArrow':
                await hoverAndCheckArrowCount(page, scriptItem);
                break;
            case 'click':
                await click(page, scriptItem);
                break;
            case 'root':
                await returnToRoot(page);
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
 * Using the value of the --n2dir command line arg and --suffix arg, find
 * all the files in that directory that end with that suffix.
 * @returns {Array} The list of discovered filenames.
 */
function findFiles() {
    let n2HtmlRegex = new RegExp('^.+' + argv.suffix + '$');
    files = fs.readdirSync(argv.n2dir);

    let n2Files = [];
    for (let filename of files) {
        if (n2HtmlRegex.test(filename)) {
            console.log('Found ' + filename);
            n2Files.push(filename);
        }
    }

    return n2Files;
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

    const knownModelNames = Object.keys(specificModelScripts);

    for (let n2Filename of n2Files) {
        console.log("\n" + lineStr + "\n" + n2Filename + "\n" + lineStr);

        // Without waitUntil: 'networkidle0', processing will begin before the page
        // is fully rendered
        await page.goto(urlPrefix + argv.n2dir + '/' + n2Filename,
            { 'waitUntil': 'networkidle0' });

        // Milliseconds to allow for the last transition animation to finish.
        // Obtain value defined in N2 code.
        transitionWait = await page.evaluate(() => N2TransitionDefaults.durationSlow + 100)
        await doGenericToolbarTests(page);

        // If this model has an associated script, run it:
        n2Basename = n2Filename.replace(argv.suffix, '');
        if (knownModelNames.includes(n2Basename)) {
            await runModelScript(page, specificModelScripts[n2Basename]);
        }
    }

    await browser.close(); // Don't forget this or the script will wait forever!
};

console.log("\n" + lineStr + "\n" + "PERFORMING N2 GUI TESTS" + "\n" + lineStr);

let n2Files = findFiles();
(async () => { await runTests(); })();
