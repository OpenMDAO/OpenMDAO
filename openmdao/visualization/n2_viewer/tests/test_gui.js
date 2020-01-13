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
        currentTestDesc = test.desc;
        console.log("  Testing: " + test.desc);
        const btnHandle = await page.$('#' + test.id);
        await btnHandle.click({ 'button': 'left', 'delay': 5 });
        await page.waitFor(test.wait);
        // await page.screenshot({ path: 'circuit_' + test.id + '.png' }, { 'fullPage': true });
    }
}

async function doCircuitModelTests(page) {
    console.log("Performing diagram-specific tests...")
    await page.reload({ 'waitUntil': 'networkidle0' });
    console.log("  Waiting " + transitionWait + "ms for page to reload...")
    await page.waitFor(transitionWait);

    // Hover over a specific cell and make sure the number of arrows is correct.
    // When it was broken, this diagram would show an arrow going offscreen to
    // an element that didn't exist.
    currentTestDesc = "Hover on circuit.R2.I and check arrow count";

    console.log("  Testing: " + currentTestDesc);

    let hndl_24_24 = await page.$("rect#cellShape_24_24.vMid");
    if (!hndl_24_24) {
        console.log("Error: Could not find matrix cell element in circuit diagram.");
        process.exit(1);
    }

    await hndl_24_24.hover();
    await page.waitFor(normalWait); // Give it a chance to draw the arrows

    const EXPECTED_ARROWS = 4;
    let arrows = await page.$$('g#n2arrows > path');
    if (!arrows) {
        console.log("Error: Could not find any arrows after hover.");
        process.exit(1);
    }
    if ( arrows.length != EXPECTED_ARROWS ) {
        console.log('Found ' + arrows.length + ' arrows after hover, expected ' + EXPECTED_ARROWS);
        process.exit(1);
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
        
        console.log(
            "\n----------------------------------------\n" +
            n2Filename +
            "\n----------------------------------------"
        );

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

    const page = await browser.newPage();
    setupErrorHandlers(page);

    await runGenericTests(page);
    await browser.close(); // Don't forget this or the script will wait forever!
};

let n2Files = findFiles();
(async () => { await runTests(); })();
