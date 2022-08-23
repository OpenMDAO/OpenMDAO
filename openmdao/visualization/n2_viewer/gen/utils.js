/** A faster way to find an object in an array than indexOf.
 * @param {array} arr The array to search.
 * @param {Object} obj The object to search for.
 * @returns The index of the first matching object, or -1 if not found.
 */
function indexFor(arr, obj) {
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] === obj) {
            return i;
        }
    }

    return -1;
}

/** Find the index of an object in the array with the first member
 * that matches the supplied value.
 * @param {any[]} arr The array to search.
 * @param {string} memName The name of the object member.
 * @param {Object} val The value of the object member to match.
 * @returns The index of the first matching object or -1 if not found.
 */
function indexForMember(arr, memName, val) {
    for (let i = 0; i < arr.length; i++) {
        if (arr[i].hasOwnProperty(memName) &&
            arr[i][memName] === val) {
            return i;
        }
    }

    return -1;
}

/** Make sure testVar is neither undefined or null.
 * @param {any} testVar The variable to check.
 * @return {Boolean} True if testVar is not undefined and not null.
*/
function exists(testVar) {
    return (testVar !== undefined && testVar !== null);
}

/**
 * Add a method to the Object prototype call propExists(),
 * which is stronger than hasOwnProperty() in that it also checks
 * the property for undefined and null values.
 * @param {string} propName The name of the property to check.
 * @return {Boolean} True if propName exists in the object, and is
 *   not undefined and not null.
 */
Object.defineProperty(Object.prototype, 'propExists', {
    value: function (propName) {
        return (this.hasOwnProperty(propName) &&
            this[propName] !== undefined &&
            this[propName] !== null);
    },
    enumerable: false,
});

/**
 * Since checking for an Array's existance as well as 
 * length > 0 is very common, combine the two.
 */
Array.isPopulatedArray = function (arr) {
    return (Array.isArray(arr) && arr.length > 0);
};

/** Preserve D3 prototypes so transitions can be disabled for complex diagrams */
d3.selection.prototype.originalFuncs = {
    'transition': d3.selection.prototype.transition,
    'duration': d3.selection.prototype.duration,
    'delay': d3.selection.prototype.delay
};
d3.selection.prototype.transitionAllowed = true;

/** Useful dummy function */
function returnThis() { return this; }

/** Log info to the console if debugging is enabled */
function debugInfo() {
    if (DebugFlags.info) console.log(...arguments);
}

/**
 * When class member functions are used with callbacks, 'this' can be redefined if
 * not managed carefully. This function double checks to make sure 'this' refers
 * to the correct class.
 * @param {Object} testThis The 'this' from inside the calling member function.
 * @param {string} className The name of the expected class.
 * @param {string} funcName The name of the calling function, for error reporting.
 * @throws {TypeError} If testThis isn't an instance of className.
 */
function testThis(testThis, className, funcName) {
    let thisClass = testThis.constructor.name;

    if (thisClass != className) {
        throw new TypeError(className + '.' + funcName +
            ": 'this' is an instance of '" + thisClass +
            "'; expected '" + className + "'.");
    }
}

/**
 * UUID generator based on crypto API
 * @returns {String} UUID in a format such as '400042c1-2d83-445b-9e84-104f7befdc68'
 */
function uuidv4() {
    return ([1e7] + -1e3 + -4e3 + -8e3 + -1e11).replace(/[018]/g, c =>
        (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
    );
}

/**
 * Delete all properties in the object without destroying the actual object.
 * @param {Object} obj The object to operate on.
 */
function wipeObj(obj) {
    for (const prop in obj) {
        if (obj.hasOwnProperty(prop)) delete obj[prop];
    }
}

/**
 * Determine if an HTML DOM element is truly visible.
 * @param {HTMLElement} elem Reference to the element to check.
 * @returns {Boolean} True is the element can be seen.
 */
function visible(elem) {
    return !!( elem.offsetWidth || elem.offsetHeight || elem.getClientRects().length );
}

/**
 * An adaptation of the common "basename()" function.
 * @param {String} filepath The path to get the basename from.
 * @returns {String} The basename of the current URL, minus the .html/.htm extension.
 */
function basename(filepath = window.location.pathname) {
    return String(filepath).replace(/^.*[\/\\](.+)\.html?$/i, "$1");
}
