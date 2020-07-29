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

const inputRegex = /^(input|unconnected_input|autoivc_input)$/;
const inputOrOutputRegex = /^(output|input|unconnected_input|autoivc_input)$/;

d3.selection.prototype.originalFuncs = {
    'transition': d3.selection.prototype.transition,
    'duration': d3.selection.prototype.duration,
    'delay': d3.selection.prototype.delay
};
d3.selection.prototype.transitionAllowed = true;

function returnThis() { return this; }

function startTimer(label) {
    if (DebugFlags.timings) console.time(label);
}

function stopTimer(label) {
    if (DebugFlags.timings) console.timeEnd(label);
}

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

    // console.log(className + '.' + funcName + ": 'this' IS an instance of the " + className + " class.");
}
