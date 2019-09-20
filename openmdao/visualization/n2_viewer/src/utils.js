/** A faster way to find an object in an array than indexOf.
 * @param {array} array The array to search.
 * @param {Object} obj The object to search for.
 * @returns The index of the object or -1 if not found.
 */
function indexFor(array, obj) {
    for (let i = 0; i < array.length; i++) {
        if (array[i] === obj) {
            return i;
        }
    }

    return -1;
}

function exists(testVar) {
    return (testVar !== undefined && testVar !== null);
}

Object.defineProperty(Object.prototype, 'propExists', {
    value: function(propName) {
        return (this.hasOwnProperty(propName) &&
            this[propName] !== undefined &&
            this[propName] !== null);
    },
    enumerable: false, 
});