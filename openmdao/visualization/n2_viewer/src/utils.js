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
    for (let i = 0; i < array.length; i++) {
        if (array[i].hasOwnProperty(memName) &&
            array[i][memName] === obj) {
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

/** Add a method to the Object prototype call propExists(),
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