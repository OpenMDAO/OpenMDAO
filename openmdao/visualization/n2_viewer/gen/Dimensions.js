/**
 * A simple interface for handling & preserving numerical coordinates and dimensions.
 * @typedef Dimensions
 * @property {Number} x X coordinate to store (if set).
 * @property {Number} y Y coordinate to store (if set).
 * @property {Number} width Horizontal length to store (if set).
 * @property {Number} height Vertical length to store (if set).
 * @property {String} unit The unit of measurement that applied to all values.
 * @property {Object} prev Previous set of coordinates.
 */
 class Dimensions {
    static allowedVals = ['x', 'y', 'z', 'height', 'width', 'margin', 'top', 'right', 'bottom', 'left'];

    constructor(obj, unit = 'px') {
        this.initFrom(obj, unit)
    }

    /**
     * Duplicate known values from any Object. Reset managed values and previous dimensions.
     * @param {Object} obj The object to find values in.
     * @param {String} unit The unit of measurement that applied to all values.
     */
    initFrom(obj, unit = 'px') {
        this.unit = unit;
        this._managedVals = new Set();
        this.prev = {};

        for (const val of Dimensions.allowedVals) {
            if (val in obj) {
                this._managedVals.add(val);
                this[val] = obj[val];
                this.prev[val] = 0;
            }
        } 
    }

    /**
     * Duplicate another Dimensions object.
     * @param {Coords} other The object to copy from.
     */
    copyFrom(other) {
        this.prev = other.prev;
        this._managedVals = other._managedVals;
        for (const val of this._managedVals) {
            this[val] = other[val]
        }
    }

    /** Backup the current values for future reference. */
    preserve() {
        for (const val of this._managedVals) {
            this.prev[val] = this[val];
        }
    }
}
