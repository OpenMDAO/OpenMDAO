/**
 * A simple interface for handling & preserving numerical coordinates and dimensions.
 * @typedef Dimensions
 * @property {String} unit The unit of measurement that applied to all values.
 * @property {Object} prev Previous set of coordinates.
 */
 class Dimensions {
    static allowedProps = ['x', 'y', 'z', 'height', 'width', 'margin', 'top', 'right', 'bottom', 'left'];

    /**
     * Load the values into the object.
     * @param {Object} obj The primary object to copy values from.
     * @param {String} [unit = 'px'] Unit to append to numerical values with valAsStyleStr(). 
     * @param {OBject} [prevObj = null] Optional object to initialize previous values with.
     */
    constructor(obj, unit = 'px', prevObj = null) {
        this.initFrom(obj, unit, true, prevObj)
    }

    /** Return the property value as a string with the unit appended. */
    valAsStyleStr(propName) {
        return `${this[propName]}${this.unit}`;
    }

    /**
     * Duplicate known values from any Object. Reset managed values and previous dimensions.
     * @param {Object} obj The object to find values in.
     * @param {String} unit The unit of measurement that applied to all values.
     * @param {Boolean} [initPrev = true] Whether to create & initialize the prev object.
     * @param {OBject} [prevObj = null] Optional object to initialize previous values with.
     */
    initFrom(obj, unit = 'px', initPrev = true, prevObj = null) {
        const self = this;
        this.unit = unit;
        this._managedProps = new Set();
        if (initPrev) this.prev = {};

        for (const prop of Dimensions.allowedProps) {
            if (prop in obj) {
                this._managedProps.add(prop);
                this[prop] = obj[prop];
                if (initPrev) this.prev[prop] = prevObj? prevObj[prop]: 0;

                // Add a getter function for the property to be used with a CSS style
                Object.defineProperty(this, `${prop}Style`, {
                    get: function() { return self.valAsStyleStr(prop); }
                });
            }
        } 
    }

    /**
     * Duplicate another Dimensions object.
     * @param {Dimensions} other The object to copy from.
     */
    copyFrom(other) {
        this.initFrom(other, other.unit, false);
        this.prev = {};

        for (const prop of other.prev) {
            this.prev[prop] = other.prev[prop];
        }
    }

    /** Backup the current values for future reference. */
    preserve() {
        this.prev = {};
        for (const prop of this._managedProps) {
            this.prev[prop] = this[prop];
        }
    }
}
