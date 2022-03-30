/**
 * Translate coordinates from the range 0 - 1 to 0 - pixel size.
 * @typedef Scale
 * @property {Function} x Scale in the horizontal direction.
 * @property {Function} y Scale in the vertical direction.
 */
class Scale {
    /**
     * Initialize the Scale.
     * @param {Object} obj Any object with width and height properties.
     */
    constructor(obj) {
        this.x = d3.scaleLinear().range([0, obj.width]);
        this.y = d3.scaleLinear().range([0, obj.height]);
        this.prev = { x: null, y: null };
    }

    /**
     * Duplicate another Scale object.
     * @param {Scale} other The object to copy from.
     */
    copyFrom(other) {
        this.x = other.x.copy();
        this.y = other.y.copy();
    }

    /** Backup the current scale objects for future reference. */
    preserve() {
        this.prev = {
            x: this.x.copy(),
            y: this.y.copy()
        }
    }
}
