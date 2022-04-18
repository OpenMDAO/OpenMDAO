// <<hpp_insert gen/InfoProp.js>>
// <<hpp_insert gen/WindowResizable.js>>

/**
 * Manage a window for displaying the value of a variable.
 * @typedef ValueInfo
 */
class ValueInfo extends WindowResizable {

    /**
     * Add a new value window if it doesn't already exist.
     * @param {String} name Variable name.
     * @param {Number} val Variable value.
     * @param {PersistentNodeInfo} pnInfo The PersistentNodeInfo window to get data from.
     * @returns {ValueInfo} The newly constructed window.
     */
    static add(name, val, pnInfo) {
        if (!ValueInfo.existingValueWindows[name]) {
            ValueInfo.existingValueWindows[name] = true;
            return new ValueInfo(name, val, pnInfo);
        }
    }

    /** Remove the name of the window from the list of existing ones. */
    static del(name) {
        if (ValueInfo.existingValueWindows[name]) {
            delete ValueInfo.existingValueWindows[name];
        }
    }

    /**
     * Build a list of the properties we care about and set up
     * references to the HTML elements.
     * @param {String} name Variable name.
     * @param {Number} val Variable value.
     * @param {PersistentNodeInfo} pnInfo The PersistentNodeInfo window to get data from.
     */
    constructor(name, val, pnInfo) {
        super('valueInfo-' + uuidv4());
        this.name = name;
        this.val = val;

        this.table = this.body.append('table');
        this.tbody = this.table.append('tbody');

        this.minWidth = 180;
        this.minHeight = 100;
        this.theme('value-info').populate(pnInfo);
    }

    /**
     * Based on the number of dimensions of the value,
     * indicate whether a value window display is needed or even practical
     * @param {Object} val int, float, list,... The variable value.
     * @returns {String} The converted array.
     */
    static canDisplay(val) {
        if (!val) return false; // if no value, cannot display

        if (!Array.isArray(val)) return false; // scalars don't need separate display

        // 1-D arrays can be displayed
        if (!Array.isArray(val[0])) return true;

        // Handle 2-D array
        if (!Array.isArray(val[0][0])) return true;

        // More than 2-D array - punt for now - no practical way to display
        return false;
    }

    /**
     * Fill the table with the data from our val array and display in a window.
     */
    populate(pnInfo) {
        // Check to see if the data is a 2d array since the rest of the code assumes
        // that it is an Array. If only 1d, make it a 2d with one row.
        let val = this.val;
        if (!Array.isArray(val[0])) {
            val = [val];
        }

        // Make the top row of the table the indices of the sub-arrays
        const topRow = this.tbody.append('tr');
        topRow.append('th'); // Top left corner spot is empty
        const valIdxArr = Array.from(val[0].keys());

        topRow.selectAll('th.node-value-index')
            .data(valIdxArr)
            .enter()
            .append('th')
            .classed('node-value-index', true)
            .text(function (d) { return d; });

        // Construct the table displaying the variable value
        const evenOdd = ['even', 'odd'];
        const rows = this.tbody
            .selectAll('tr.array-row')
            .data(val)
            .enter()
            .append('tr')
            // Style alternating rows differently:
            .attr('class', function (d, i) { return `array-row ${evenOdd[i % 2]}`; });

        // Insert the array index into the first column:
        rows.append('th').text(function (d, i) { return i; });

        // Add the contents of the array:
        rows.selectAll('td')
            .data(function (row) { return row; })
            .enter()
            .append('td')
            .text(function (d) { return InfoPropArray.floatFormatter(d); });

        const pnInfoPos = pnInfo._getPos();
        this.sizeToContent(17, 17) // TODO: Properly find size of scrollbar + 2
            .title(this.name)
            .move(pnInfoPos.left + 20, pnInfoPos.top + 20)
            .show();

        // Save the width and height of the table when it is fully
        // constructed. This will be used later to limit the resizing
        // of the window. No need to let the user resize to a size
        // larger than full size
        const pos = this._getPos();
        this.maxWidth = pos.width;
        this.maxHeight = pos.height;
        if (this.minHeight > pos.height) this.minHeight = pos.height;
    }

    /** Remove our name from the list of existing windows before closing. */
    close(e) {
        ValueInfo.del(this.name);
        super.close(e);
    }
}

// "Class" variable and function for ValueInfo
ValueInfo.TRUNCATE_LIMIT = 80;
ValueInfo.existingValueWindows = {};
