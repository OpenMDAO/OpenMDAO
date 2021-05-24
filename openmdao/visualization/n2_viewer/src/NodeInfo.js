/**
 * Manage info for each node metadata property
 * @typedef InfoPropDefault
 * @property {String} key The identifier of the property.
 * @property {String} desc The description (label) to display.
 * @property {Boolean} [ capitalize = false ] Whether to capitialize every word in the desc.
 */
class InfoPropDefault {
    constructor(key, desc, capitalize = false) {
        this.key = key;
        this.desc = desc;
        this.capitalize = capitalize;
    }

    /** Return the same message since this is the base class. */
    output(msg) { return msg; }

    /** Make sure the node has the property and it's got a value. */
    canShow(node) { return (node.propExists(this.key) && node[this.key] != '') }

    /**
     * Add a table row with the supplied description and value.
     * @param {Object} tbody D3 reference to an existing table body.
     * @param {String} desc Description of the value.
     * @param {String} val The text or html value to display.
     * @param {Boolean} [ capitalize = false ] Whether to apply the caps class.
     */
    static addRowWithVal(tbody, desc, val, capitalize = false) {
        const newRow = tbody.append('tr');

        newRow.append('th')
            .attr('scope', 'row')
            .text(desc);

        const td = newRow.append('td').html(val);
        if (capitalize) td.attr('class', 'caps');
    }

    /**
     * If the object contains a non-empty property with our key, create
     * a new row with it in the supplied table body.
     * @param {Object} tbody D3 reference to an existing table body.
     * @param {N2TreeNode} node Reference to the node that may have the property.
     */
    addRow(tbody, node) {
        if (this.canShow(node)) {
            InfoPropDefault.addRowWithVal(tbody, this.desc,
                this.output(node[this.key]), this.capitalize)
        }
    }
}

/**
 * Output a Yes or No to display.
 * @typedef InfoPropYesNo
 */
class InfoPropYesNo extends InfoPropDefault {
    constructor(key, desc, showIfFalse = false) {
        super(key, desc, false);
        this.showIfFalse = showIfFalse;
    }

    /** Return Yes or No when given True or False */
    output(boolVal) { return boolVal ? 'Yes' : 'No'; }

    /** Determine whether the value represents False */
    isFalse(node) {
        const val = node[this.key];
        if (!val) return true;
        return (val.toString().match(/0|no|false|off/i));
    }

    /** Also check the showIfFalse flag */
    canShow(node) {
        const valIsFalse = this.isFalse(node);
        const showAble = (!valIsFalse || (valIsFalse && this.showIfFalse));
        return (super.canShow(node) && showAble);
    }
}

/**
 * Output a message if the value is True. 
 * @typedef InfoPropMessage
 */
class InfoPropMessage extends InfoPropYesNo {
    constructor(key, desc, message, showIfFalse = false) {
        super(key, desc, false);
        this.message = message;
    }

    /** Return message when value is True */
    output(boolVal) { 
        return boolVal ? this.message : ''; 
    }
}

/** Display a subsection of options values in the info panel */
class InfoPropOptions extends InfoPropDefault {
    constructor(key, desc, solverType = null) {
        super(key, desc, false);
        this.solverType = solverType;
    }

    /** Also check whether there are any options in the list */
    canShow(node) {
        return (super.canShow(node) && Object.keys(node[this.key]).length > 0);
    }

    /**
     * There may be a list of options, so create a subsection in the table for them.
     * @param {Object} tbody D3 reference to an existing table body.
     * @param {N2TreeNode} node Reference to the node that may have the property.
     */
    addRow(tbody, node) {
        if (!this.canShow(node)) return;

        const val = node[this.key];

        let desc = this.desc;
        if (this.solverType) {
            desc += ': ' + node[this.solverType + '_solver'].substring(3);
        }

        // Add a subsection header for the option rows to follow
        tbody.append('tr').append('th')
            .text(desc)
            .attr('colspan', '2')
            .attr('class', 'options-header');

        for (const key of Object.keys(val).sort()) {
            const optVal = (val[key] === null) ? 'None' : val[key];
            InfoPropDefault.addRowWithVal(tbody, key, optVal);
        }
    }
}

/** Display a subsection of expression values in the info panel for ExecComps */
class InfoPropExpr extends InfoPropDefault {
    constructor(key, desc) {
        super(key, desc, false);
    }

    /**
     * There may be a list of expressions, so create a subsection in the table for them.
     * @param {Object} tbody D3 reference to an existing table body.
     * @param {N2TreeNode} node Reference to the node that may have the property.
     */
    addRow(tbody, node) {
        if (!this.canShow(node)) return;

        const exprArr = node[this.key];

        // Add a subsection header for the option rows to follow
        tbody.append('tr').append('th')
            .text(this.desc)
            .attr('colspan', '2')
            .attr('class', 'options-header');

        for (const idx in exprArr) {
            const splitExpr = exprArr[idx].split(/\s*=\s*/);

            // In case the equals sign is missing for some reason:
            const displayVal = (splitExpr.length > 1)? splitExpr : [ 'Expr ' + idx, splitExpr[0]];
            InfoPropDefault.addRowWithVal(tbody, displayVal[0], displayVal[1]);
        }
    }
}

/**
 * Handles properties that are arrays.
 * @typedef InfoPropArray
 */
class InfoPropArray extends InfoPropDefault {
    constructor(key, desc, values, capitalize = false) {
        super(key, desc, capitalize);

        this.values = values;
    }

    /**
     * Convert an element to a string that is human readable.
     * @param {Object} element The scalar item to convert.
     * @returns {String} The string representation of the element.
     */
    static elementToString(element) {
        if (typeof element === 'number') {
            if (Number.isInteger(element)) { return element.toString(); }
            return this.floatFormatter(element); /* float */
        }

        if (element === 'nan') { return element; }
        return JSON.stringify(element);
    }

    /**
     * Convert a value to a string that is human readable.
     * @param {Object} val The item to convert.
     * @param {Number} level The level of nesting in the display.
     * @returns {String} The string version of the converted array.
     */
    static valToString(val, level = 0) {
        if (!Array.isArray(val)) { return this.elementToString(val); }

        let indent = ' '.repeat(level);
        let valStr = indent + '[';

        for (const element of val) {
            valStr += this.valToString(element, level + 1) + ' ';
        }

        return valStr.replace(/^(.+) $/, '$1]\n');
    }

    /**
     * Convert a value to a string that can be used in Python code.
     * @param {Object} val The value to convert.
     * @returns {String} The string of the converted object.
     */
    static valToCopyString(val) {
        if (!Array.isArray(val)) { return this.elementToString(val); }

        let valStr = 'array([';
        for (const element of val) {
            valStr += this.valToCopyString(element) + ', ';
        }

        if (val.length > 0) {
            return valStr.replace(/^(.+)(, )$/, '$1])');
        }
    }

    /**
     * Convert the array to a string that can be displayed in the info panel. Save
     * the array value, the string, and a Python version of the string in the
     * values Object so it can be copied if the panel is pinned.
     * @param {Array} array The array to display and save.
     * @returns {String} A string representation of the array.
     */
    output(array) {
        if (array == null) { return 'Value too large to include in N2'; }

        const valStr = InfoPropArray.valToString(array);
        const maxLen = ValueInfo.TRUNCATE_LIMIT;
        const isTruncated = valStr.length > maxLen;

        let html = isTruncated ? valStr.substring(0, maxLen - 3) + "..." : valStr;

        if (isTruncated && ValueInfo.canDisplay(array)) {
            html += ` <button type='button' class='show_value_button' id='${this.key}'>Show more</button>`;
        }
        html += ` <button type='button' class='copy_value_button' id='${this.key}'>Copy</button>`;

        // Store the original value and formatted value so they can be passed if the panel is pinned.
        this.values[this.key] = {
            'val': array,
            'str': valStr,
            'copyStr': InfoPropArray.valToCopyString(array),
            'isTruncated': isTruncated
        }

        return html;
    }

    /** Make sure the node has the property and it's got a value. */
    canShow(node) { return node.propExists(this.key); }
}

InfoPropArray.floatFormatter = d3.format('g');

/**
 * Manage a window for displaying the value of a variable.
 * @typedef ValueInfo
 */
class ValueInfo extends N2WindowResizable {

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
    close() {
        ValueInfo.del(this.name);
        super.close();
    }
}

// "Class" variable and function for ValueInfo
ValueInfo.TRUNCATE_LIMIT = 80;
ValueInfo.existingValueWindows = {};

/**
 * Manage a table containing all available metadata properties for
 * the currently active node, as well as whether the table is
 * visible or not.
 * @typedef NodeInfo
 */
class NodeInfo extends N2Window {
    /**
     * Build a list of the properties we care about and set up
     * references to the HTML elements.
     */
    constructor(ui) {
        super('nodeInfo-' + uuidv4());
        this.values = {};

        // Potential properties
        this.propList = [
            new InfoPropDefault('promotedName', 'Promoted Name'),
            new InfoPropDefault('absPathName', 'Absolute Name'),
            new InfoPropDefault('class', 'Class'),
            new InfoPropDefault('surrogate_name', 'Surrogate'),
            new InfoPropDefault('type', 'Type', true),
            new InfoPropDefault('dtype', 'DType'),

            new InfoPropDefault('units', 'Units'),
            new InfoPropDefault('shape', 'Shape'),
            new InfoPropYesNo('is_discrete', 'Discrete'),
            new InfoPropMessage('initial_value', '** Note **',
                                'Non-local values are not available under MPI, showing initial value.'),
            new InfoPropYesNo('distributed', 'Distributed'),
            new InfoPropArray('value', 'Value', this.values),

            new InfoPropDefault('subsystem_type', 'Subsystem Type', true),
            new InfoPropDefault('component_type', 'Component Type', true),
            new InfoPropYesNo('implicit', 'Implicit'),
            new InfoPropYesNo('is_parallel', 'Parallel'),
            new InfoPropDefault('linear_solver', 'Linear Solver'),
            new InfoPropDefault('nonlinear_solver', 'Non-Linear Solver'),
            new InfoPropExpr('expressions', 'Expressions'),

            new InfoPropOptions('options', 'Options'),
            new InfoPropOptions('linear_solver_options', 'Linear Solver Options', 'linear'),
            new InfoPropOptions('nonlinear_solver_options', 'Non-Linear Solver Options', 'nonlinear'),
        ];

        // Potential solver properties
        this.propListSolvers = [
            new InfoPropDefault('absPathName', 'Absolute Name'),
            new InfoPropOptions('linear_solver_options', 'Linear Solver Options', 'linear'),
            new InfoPropOptions('nonlinear_solver_options', 'Non-Linear Solver Options', 'nonlinear'),
        ];

        this.ui = ui;
        this.table = this.body.append('table').attr('class', 'node-info-table');
        this.tbody = this.table.append('tbody');
        this.toolbarButton = d3.select('#info-button');
        this.dataDiv = this.main.append('div').attr('class', 'node-info-data');
        this.theme('node-info');
        this.hideCloseButton();

        // Becomes active when node info mode is selected on toolbar
        this.active = false;
    }

    /** Make the info box visible if it's hidden */
    activate() {
        this.active = true;
        this.hidden = false;
        this.toolbarButton.classed('active-tab-icon', true);
        d3.select('#all_pt_n2_content_div').classed('node-data-cursor', true);
        return this;
    }

    /** Make the info box hidden if it's visible */
    deactivate() {
        this.active = false;
        this.hidden = true;
        this.toolbarButton.classed('active-tab-icon', false);
        d3.select('#all_pt_n2_content_div').classed('node-data-cursor', false);
        return this;
    }

    /** Toggle the active mode */
    toggle() {
        if (this.active) return this.deactivate();
        
        return this.activate();
    }

    pin() {
        if (this.tbody.html() == '') return; // Was already pinned so is empty

        new PersistentNodeInfo(this);
        this.hidden = true;
        this.clear();
        return this;
    }

    moveNearMouse(event, offset = 15) {
        if (!this.active) return this;

        return super.moveNearMouse(event, offset);
    }

    /** Wipe the contents of the table body */
    clear() {
        if (!this.active) return;
        // this.hidden = true;

        this.setList({
            width: null, height: null, left: null, right: null,
            top: null, bottom: null
        })
        this.dataDiv.html('');
        this.tbody.html('');

        // Don't just replace with {} because some InfoProps rely
        // on the reference to this.values:
        // wipeObj(this.values);
        return this;
    }

    /**
     * Iterate over the list of known properties and display them
     * if the specified object contains them.
     * @param {Object} event The related event so we can get position.
     * @param {N2TreeNode} node The node to examine.
     * @param {String} color Match the color of the node for the header/footer.
     * @param {Boolean} [isSolver = false] Whether to use solver properties or not.
     */
    update(event, node, color, isSolver = false) {
        if (!this.active) return;

        this.clear();

        this.name = node.absPathName;
        this.ribbonColor(color);

        if (DebugFlags.info && node.hasChildren()) {
            InfoPropDefault.addRowWithVal(this.tbody, 'Children', node.children.length);
            InfoPropDefault.addRowWithVal(this.tbody, 'Descendants', node.numDescendants);
            InfoPropDefault.addRowWithVal(this.tbody, 'Leaves', node.numLeaves);
            InfoPropDefault.addRowWithVal(this.tbody, 'Manually Expanded', node.manuallyExpanded.toString());
        }

        const propList = isSolver ? this.propListSolvers : this.propList;

        for (const prop of propList) {
            prop.addRow(this.tbody, node);
        }

        this.sizeToContent()
            .title(node.name)
            .moveNearMouse(event)
            .show();
    }
}

/**
 * Make a persistent copy of the NodeInfo panel and handle its drag/close events
 * @typedef PersistentNodeInfo
 */
class PersistentNodeInfo extends N2WindowDraggable {
    constructor(nodeInfo) {
        super('persistentNodeInfo-' + uuidv4(), '#' + nodeInfo.window.attr('id'));

        // Avoid just copying the reference because nodeInfo.values will be wiped:
        this.values = JSON.parse(JSON.stringify(nodeInfo.values));
        this.ui = nodeInfo.ui;

        this._setupShowMoreButtons(nodeInfo.name)
            ._setupCopyButtons()
            .showCloseButton()
            .show();
    }

    /** Set up event handlers for any "Show More" buttons in the panel */
    _setupShowMoreButtons(name) {
        const self = this;

        for (const valName in this.values) {
            if (this.values[valName].isTruncated) {
                this.window
                    .select(`button#${valName}.show_value_button`)
                    .on('click', c => {
                        ValueInfo.add(name, self.values[valName].val, self);
                    })
            }
        }

        return this;
    }

    /** Set up event handlers for any "Copy" buttons in the panel */
    _setupCopyButtons() {
        for (const valName in this.values) {
            this.window
                .select(`button#${valName}.copy_value_button`)
                .on('click', c => {
                    const copyText = d3.select("#input-for-pastebuffer");
                    copyText.text(this.values[valName].copyStr);
                    copyText.node().select();
                    document.execCommand('copy');
                })
        }

        return this;
    }
}
