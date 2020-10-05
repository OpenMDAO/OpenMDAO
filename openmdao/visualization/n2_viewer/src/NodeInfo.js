/**
 * Manage info for each node metadata property
 * @typedef InfoPropDefault
 * @property {String} key The identifier of the property.
 * @property {String} desc The description (label) to display.
 * @property {Boolean} capitalize Whether to capitialize every word in the desc.
 */
class InfoPropDefault {
    constructor(key, desc, capitalize = false) {
        this.key = key;
        this.desc = desc;
        this.capitalize = capitalize;
    }

    /** Return the same message since this is the base class */
    output(msg) { return msg; }

    canShow(obj) { return (obj.propExists(this.key) && obj[this.key] != '') }
}

/**
 * Outputs a Yes or No to display. 
 * @typedef InfoPropYesNo
 */
class InfoPropYesNo extends InfoPropDefault {
    constructor(key, desc, capitalize = false, showIfFalse = false) {
        super(key, desc, capitalize);
        this.showIfFalse = showIfFalse;
    }

    /** Return Yes or No when given True or False */
    output(boolVal) { return boolVal ? 'Yes' : 'No'; }

    canShow(obj) { return (obj.propExists(this.key) && obj[this.key] != '' && this.showIfFalse == false) }
}

// Used to format that floats displayed
let val_float_formatter = d3.format("g");


/**
 * Convert an element to a string that is human readable.
 * @param {element} string, int,... The item to convert. Not an array!
 * @returns {str} the string of the converted element.
 */
function element_to_string(element) {
    let val_string;
    if (typeof element === 'number') {
        if (Number.isInteger(element)) {
            val_string = element.toString();
        } else { /* float */
            val_string = val_float_formatter(element);
        }
    } else {
        if (element === 'nan') {
            return element;
        } else {
            return JSON.stringify(element);
        }
    }
    return val_string;
}

/**
 * Convert an item to a string that is human readable.
 * @param {val} arr,string, int,... The item to convert.
 * @param {level} int The level of nesting in the display.
 * @returns {str} the string of the converted array.
 */
function val_to_string(val, level = 0) {
    if (!Array.isArray(val)) {
        return element_to_string(val);
    }
    let indent = ' '.repeat(level);
    let s = indent + '[';

    for (const element of val) {
        if (Array.isArray(element)) {
            s += val_to_string(element, level + 1);
        } else {
            s += element_to_string(element);
        }
        s += ' ';
    }
    if (val.length > 0) {
        s = s.slice(0, -1); // chop off the last space
    }
    s += ']\n';
    return s;
}

/**
 * Convert the value to a string that can be used in Python code.
 * @param {val} array,string,int,... The value to convert.
 * @returns {str} the string of the converted array.
 */
function val_to_copy_string(val) {
    if (!Array.isArray(val)) {
        return element_to_string(val);
    }
    let s = 'array([';
    for (const element of val) {
        if (Array.isArray(element)) {
            s += val_to_copy_string(element);
        } else {
            s += element_to_string(element);
        }
        s += ', ';
    }
    if (val.length > 0) {
        s = s.slice(0, -2); // chop off the last comma and space
    }
    s += '])';
    return s;
}

/**
 * Handles properties that are arrays.
 * @typedef InfoPropArray
 */
class InfoPropArray extends InfoPropDefault {
    constructor(key, desc, capitalize = false) {
        super(key, desc, capitalize);
    }

    /** Simply return the array value */
    output(array) {
        return array;
    }

}

/**
 * Manage the windows that display the values of variables
 * @typedef ValueInfoManager
 */
class ValueInfoManager {
    /**
     * Manage the value info windows.
     */
    constructor(ui) {
        this.ui = ui;
        this.valueInfoWindows = {};
    }

    add(name, val) {
        // Check to see if already exists before opening a new one
        if (!this.valueInfoWindows[name]) {
            let valueInfoBox = new ValueInfo(name, val, this.ui);
            this.valueInfoWindows[name] = valueInfoBox;
        }
    }

    remove(name) {
        this.valueInfoWindows[name].clear() // remove the DOM elements
        delete this.valueInfoWindows[name]; // remove the reference and let GC cleanup
    }
};

/**
 * Manage a window for displaying the value of a variable.
 * @typedef ValueInfo
 */
class ValueInfo {
    /**
     * Build a list of the properties we care about and set up
     * references to the HTML elements.
     * @param {str} name Variable name.
     * @param {Number} val Variable value.
     */
    constructor(name, val, ui) {
        this.name = name;
        this.val = val;

        /* Construct the DOM elements that make up the window */
        const top_container = d3.select('#node-value-containers');
        this.container = top_container.append('div').attr('class', 'node-value-container');
        this.header = this.container.append('div').attr('class', 'node-value-header');
        const close_button = this.header.append('span').attr('class', 'close-value-window-button').text('x');
        this.title = this.header.append('span').attr('class', 'node-value-title');
        this.table_div = this.container.append('div').attr('class', 'node-value-table-div');
        this.table = this.table_div.append('table').attr('class', 'node-value-table')
        this.container.append('div').attr('class', 'node-value-footer');
        const resizer_box = this.container.append('div').attr('class', 'node-value-resizer-box inactive-resizer-box')
        this.resizer_handle = resizer_box.append('p').attr('class', 'node-value-resizer-handle inactive-resizer-handle')

        close_button.on(
            'click',
            function () {
                ui.valueInfoManager.remove(name);
            }
        );

        this.update();
        this._setupDrag();
        this._setupResizerDrag();
    }

    clear() {
        const node = this.container.node();
        node.parentNode.removeChild(node);
    }

    update() {
        this.title.text("Initial value for " + this.name);

        // Capture the width of the header before the table is created
        // We use this to limit how small the window can be as the user resizes
        this.header_width = parseInt(this.header.style('width'));

        // Check to see if the data is a 2d array since the rest of the code assumes that it is an Array
        // If only 1d, make it a 2d with one row
        let val = this.val;
        if (!Array.isArray(val[0])) {
            val = [val];
        }

        // Construct the table displaying the variable value
        const tbody = this.table.append("tbody");
        const rows = tbody.selectAll('tr').data(val).enter().append('tr')
        const cells = rows.selectAll('td')
            .data(function (row) {
                return row;
            })
            .enter()
            .append('td')
            .text(function (d) {
                return val_float_formatter(d);
            })

        // Save the width and height of the table when it is fully
        // constructed. This will be used later to limit the resizing
        // of the window. No need to let the user resize to a size
        // larger than full size
        this.initial_width = parseInt(this.table.style('width'));
        this.initial_height = parseInt(this.table.style('height'));
    }

    /** Listen for the event to begin dragging the value the value window */
    _setupDrag() {
        const self = this;

        this.title.on('mousedown', function () {
            self.title.style('cursor', 'grabbing');
            const dragDiv = self.container;
            dragDiv.style('cursor', 'grabbing')
                // top style needs to be set explicitly before releasing bottom:
                .style('top', dragDiv.style('top'))
                .style('bottom', 'initial');

            self._startPos = [d3.event.clientX, d3.event.clientY]
            self._offset = [d3.event.clientX - parseInt(dragDiv.style('left')),
            d3.event.clientY - parseInt(dragDiv.style('top'))];

            const w = d3.select(window)
                .on("mousemove", e => {
                    dragDiv
                        .style('top', (d3.event.clientY - self._offset[1]) + 'px')
                        .style('left', (d3.event.clientX - self._offset[0]) + 'px');
                })
                .on("mouseup", e => {
                    dragDiv.style('cursor', 'grab');
                    w.on("mousemove", null).on("mouseup", null);

                });

            d3.event.preventDefault();
        })
    }

    /** Set up event handlers for grabbing the bottom corner and dragging */
    _setupResizerDrag() {
        const handle = this.resizer_handle;
        const body = d3.select('body');
        const tableDiv = this.table_div;

        handle.on('mousedown', e => {
            const startPos = {
                'x': d3.event.clientX,
                'y': d3.event.clientY
            };
            const startDims = {
                'width': parseInt(tableDiv.style('width')),
                'height': parseInt(tableDiv.style('height'))
            };
            body.style('cursor', 'nwse-resize')
                .on('mouseup', e => {
                    // Get rid of the drag event handlers
                    body.style('cursor', 'default')
                        .on('mousemove', null)
                        .on('mouseup', null);
                })
                .on('mousemove', e => {
                    let newWidth = d3.event.clientX - startPos.x + startDims.width;
                    let newHeight = d3.event.clientY - startPos.y + startDims.height;

                    // Do not let get it too big so that you get empty space
                    newWidth = Math.min(newWidth, this.initial_width);
                    newHeight = Math.min(newHeight, this.initial_height);

                    // Don't let it get too small or things get weird
                    newWidth = Math.max(newWidth, this.header_width);

                    tableDiv.style('width', newWidth + 'px');
                    tableDiv.style('height', newHeight + 'px');
                });

            d3.event.preventDefault();
        });

    }
}

// "Class" variable and function for ValueInfo
ValueInfo.TRUNCATE_LIMIT = 80;

/**
 * Based on the number of dimensions of the value,
 * indicate whether a value window display is needed or even practical
 * @param {val} int, float, list,... The variable value.
  * @returns {str} the string of the converted array.
*/
ValueInfo.canValueBeDisplayedInValueWindow = function (val) {
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
 * Manage a table containing all available metadata properties for
 * the currently active node, as well as whether the table is
 * visible or not.
 * @typedef NodeInfo
 */
class NodeInfo {
    /**
     * Build a list of the properties we care about and set up
     * references to the HTML elements.
     */
    constructor(ui) {
        this.propList = [
            new InfoPropDefault('absPathName', 'Absolute Name'),
            new InfoPropDefault('class', 'Class'),
            new InfoPropDefault('type', 'Type', true),
            new InfoPropDefault('dtype', 'DType'),

            new InfoPropDefault('units', 'Units'),
            new InfoPropDefault('shape', 'Shape'),
            new InfoPropYesNo('is_discrete', 'Discrete'),
            new InfoPropYesNo('distributed', 'Distributed'),
            new InfoPropArray('value', 'Value'),

            new InfoPropDefault('subsystem_type', 'Subsystem Type', true),
            new InfoPropDefault('component_type', 'Component Type', true),
            new InfoPropYesNo('implicit', 'Implicit'),
            new InfoPropYesNo('is_parallel', 'Parallel'),
            new InfoPropDefault('linear_solver', 'Linear Solver'),
            new InfoPropDefault('nonlinear_solver', 'Non-Linear Solver'),
            new InfoPropDefault('options', 'Options'),
            new InfoPropDefault('linear_solver_options', 'Linear Solver Options'),
            new InfoPropDefault('nonlinear_solver_options', 'Non-Linear Solver Options'),
        ];

        this.propListSolvers = [
            new InfoPropDefault('absPathName', 'Absolute Name'),
            new InfoPropDefault('linear_solver_options', 'Linear Solver Options'),
            new InfoPropDefault('nonlinear_solver_options', 'Non-Linear Solver Options'),
        ];

        this.ui = ui;
        this.table = d3.select('#node-info-table');
        this.container = d3.select('#node-info-container');
        this.thead = this.table.select('thead');
        this.tbody = this.table.select('tbody');
        this.toolbarButton = d3.select('#info-button');
        this.hidden = true;
        this.pinned = false;

        const self = this;
        this.pinButton = d3.select('#node-info-pin')
            .on('click', e => { self.unpin(); })
    }

    /** Make the info box visible if it's hidden */
    show() {
        this.toolbarButton.attr('class', 'fas icon-info-circle active-tab-icon');
        this.hidden = false;
        d3.select('#all_pt_n2_content_div').classed('node-data-cursor', true);
    }

    /** Make the info box hidden if it's visible */
    hide() {
        this.toolbarButton.attr('class', 'fas icon-info-circle');
        this.hidden = true;
        d3.select('#all_pt_n2_content_div').classed('node-data-cursor', false);
    }

    /** Toggle the visibility setting */
    toggle() {
        if (this.hidden) this.show();
        else this.hide();
    }

    pin() {
        this.pinned = true;
        this.pinButton.attr('class', 'info-visible');
    }

    unpin() {
        this.pinned = false;
        this.pinButton.attr('class', 'info-hidden');
        this.clear();
    }

    togglePin() {
        if (this.pinned) this.unpin();
        else this.pin();
    }

    _addPropertyRow(label, val, obj, capitalize = false) {
        if (!['Options', 'Linear Solver Options', 'Non-Linear Solver Options'].includes(label)) {
            const newRow = this.tbody.append('tr');

            const th = newRow.append('th')
                .attr('scope', 'row')
                .text(label)

            let nodeInfoVal = val;
            let td;
            if (label === 'Value') {
                if (val == null) {
                    td = newRow.append('td')
                        .html("Value too large to include in N2");
                } else {
                    let val_string = val_to_string(val)
                    let max_length = ValueInfo.TRUNCATE_LIMIT;
                    let isTruncated = val_string.length > max_length;
                    nodeInfoVal = isTruncated ?
                        val_string.substring(0, max_length - 3) + "..." :
                        val_string;

                    let html = nodeInfoVal;
                    if (isTruncated && ValueInfo.canValueBeDisplayedInValueWindow(val)) {
                        html += " <button type='button' class='show_value_button'>Show more</button>";
                    }
                    html += " <button type='button' class='copy_value_button'>Copy</button>";
                    td = newRow.append('td').html(html);

                    if (isTruncated && ValueInfo.canValueBeDisplayedInValueWindow(val)) {
                        let showValueButton = td.select('.show_value_button');
                        const self = this;
                        showValueButton.on('click', function () {
                            self.ui.valueInfoManager.add(self.name, val);
                        });
                    }
                    // Copy value button
                    let copyValueButton = td.select('.copy_value_button');
                    copyValueButton.on('click',
                        function () {
                            // This is the strange way you can get something on the clipboard
                            let copyText = document.querySelector("#input-for-pastebuffer");
                            copyText.value = val_to_copy_string(val);
                            copyText.select();
                            document.execCommand("copy");
                        }
                    );
                }
            }
            else {
                td = newRow.append('td')
                    .text(nodeInfoVal);
            }
            if (capitalize) td.attr('class', 'caps');
        } else {
            // Add Options to the Node Info table
            if (Object.keys(val).length !== 0) {
                if (label === 'Non-Linear Solver Options') {
                    label += ': ' + obj.nonlinear_solver.substring(3);
                }
                if (label === 'Linear Solver Options') {
                    label += ': ' + obj.linear_solver.substring(3);
                }
                const tr = this.tbody.append('th').text(label).attr('colspan', '2').attr('class', 'options-header');

                for (const key of Object.keys(val).sort()) {
                    const tr = this.tbody.append('tr');
                    const th = tr.append('th').text(key);
                    let v;
                    if (val[key] === null) {
                        v = "None";
                    } else {
                        v = val[key];
                    }
                    const td = tr.append('td').text(v);
                }
            }
        }
    }

    /**
     * Iterate over the list of known properties and display them
     * if the specified object contains them.
     * @param {Object} event The related event so we can get position.
     * @param {N2TreeNode} obj The node to examine.
     * @param {N2TreeNode} color The color to make the title bar.
     */
    update(event, obj, color = '#42926b') {
        if (this.hidden || this.pinned) return;

        this.clear();
        // Put the name in the title
        this.table.select('thead th')
            .style('background-color', color)
            .text(obj.name);

        this.name = obj.absPathName;

        this.table.select('tfoot th')
            .style('background-color', color);

        if (obj.promotedName) {
            this._addPropertyRow('Promoted Name', obj.promotedName, obj);
        }

        if (DebugFlags.info && obj.hasChildren()) {
            this._addPropertyRow('Children', obj.children.length, obj);
            this._addPropertyRow('Descendants', obj.numDescendants, obj);
            this._addPropertyRow('Leaves', obj.numLeaves, obj);
            this._addPropertyRow('Manually Expanded', obj.manuallyExpanded.toString(), obj)
        }

        for (const prop of this.propList) {
            if (prop.key === 'value') {
                if (obj.hasOwnProperty('value')) {
                    this._addPropertyRow(prop.desc, prop.output(obj[prop.key]), obj, prop.capitalize)
                }
            } else {
                if (prop.canShow(obj)) {
                    this._addPropertyRow(prop.desc, prop.output(obj[prop.key]), obj, prop.capitalize)
                }
            }
        }

        // Solidify the size of the table after populating so that
        // it can be positioned reliably by move().
        this.table
            .style('width', this.table.node().scrollWidth + 'px')
            .style('height', this.table.node().scrollHeight + 'px')

        this.move(event);
        this.container.attr('class', 'info-visible');
    }

    /**
     * Iterate over the list of known properties of solvers and display them
     * if the specified object contains them.
     * @param {Object} event The related event so we can get position.
     * @param {N2TreeNode} obj The node to examine.
     * @param {N2TreeNode} color The color to make the title bar.
     */
    update_solver(event, obj, color = '#42926b') {
        if (this.hidden || this.pinned) return;

        this.clear();
        // Put the name in the title
        this.table.select('thead th')
            .style('background-color', color)
            .text(obj.name);

        this.name = obj.absPathName;

        this.table.select('tfoot th')
            .style('background-color', color);

        if (obj.promotedName) {
            this._addPropertyRow('Promoted Name', obj.promotedName, obj);
        }

        if (DebugFlags.info && obj.hasChildren()) {
            this._addPropertyRow('Children', obj.children.length, obj);
            this._addPropertyRow('Descendants', obj.numDescendants, obj);
            this._addPropertyRow('Leaves', obj.numLeaves, obj);
            this._addPropertyRow('Manually Expanded', obj.manuallyExpanded.toString(), obj)
        }

        // for (const prop of this.propList) {
        for (const prop of this.propListSolvers) {
            // if (obj.propExists(prop.key) && obj[prop.key] != '') {
            if (prop.key === 'value') {
                if (obj.hasOwnProperty('value')) {
                    this._addPropertyRow(prop.desc, prop.output(obj[prop.key]), obj, prop.capitalize)
                }
            } else {
                if (prop.canShow(obj)) {
                    this._addPropertyRow(prop.desc, prop.output(obj[prop.key]), obj, prop.capitalize)
                }
            }
        }

        // Solidify the size of the table after populating so that
        // it can be positioned reliably by move().
        this.table
            .style('width', this.table.node().scrollWidth + 'px')
            .style('height', this.table.node().scrollHeight + 'px')

        this.move(event);
        this.container.attr('class', 'info-visible');
    }

    /** Wipe the contents of the table body */
    clear() {
        if (this.hidden || this.pinned) return;
        this.container
            .attr('class', 'info-hidden')
            .style('width', 'auto')
            .style('height', 'auto');

        this.table
            .style('width', 'auto')
            .style('height', 'auto');

        this.tbody.html('');
    }

    /**
     * Relocate the table to a position near the mouse
     * @param {Object} event The triggering event containing the position.
     */
    move(event) {
        if (this.hidden || this.pinned) return;
        const offset = 30;

        // Mouse is in left half of window, put box to right of mouse
        if (event.clientX < window.innerWidth / 2) {
            this.container.style('right', 'auto');
            this.container.style('left', (event.clientX + offset) + 'px')
        }
        // Mouse is in right half of window, put box to left of mouse
        else {
            this.container.style('left', 'auto');
            this.container.style('right', (window.innerWidth - event.clientX + offset) + 'px')
        }

        // Mouse is in top half of window, put box below mouse
        if (event.clientY < window.innerHeight / 2) {
            this.container.style('bottom', 'auto');
            this.container.style('top', (event.clientY - offset) + 'px')
        }
        // Mouse is in bottom half of window, put box above mouse
        else {
            this.container.style('top', 'auto');
            this.container.style('bottom', (window.innerHeight - event.clientY - offset) + 'px')
        }
    }
}
