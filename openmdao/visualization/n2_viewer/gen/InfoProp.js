// <<hpp_insert gen/WindowResizable.js>>

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
    canShow(node) { return (node.propExists(this.key) && String(node[this.key]) != '') }

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
     * @param {TreeNode} node Reference to the node that may have the property.
     */
    addRow(tbody, node) {
        if (this.canShow(node)) {
            InfoPropDefault.addRowWithVal(tbody, this.desc,
                this.output(node[this.key]), this.capitalize)
        }
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
}
/**
 * Output a human-readable number.
 * @typedef InfoPropNumber
 */
class InfoPropNumber extends InfoPropDefault {
    constructor(key, desc) {
        super(key, desc, false);
    }

    /** Truncate a possible float to something readable. */
    output(val) { return InfoPropDefault.valToString(val); }
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

/** Display a subsection of expression values in the info panel for ExecComps */
class InfoPropExpr extends InfoPropDefault {
    constructor(key, desc) {
        super(key, desc, false);
    }

    /**
     * There may be a list of expressions, so create a subsection in the table for them.
     * @param {Object} tbody D3 reference to an existing table body.
     * @param {TreeNode} node Reference to the node that may have the property.
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

    addRow(tbody, node) {
        if (this.canShow(node)) {
            InfoPropDefault.addRowWithVal(tbody, this.desc,
                this.output(node[this.key], node))
        }
    }

    /**
     * Convert the array to a string that can be displayed in the info panel. Save
     * the array value, the string, and a Python version of the string in the
     * values Object so it can be copied if the panel is pinned.
     * @param {Array} array The array to display and save.
     * @returns {String} A string representation of the array.
     */
    output(array, node) {
        if (array == null) { return 'Value too large to include in diagram'; }

        const valStr = InfoPropDefault.valToString(array);
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
            'min': node.val_min,
            'min_idx': node.val_min_indices,
            'max': node.val_max,
            'max_idx': node.val_max_indices,
            'str': valStr,
            'copyStr': InfoPropDefault.valToCopyString(array),
            'isTruncated': isTruncated
        }

        return html;
    }

    /** Make sure the node has the property and it's got a value. */
    canShow(node) { return node.propExists(this.key); }
}

/** Display connections associated with a node */
class InfoPropConns extends InfoPropDefault {
    constructor(key, desc) {
        super(key, desc, false);
    }

    addRow(tbody, node) {
        tbody.append('tr').append('th').attr('colspan', '3')
            .attr('class', 'options-header')
            .text(this.desc);

            /*
        // TODO: Add children
        for (const conn of node.connTargets) {
            const newRow = tbody.append('tr');

            newRow.append('td')
                .attr('scope', 'row')
                .text(node.path);

            newRow.append('td').text(' --> ');
    
            newRow.append('td').text(conn);
        }

        for (const conn of node.connSources) {
            const newRow = tbody.append('tr');
    
            newRow.append('td').text(conn);

            newRow.append('td').text(' --> ');

            newRow.append('td')
                .attr('scope', 'row')
                .text(node.path);
        } */
    }
}

InfoPropDefault.floatFormatter = d3.format('g');
