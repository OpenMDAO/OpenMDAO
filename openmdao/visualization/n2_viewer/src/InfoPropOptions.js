// <<hpp_insert gen/InfoProp.js>>

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
     * @param {TreeNode} node Reference to the node that may have the property.
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

