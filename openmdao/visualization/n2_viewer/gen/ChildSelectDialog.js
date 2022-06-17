// <<hpp_insert gen/WindowDraggable.js>>

/**
 * Manage a window that allows the user to select which variables to display.
 * TODO: Make column headers sticky, improve autosizing
 * without using sizeToContent(3,30)
 * @typedef ChildSelectDialog
 */
class ChildSelectDialog extends WindowDraggable {
    /**
     * Setup the basic structure of the variable selection dialog.
     * @param {TreeNode} node The node to examine the variables of.
     * @param {String} color The color to make the window header/footer ribbons.
     * @param {Diagram} diag The diagram object to work with.
     */
     constructor(e, node, color, diag) {
        super('childSelect-' + node.toId());
        this.node = node;
        this.nodeColor = color;
        this.diag = diag;
        this.scrollbarWidth = 14;
        this.searchTerm = ('searchTerm' in node)? node.searchTerm : null;

        this.ribbonColor(this.nodeColor);
        
        // Don't do anything else if the node has no variables
        if ( ! this._fetchVarNames() ) { this.close(); }
        else { this._initialSetup(e); }
    }
    
    /**
     * Find all the children of the node that are variables.
     * @returns {Boolean} True if any variables were found, otherwise false.
     */
    _fetchVarNames() {
        // Only add children that are variables.
        let foundVariables = false;

        this.varNames = {};
        this.varNameArr = []; // For sorting purposes

        for (const child of this.node.children) {
            if (!child.isInputOrOutput()) { continue; }
            foundVariables = true;

            // Use getTextName() because Auto-IVC variable names are not usually descriptive.
            const varName = child.getTextName();
            this.varNames[varName] = child;
            this.varNameArr.push(varName)

        }

        return foundVariables;
    }

    /**
     * Configure the window structure, then call repopulate() to list the variable names.
     */
    _initialSetup(e) {
        const self = this;

        this.minWidth = 300;
        this.minHeight = 100;
        this.theme('child-select');
        this.title(this.node.name);
        this.copyBody('#variable-selection-body');

        this.headerTable = this.body.select('table.header');
        const topRow = this.headerTable.select('tr');
        const childName = topRow.select('th:first-child');

        childName.select('span.sort-up') // Up arrow
            .on('click', e => {
                self.varNameArr.sort();
                self.repopulate();
            });

        childName.select('span.sort-down') // Down arrow
            .on('click', e => {
                self.varNameArr.sort();
                self.varNameArr.reverse();
                self.repopulate();
            });

        this.tableContainer = this.body.select('div.table-container');
        this.table = this.tableContainer.select('table.variables');
        const UA = navigator.userAgent;
        if (! /Chrom/.test(UA)) {          
            // Chrome puts the scrollbar outside the element, other browsers inside
            this.table.style('margin-right', `${this.scrollbarWidth}px`);
        }

        this.tbody = this.table.select('tbody');

        // Search
        this.searchContainer = this.body.select('div.search-container');
        this.searchBox = this.searchContainer.select('input');
        if (this.searchTerm) { this.searchBox.property('value', this.searchTerm); }
        this.searchBox.on('keyup', self.updateSearch.bind(self));

        // Clear the search box and repopulate the variable list by clicking on the X button
        this.searchContainer.select('.search-clear').on('click', e => {
            self.searchTerm = '';
            self.searchBox.property('value', '');
            self.updateSearch(e, true);
        });

        // Execute the search by clicking on the arrow button
        this.searchContainer.select('.search-perform').on('click', e => {
            self.updateSearch(e, true);
        });

        this.buttonContainer = this.body.select('div.button-container');

        // Apply button
        this.buttonContainer.select('button.select-all-variables')
            .on('click', self.selectAll.bind(self));

        // Select None button
        this.buttonContainer.select('button.select-no-variables')
            .on('click', self.selectNone.bind(self));

        // Select All button
        this.buttonContainer.select('button.apply-variable-selection')
            .on('click', self.apply.bind(self));

        this.scrollbarSpacer = topRow.select('th.scrollbar-spacer');
        this.repopulate();
        this.resize();
        this.modal(true)
            .moveNearMouse(e)
            .show();
    }

    /** Reset the array of hidden vars from the node's array if it exists. */
    _initHiddenVars() {
        this.hiddenVars = [];
        this.existingHiddenVars = false;

        for (const filter of this.node.getFilterList()) {
            if (filter.count > 0) {
                for (const child of filter.children) {
                    this.hiddenVars.push(child);
                };
                this.existingHiddenVars = true;
            }
        }
    }

    /** Add all the variables, their display status, and the control buttons. */
    repopulate() {
        const self = this;
        this.tbody.html('');
        
        // If a search term was used, treat it as a regular expression.
        const matchRe = this.searchTerm? new RegExp(this.searchTerm, 'i') : null;
        this._initHiddenVars();
        this.foundSearchVars = [];

        let isEven = true;
        this.varCount = 0;
        for (const varName of this.varNameArr) {
            const child = this.varNames[varName];

            if (matchRe) {
                // Variable names not matching the regexp are hidden.
                if (! matchRe.test(varName)) { 
                    this.hiddenVars.push(child);
                    continue;
                }
                // Matching variable names are remembered.
                else {
                    this.foundSearchVars.push(child);
                }
            }
            
            this.varCount++;

            // Alternate row colors:
            const row = this.tbody.append('tr').attr('class', isEven? 'even' : 'odd');
            isEven = !isEven;

            row.append('td').attr('class', 'varname').text(varName);
            const checkId = `${child.toId()}-visible-check`;
            if (child.isFilteredVariable() && ! this.existingHiddenVars) { this.hiddenVars.push(child); }

            // Add a checkbox. When checked, the variable will be displayed.
            row.append('td').attr('class', 'varvis')
                .append('input')
                .attr('type', 'checkbox')
                .property('checked', !child.isFilteredVariable())
                .attr('id', checkId)
                .on('change', e => {
                    const isVisible = d3.select(`#${checkId}`).property('checked');
                    if (!isVisible) { this.hiddenVars.push(child); }
                    else { 
                        const idx = this.hiddenVars.indexOf(child);
                        if (idx > -1 ) { this.hiddenVars.splice(idx, 1); }
                        else { console.warn('Could not find child in hiddenVars array.', child); }
                    }
                })
        }

        if (this.varCount == 0) {
            this.tbody.append('tr')
                .append('td')
                .style('text-align', 'center')
                .attr('colspan','2')
                .text('No matching variables found.')
        }

        return this;
    }

    /**
     * If the container scroll height is larger than the visible height the scrollbar is there.
     * @returns {Boolean} True if the scrollbar is visible, false otherwise.
     */
    scrollbarIsVisible() {
        return (this.tableContainer.node().scrollHeight > this.tableContainer.node().clientHeight);
    }

    /** Clicking Apply closes the dialog and updates the diagram. */
    apply() {
        if (this.hiddenVars.length == this.node.children.length || this.varCount == 0) {
            // If every variable was hidden, just collapse the node if it's expanded
            if (! this.node.draw.minimized) { this.diag.ui.rightClick(this.node); }
        }
        else {
            this.diag.ui.rightClickedNode = this.node;
            this.diag.ui.addBackButtonHistory();

            this.node.searchTerm = this.searchTerm;
            this.node.wipeFilters();

            for (const child of this.hiddenVars) {
                this.node.addToFilter(child);
            }
            
            if (this.node.draw.minimized) {
                // If node itself is collapsed, expand it
                this.node.draw.manuallyExpanded = true;
                this.node.expand();
            }
            this.diag.update();
        }
        this.close();
    }

    /**
     * Make all variable names visible. If a search term is active, make all
     * matching variable names visible.
     */
    selectAll() {
        d3.selectAll('.window-theme-child-select input[type="checkbox"]')
            .property('checked', true);
        this.hiddenVars = [];

        if (this.foundSearchVars.length > 0) {
            // Select variables not found by search
            for (const child of this.node.children) {
                if (child.isFilter()) continue;
                if (this.foundSearchVars.indexOf(child) < 0) {
                    this.hiddenVars.push(child);
                }
            }
        }
    }

    /** Hide all variable names */
    selectNone() {
        d3.selectAll('.window-theme-child-select input[type="checkbox"]')
            .property('checked', false);
        this.hiddenVars = [];

        for (const child of this.node.children) {
            if (child.isFilter()) continue;
            this.hiddenVars.push(child);
        }

    }

    /** Adjust the size of the window body object based on the size of their contents. */
    resize() {
        const headerTableNode = this.headerTable.node(),
            searchConNode = this.searchContainer.node(),
            newHeight =
                `${headerTableNode.scrollHeight +
                this.tableContainer.node().clientHeight +
                searchConNode.scrollHeight +
                this.buttonContainer.node().scrollHeight + 7}px`,
            newWidth = `${this.table.node().scrollWidth + this.scrollbarWidth}px`;

        this.body.style('height', newHeight).style('width', newWidth);
        this.tableContainer.style('width', newWidth);
        this.headerTable.style('width', newWidth);

        if (this.varCount > 0) {
            this.headerTable.select('th.varname')
                .style('width', this.table.select('td.varname').style('width'));
            this.headerTable.select('th.varvis')
                .style('width', this.table.select('td.varvis').style('width'));
        }

        this.searchContainer.style('width', newWidth);

        const displayScrollbar = this.scrollbarIsVisible()? null : 'none';
        this.scrollbarSpacer.style('display', displayScrollbar);
        
        this.sizeToContent(3,4);

        return this;
    }

    /** Update the variable list based on the provided search term. */
    updateSearch(e, clicked = false) {
        if (e.keyCode == 13 || clicked) {
                this.searchTerm = this.searchBox.property('value');
                this.repopulate();
                this.resize();
        }

        return this;
    }
}
