// <<hpp_insert gen/Dimensions.js>>

/**
 * Manage display info associated with the node.
 * @typedef NodeDisplayData
 */
class NodeDisplayData {
    constructor() {
        this.nameWidthPx = 1; // Width of the label in pixels as computed by Layout
        this.numLeaves = 0; // Set by Layout
        this.minimized = false; // When true, do not draw children
        this.hidden = false; // Do add to matrix at all
        this.filtered = false; // Node is a child to be shown w/partially collapsed parent
        this.filterParent = null; // When filtered, reference to FilterNode container
        this.manuallyExpanded = false; // Node was pre-collapsed but expanded by user

        this.dims = new Dimensions({ x: 1e-6, y: 1e-6, width: 1, height: 1 });
        this.dims.preserve();
    }
}

/**
 * Essentially the same as a JSON object from the model tree,
 * with some utility functions.
 * @typedef {TreeNode}
 * @property {Object} draw Information used for node display.
 * @property {Number} depth The index of the column this node appears in.
 */
class TreeNode {
    /**
     * Absorb all the properties of the provided JSON object from the model tree.
     * @param {Object} origNode The node to work from.
     * @param {Object} attribNames Names of variables in the model.
     */
    constructor(origNode, attribNames, parent) {
        // Merge all of the props from the original JSON tree node to us.
        Object.assign(this, origNode);
        if (attribNames.descendants != 'children') {
            this.children = this[attribNames.descendants];
            delete this[attribNames.descendants];
        }

        this.parent = parent;

        // All nodes connected to this one as either a source or target
        this.connSources = new Set();
        this.connTargets = new Set();

        this.sourceParentSet = new Set();
        this.targetParentSet = new Set();

        // Node display data.
        this.draw = this._newDisplayData()

        this.childNames = new Set(); // Set by ModelData
        this.depth = -1; // Set by ModelData
        this.id = -1; // Set by ModelData
        this.path = ''; // Set by ModelData
        this.numDescendants = 0; // Set by ModelData

        this.rootIndex = -1;
    }

    _newDisplayData() {
        return new NodeDisplayData();
    }

    /** In the matrix grid, draw a box around variables that share the same boxAncestor() */
    boxAncestor() {
        return this.isInputOrOutput()? this.parent : null;
    }

    // Accessor functions for this.draw.minimized - whether or not to draw children
    minimize() { this.draw.minimized = true; return this; }
    expand() { this.draw.minimized = false; return this; }

    // Accessor functions for this.draw.hidden - whether to draw at all
    hide() { this.draw.hidden = true; return this; }
    show() { this.draw.hidden = false; return this; }

    /**
     * Create a backup of our position and other info.
     * @param {number} leafNum Identify this as the nth leaf of the tree
     */
    preserveDims(leafNum) {
        this.draw.dims.preserve();

        if (this.rootIndex < 0) this.rootIndex = leafNum;
        this.prevRootIndex = this.rootIndex;
    }

    /**
     * Determine if a children array exists and has members.
     * @param {string} [childrenPropName = 'children'] Usually "children", but
     *   some subclasses may have additional child arrays.
     * @return {boolean} True if the children property is an Array and length > 0.
     */
    hasChildren(childrenPropName = 'children') {
        return (Array.isPopulatedArray(this[childrenPropName]));
    }

    /** True if this.type is 'input' or 'unconnected_input'. */
    isInput() { return this.type.match(/^(input|unconnected_input)$/); }

    /** True if this is an input and connected. */
    isConnectedInput() { return (this.type == 'input'); }

    /** True if this an input and unconnected. */
    isUnconnectedInput() { return (this.type == 'unconnected_input'); }

    /** True if this.type is 'output'. */
    isOutput() { return (this.type == 'output'); }

    /** True if this is the root node in the model */
    isRoot() { return (this.type == 'root'); }

    /** True if this.type is 'input', 'unconnected_input', or 'output'. */
    isInputOrOutput() { return this.type.match(/^(output|input|unconnected_input)$/); }

    /** True if it's a group */
    isGroup() { return this.type == 'group'; }

    /** True if this is a "fake" node that manages filtered children. */
    isFilter() { return (this.type == 'filter'); }

    /** True if this node can use filters (always false for base class) */
    canFilter() { return false; } 

    /** Not connectable if this is an input group or parents are minimized. */
    isConnectable() {
        if (this.isInputOrOutput() && !(this.hasChildren() ||
            this.parent.draw.minimized)) return true;

        return this.draw.minimized;
    }

    /** Return false if the node is hidden, or filtered */
    isVisible() {
        return !(this.draw.hidden || this.draw.filtered);
    }

    /** Return true if there are no children */
    isLeaf() {
        return (!this.hasChildren());
    }

    /** True if node is not hidden and has no visible children */
    isVisibleLeaf() {
        if (this.draw.hidden || this.draw.filtered) return false; // Any explicitly hidden node
        if (this.isInputOrOutput()) return !this.draw.filtered; // Variable
        if (this.isLeaf()) return true; // Group w/out children
        return this.draw.minimized; // Collapsed non-variable
    }

    /** True if this is a variable and will be displayed as partially collapsed. */
    isFilteredVariable() {
        return (this.isInputOrOutput() && this.draw.filtered);
    }

    /**
     * Look for the supplied node in the set of child names.
     * @param {TreeNode} compareNode The node to look for.
     * @param {Boolean} includeSelf If true, return true if compareNode is us.
     * @returns {Boolean} True if a match is found, otherwise false.
     */
    hasNodeInChildren(compareNode, includeSelf = false) {
        if (includeSelf && compareNode === this) return true;
        return this.childNames.has(compareNode.path);
    }

    /** Look for the supplied node in the parentage of this one.
     * @param {TreeNode} compareNode The node to look for.
     * @param {TreeNode} [parentLimit = null] Stop searching at this common parent.
     * @returns {Boolean} True if the node is found, otherwise false.
     */
    hasParent(compareNode, parentLimit = null, includeSelf = false) {
        if (includeSelf && compareNode === this) return true;
        for (let obj = this.parent; obj != null && obj !== parentLimit; obj = obj.parent) {
            if (obj === compareNode) {
                return true;
            }
        }

        return false;
    }

    /**
     * Look for the supplied node in the lineage of this one.
     * @param {TreeNode} compareNode The node to look for.
     * @param {TreeNode} [parentLimit = null] Stop searching at this common parent.
     * @returns {Boolean} True if the node is found, otherwise false.
     */
    hasNode(compareNode, parentLimit = null) {
        if (this.type == 'root') return true;

        if (this === compareNode) return true;

        // Check parents first.
        if (this.hasParent(compareNode, parentLimit)) return true;

        return this.hasNodeInChildren(compareNode);
    }

    /**
     * Find the closest parent shared by two nodes; farthest should be tree root.
     * @param {TreeNode} other Another node to compare parents with.
     * @returns {TreeNode} The first common parent found.
     */
    nearestCommonParent(other) {
        for (let myParent = this.parent; myParent != null; myParent = myParent.parent)
            for (let otherParent = other.parent; otherParent != null; otherParent = otherParent.parent)
                if (myParent === otherParent) return myParent;

        // Should never get here because root is parent of all
        debugInfo("No common parent found between two nodes: ", this, other);
        return null;
    }

    _preCollapseDepth() { return Precollapse.grpDepthStart; }

    /**
     * If the node has a lot of descendants and it wasn't manually expanded,
     * minimize it.
     * @param {Number} depthCount The number of nodes at the next depth down.
     * @returns {Boolean} True if minimized here, false otherwise.
     */
    minimizeIfLarge(depthCount) {
        if (!(this.isRoot() || this.draw.manuallyExpanded) &&
            (this.depth >= this._preCollapseDepth() &&
                this.numDescendants > Precollapse.threshold &&
                this.children.length > Precollapse.children - this.depth &&
                depthCount > Precollapse.depthLimit)) {
            debugInfo(`Precollapsing node ${this.path}`)
            this.minimize();
            return true;
        }

        return false;
    }

    /**
     * Convert a path to a string that's safe to use as an HTML id. Escapes
     * space, greater-than, less-than, period, and colon characters.
     * @param {String} path The name to convert.
     * @returns {String} The HTML-safe id.
     */
    static pathToId(path) {
        return path.replace(/[\.<> :|]/g, function (c) {
            return {
                ' ': '__',
                '<': '_LT',
                '>': '_GT',
                '.': '_',
                ':': '-',
                '|': '--'
            }[c];
        })
    }

    toId() { return TreeNode.pathToId(this.path); }

    _insertAsLastInput(newChild) {
        if (!this.hasChildren()) return;

        let idx = -1;
        for (idx = 0; idx < this.children.length - 1; idx++ ) {
            if (this.children[idx+1].isOutput()) break;
        }

        this.children.splice(idx + 1, 0, newChild);
    }

    isFilter() { return false; } // Always false in base class
    hasFilters() { return ('filter' in this); } // True if we contain filters
    isInputFilter() { return false; } // Always false in base class
    isOutputFilter() { return false; } // Always false in base class

    /**
     * Create a simple object that can be used to save state to a file.
     * @returns {Object} Reference to required info.
     */
    getStateForSave() {
        return {
            'minimized': this.draw.minimized,
            'manuallyExpanded': this.draw.manuallyExpanded,
            'hidden': this.draw.hidden,
            'filtered': this.draw.filtered
        };
    }

    /**
     * Provided with loaded state information, update our settings.
     * @param {Object} state The state loaded from file.
     */
    setStateFromLoad(state) {
        this.draw.minimized = state.minimized;
        this.draw.manuallyExpanded = state.manuallyExpanded;
        this.draw.hidden = state.hidden;
        this.filterSelf(state.filtered);
    }

    getTextName() { return this.name; }
}

/**
 * Special TreeNode subclass whose children are filtered variables of the parent node.
 * Not intended to be used outside of a FilterCapableNode.
 * @typedef FilterNode
 */
class FilterNode extends TreeNode {
    /**
     * Give ourselves a special name and the "filter" type.
     * @param {TreeNode} parent The node that we are filtering variables for.
     * @param {String} suffix Either "inputs" or "outputs".
     */
    constructor(parent, attribNames, suffix) {
        super(
            {
                name: `${parent.name}_FILTER_${suffix}`,
                type: 'filter'
            },
            attribNames
        )

        this.parent = parent;
        this.hide();
        this.minimize();
        this.suffix = suffix;
    }

    getTextName() {
        return `Filtered ${this.suffix[0].toUpperCase() + this.suffix.slice(1)}`;
    }

    /**
     * Add a node to our filtered children, update its state, and make ourselves visible.
     * @param {TreeNode} node Reference to the node to filter.
     */
    add(node) {
        if (!this.hasChildren()) { this.children = []; }
        this.children.push(node);
        this.childNames.add(node.path);
        this.numDescendants += 1;
        node.doFilter(this);
        this.show();
    }
    
    /**
     * Update the node's state and remove it from our children. If nothing is left in
     * the children array, delete it and hide ourselves.
     * @param {TreeNode} node Reference to the node to unfilter.
     * @returns {Boolean} True if the node was found in children, otherwise false.
     */
    del(node) {
        node.undoFilter(); // Reset state regardless of being found
        if (this.hasChildren()) {
            const idx = this.children.indexOf(node);
            if (idx >= 0) {
                this.children.splice(idx);
                this.childNames.delete(node);
                this.numDescendants -= 1;
                if (this.children.length == 0) {
                    this.children = null;
                    this.hide();
                }
                return true;
            }
        }
        return false;
    }

    /** Set the state of all children to unfiltered and delete the array */
    wipe() {
        if (this.hasChildren()) {
            for (const child of this.children) { child.undoFilter(); }
            this.children = null;

            this.numDescendants = 0;
            this.childNames.clear();
            this.childNames.add(this.path);
            this.hide();
        }
    }

    /**
     * Determine if the referenced node is in our array of children.
     * @param {TreeNode} child The node to find.
     * @returns {Boolean} True if the child was found, false otherwise.
     */
    hasChild(child) {
        return (this.hasChildren()? this.children.indexOf(child) >= 0 : false);
    }

    /** Return the length of the children array or 0 if it doesn't exist. */
    get count() { return (this.hasChildren()? this.children.length : 0); }
    
    /** Don't expand, always stay minimized. */
    expand() { return this; }

    /** Only show if there are filtered nodes stored as children */
    show() {
        if (this.hasChildren()) { super.show(); }
        return this;
    }

    isFilter() { return true; } // Always true for the TreeNode class
    hasFilters() { return false; }
    isInputFilter() { return this.suffix == 'inputs'; } // True if this manages input filters
    isOutputFilter() { return this.suffix == 'outputs'; } // True if this manages output filters

    /**
     * For normal nodes, targetParentSet and sourceParentSet are built when the model
     * is initialized. Since filters are created after that, _genParentSet() generates a
     * set dynamically based on the specified parent sets of its children.
     * @param {String} setName Either "source" or "target"
     * @returns {Set} The merged contents of the parent sets of all children.
     */
    _genParentSet(setName) {
        const tmpSet = new Set();
        const setPropName = `${setName}ParentSet`;

        if (this.hasChildren()) {
            for (const child of this.children) {
                for (const childParent of child[setPropName]) {
                    tmpSet.add(childParent);
                }
            }
        }

        return tmpSet;
    }

    get sourceParentSet() { return this._genParentSet('source'); }
    set sourceParentSet(val) { return val; }

    get targetParentSet() { return this._genParentSet('target'); }
    set targetParentSet(val) { return val; }
}

/**
 * Add the ability of a TreeNode to contain FilterNodes managing its input/output variables.
 * The TreeNode base class has some accessor methods that refer to filters, but no
 * ability to actually filter anything or be filtered by itself.
 * @typedef FilterCapableNode
 * @property {FilterNode} filter.inputs Children that are inputs to be viewed as collapsed.
 * @property {FilterNode} filter.outputs Children that are outputs to be viewed as collapsed.
 */
class FilterCapableNode extends TreeNode {
    constructor(origNode, attribNames, parent) {
        super(origNode, attribNames, parent);
    }

    // Accessor functions for this.draw.filtered - whether a variable is shown in collapsed form
    doFilter(filterNode) { this.draw.filtered = true; this.draw.filterParent = filterNode; }
    undoFilter() { this.draw.filtered = false; this.draw.filterParent = null; }

    canFilter() { return true; }

    /** If this node has children add special children that can hold filtered variables */
    addFilterChild(attribNames) {
        if (this.hasChildren()) {

            // Separate FilterNodes are added for inputs and outputs so
            // they can be inserted at the correct place in the diagram.
            this.filter = {
                inputs: new FilterNode(this, attribNames, 'inputs'),
                outputs: new FilterNode(this, attribNames, 'outputs')
            };
            this._insertAsLastInput(this.filter.inputs);
            this.children.push(this.filter.outputs);
        }
    }

    /** Add ourselves to the correct parental filter */
    addSelfToFilter() {
        if (this.isInput()) { this.parent.filter.inputs.add(this); }
        else if (this.isOutput()) { this.parent.filter.outputs.add(this); }
    }

    /** Remove ourselves from the correct parental filter */
    removeSelfFromFilter() {
        if (this.isInput()) { this.parent.filter.inputs.del(this); }
        else if (this.isOutput()) { this.parent.filter.outputs.del(this); }
    }

    getFilterList() { return [ this.filter.inputs, this.filter.outputs]; }

    wipeFilters() {
        this.filter.inputs.wipe();
        this.filter.outputs.wipe();
    }

    addToFilter(node) {
        if (node.isInput()) { this.filter.inputs.add(node); }
        else { this.filter.outputs.add(node); }
    }

    /**
     * Filter ourselves based on the supplied filter state.
     * @param {Boolean} filtered Whether to filter or not.
     * @return {Boolean} The newly set state.
     */
    filterSelf(filtered) {
        if (filtered) { this.addSelfToFilter(); }
        else { this.removeSelfFromFilter(); }

        return this.draw.filtered;
    }

}
