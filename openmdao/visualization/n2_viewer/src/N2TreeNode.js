/**
 * Essentially the same as a JSON object from the model tree,
 * with some utility functions.
 * @typedef {N2TreeNode}
 * @property {Object} draw Information used for node display.
 * @property {Number} depth The index of the column this node appears in.
 */
class N2TreeNode {

    /**
     * Absorb all the properties of the provided JSON object from the model tree.
     * @param {Object} origNode The node to work from.
     */
    constructor(origNode) {
        // Merge all of the props from the original JSON tree node to us.
        Object.assign(this, origNode);

        this.sourceParentSet = new Set();
        this.targetParentSet = new Set();

        // Node display data
        this.draw = {
            nameWidthPx: 1, // Width of the label in pixels as computed by N2Layout
            nameSolverWidthPx: 1, // Solver-side label width pixels as computed by N2Layout
            numLeaves: 0, // Set by N2Layout
            minimized: false, // When true, do not draw children
            hidden: false, // Do add to matrix at all
            filtered: false, // Node is a child to be shown w/partially collapsed parent
            filterParent: null, // When filtered, reference to N2FilterNode container
            manuallyExpanded: false, // Node was pre-collapsed but expanded by user
            dims: { x: 1e-6, y: 1e-6, width: 1, height: 1 },
            prevDims: { x: 1e-6, y: 1e-6, width: 1e-6, height: 1e-6 },
            solverDims: { x: 1e-6, y: 1e-6, width: 1, height: 1 },
            prevSolverDims: { x: 1e-6, y: 1e-6, width: 1e-6, height: 1e-6 }
        }

        this.childNames = new Set(); // Set by ModelData
        this.depth = -1; // Set by ModelData
        this.parent = null; // Set by ModelData
        this.id = -1; // Set by ModelData
        this.absPathName = ''; // Set by ModelData
        this.numDescendants = 0; // Set by ModelData

        // Solver names may be empty, so set them to "None" instead.
        if (this.linear_solver == "") this.linear_solver = "None";
        if (this.nonlinear_solver == "") this.nonlinear_solver = "None";

        this.rootIndex = -1;

    }

    // Accessor functions for this.draw.minimized - whether or not to draw children
    minimize() { this.draw.minimized = true; return this; }
    expand() { this.draw.minimized = false; return this; }

    // Accessor functions for this.draw.hidden - whether to draw at all
    hide() { this.draw.hidden = true; return this; }
    show() { this.draw.hidden = false; return this; }

    // Accessor functions for this.draw.filtered - whether a variable is shown in collapsed form
    doFilter(filterNode) { this.draw.filtered = true; this.draw.filterParent = filterNode; }
    undoFilter() { this.draw.filtered = false; this.draw.filterParent = null; }

    /**
     * Create a backup of our position and other info.
     * @param {boolean} solver Whether to use .dims or .solverDims.
     * @param {number} leafNum Identify this as the nth leaf of the tree
     */
    preserveDims(solver, leafNum) {
        const dimProp = solver ? 'dims' : 'solverDims';
        const prevDimProp = solver ? 'prevDims' : 'prevSolverDims';

        Object.assign(this.draw[prevDimProp], this.draw[dimProp]);

        if (this.rootIndex < 0) this.rootIndex = leafNum;
        this.prevRootIndex = this.rootIndex;
    }

    /**
     * Determine if the children array exists and has members.
     * @param {string} [childrenPropName = 'children'] Usually children, but
     *   sometimes 'subsystem_children'
     * @return {boolean} True if the children property is an Array and length > 0.
     */
    hasChildren(childrenPropName = 'children') {
        return (Array.isPopulatedArray(this[childrenPropName]));
    }

    /** True if this.type is 'input' or 'unconnected_input'. */
    isInput() { return this.type.match(inputRegex); }

    /** True if this is an input and connected. */
    isConnectedInput() { return (this.type == 'input'); }

    /** True if this an input and unconnected. */
    isUnconnectedInput() { return (this.type == 'unconnected_input'); }

    /** True if this is an input whose source is an auto-ivc'd output */
    isAutoIvcInput() { return (this.type == 'autoivc_input'); }

    /** True if this.type is 'output'. */
    isOutput() { return (this.type == 'output'); }

    /** True if this is the root node in the model */
    isRoot() { return (this.type == 'root'); }

    /** True if this is an output and it's not implicit */
    isExplicitOutput() { return (this.isOutput() && !this.implicit); }

    /** True if this is an output and it is implicit */
    isImplicitOutput() { return (this.isOutput() && this.implicit); }

    /** True if this.type is 'input', 'unconnected_input', or 'output'. */
    isInputOrOutput() { return this.type.match(inputOrOutputRegex); }

    /** True is this.type is 'subsystem' */
    isSubsystem() { return (this.type == 'subsystem'); }

    /** True if it's a subsystem and this.subsystem_type is 'group' */
    isGroup() { return (this.isSubsystem() && this.subsystem_type == 'group'); }

    /** True if it's a subsystem and this.subsystem_type is 'component' */
    isComponent() { return (this.isSubsystem() && this.subsystem_type == 'component'); }

    /** True if this is a "fake" node that manages filtered children. */
    isFilter() { return (this.type == 'filter'); }

    /** Not connectable if this is an input group or parents are minimized. */
    isConnectable() {
        if (this.isInputOrOutput() && !(this.hasChildren() ||
            this.parent.draw.minimized || this.parentComponent.draw.minimized)) return true;

        return this.draw.minimized;
    }

    /** Return false if the node is hidden, or filtered */
    isVisible() {
        return !(this.draw.hidden || this.draw.filtered);
    }

    /** True if node is not hidden and has no visible children */
    isVisibleLeaf() {
        if (this.draw.hidden) return false; // Any explicitly hidden node
        if (this.isInputOrOutput()) return !this.draw.filtered; // Variable
        if (!this.hasChildren()) return true; // Group or component w/out children
        return this.draw.minimized; // Collapsed non-variable
    }

    isFilteredVariable() {
        return (this.isInputOrOutput() && this.draw.filtered);
    }

    /**
     * Look for the supplied node in the set of child names.
     * @returns {Boolean} True if a match is found, otherwise false.
     */
    hasNodeInChildren(compareNode) {
        return this.childNames.has(compareNode.absPathName);
    }

    /** Look for the supplied node in the parentage of this one.
     * @param {N2TreeNode} compareNode The node to look for.
     * @param {N2TreeNode} [parentLimit = null] Stop searching at this common parent.
     * @returns {Boolean} True if the node is found, otherwise false.
     */
    hasParent(compareNode, parentLimit = null) {
        for (let obj = this.parent; obj != null && obj !== parentLimit; obj = obj.parent) {
            if (obj === compareNode) {
                return true;
            }
        }

        return false;
    }

    /**
     * Look for the supplied node in the lineage of this one.
     * @param {N2TreeNode} compareNode The node to look for.
     * @param {N2TreeNode} [parentLimit = null] Stop searching at this common parent.
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
     * Add ourselves to the supplied array if we contain a cycleArrows property.
     * @param {Array} arr The array to add to.
     */
    _getNodesInChildrenWithCycleArrows(arr) {
        if (this.cycleArrows) {
            arr.push(this);
        }

        if (this.hasChildren()) {
            for (let child of this.children) {
                child._getNodesInChildrenWithCycleArrows(arr);
            }
        }
    }

    /**
     * Populate an array with nodes in our lineage that contain a cycleArrows member.
     * @returns {Array} The array containing all the found nodes with cycleArrows.
     */
    getNodesWithCycleArrows() {
        let arr = [];

        // Check parents first.
        for (let obj = this.parent; obj != null; obj = obj.parent) {
            if (obj.cycleArrows) {
                arr.push(obj);
            }
        }

        // Check all descendants as well.
        this._getNodesInChildrenWithCycleArrows(arr);

        return arr;
    }

    /**
     * Find the closest parent shared by two nodes; farthest should be tree root.
     * @param {N2TreeNode} other Another node to compare parents with.
     * @returns {N2TreeNode} The first common parent found.
     */
    nearestCommonParent(other) {
        for (let myParent = this.parent; myParent != null; myParent = myParent.parent)
            for (let otherParent = other.parent; otherParent != null; otherParent = otherParent.parent)
                if (myParent === otherParent) return myParent;

        // Should never get here because root is parent of all
        debugInfo("No common parent found between two nodes: ", this, other);
        return null;
    }

    /**
     * If the node has a lot of descendants and it wasn't manually expanded,
     * minimize it.
     * @param {Number} depthCount The number of nodes at the next depth down.
     * @returns {Boolean} True if minimized here, false otherwise.
     */
    minimizeIfLarge(depthCount) {
        if (!(this.isRoot() || this.draw.manuallyExpanded) &&
            (this.depth >= (this.isComponent() ?
                Precollapse.cmpDepthStart : Precollapse.grpDepthStart) &&
                this.numDescendants > Precollapse.threshold &&
                this.children.length > Precollapse.children - this.depth &&
                depthCount > Precollapse.depthLimit)) {
            debugInfo(`Precollapsing node ${this.absPathName}`)
            this.minimize();
            return true;
        }

        return false;
    }

    /**
     * Convert an absolute path name to a string that's safe to use as an HTML id.
     * @param {String} absPathName The name to convert.
     * @returns {String} The HTML-safe id.
     */
    static absPathToId(absPathName) {
        return absPathName.replace(/[\.<> :]/g, function (c) {
            return {
                ' ': '__',
                '<': '_LT',
                '>': '_GT',
                '.': '_',
                ':': '-'
            }[c];
        })
    }

    toId() { return N2TreeNode.absPathToId(this.absPathName); }

    _insertAsLastInput(newChild) {
        if (!this.hasChildren()) return;

        let idx = -1;
        for (idx = 0; idx < this.children.length - 1; idx++ ) {
            if (this.children[idx+1].isOutput()) break;
        }

        this.children.splice(idx + 1, 0, newChild);
    }

    addFilterChildIfComponent() {
        if (this.isComponent() && this.hasChildren()) {

            // Separate N2FilterNodes are added for inputs and outputs so
            // they can be inserted at the correct place in the diagram.
            this.filter = {
                inputs: new N2FilterNode(this, 'inputs'),
                outputs: new N2FilterNode(this, 'outputs')
            };
            this._insertAsLastInput(this.filter.inputs);
            this.children.push(this.filter.outputs);
        }
    }

    addSelfToFilter() {
        if (this.isInput()) { this.parent.filter.inputs.add(this); }
        else if (this.isOutput()) { this.parent.filter.outputs.add(this); }
    }

    removeSelfFromFilter() {
        if (this.isInput()) { this.parent.filter.inputs.del(this); }
        else if (this.isOutput()) { this.parent.filter.outputs.del(this); }
    }

    isFilter() { return false; }
    hasFilters() { return ('filter' in this); }
    isInputFilter() { return false; }
    isOutputFilter() { return false; }
}

class N2FilterNode extends N2TreeNode {
    constructor(parentComponent, suffix) {
        super(
            {
                name: `${parentComponent.name}_N2_FILTER_${suffix}`,
                type: 'filter'
            }
        )

        this.parentComponent = parentComponent;
        this.hide();
        this.minimize();
        this.suffix = suffix;
    }

    add(node) {
        if (!this.hasChildren()) { this.children = []; }
        this.children.push(node);
        this.childNames.add(node.absPathName);
        this.numDescendants += 1;
        node.doFilter(this);
        this.show();
    }
    
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

    wipe() {
        if (this.hasChildren()) {
            for (const child of this.children) { child.undoFilter(); }
            this.children = null;

            this.numDescendants = 0;
            this.childNames.clear();
            this.childNames.add(this.absPathName);
            this.hide();
        }
    }

    hasChild(child) {
        return (this.hasChildren()? this.children.indexOf(child) >= 0 : false);
    }

    get count() { return (this.hasChildren()? this.children.length : 0); }
    
    /** Don't expand, always stay minimized. */
    expand() { return this; }

    /** Only show if there are filtered nodes stored as children */
    show() {
        if (this.hasChildren()) { super.show(); }
        return this;
    }

    isFilter() { return true; }
    hasFilters() { return false; }
    isInputFilter() { return this.suffix == 'inputs'; }
    isOutputFilter() { return this.suffix == 'outputs'; }

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