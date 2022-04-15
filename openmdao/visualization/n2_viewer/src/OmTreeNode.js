// <<hpp_insert gen/TreeNode.js>>

/**
 * Extend NodeDisplayData by handling solvers.
 * @typedef NodeDisplayData
 */
class OmNodeDisplayData extends NodeDisplayData {
    constructor() {
        super();
        this.nameSolverWidthPx = 1; // Solver-side label width pixels as computed by Layout
        this.solverDims = new Dimensions({ x: 1e-6, y: 1e-6, width: 1, height: 1 });
        this.solverDims.preserve();
    }
}

/**
 * Extend FilterCapableTreeNode by adding support for feedback cycle arrows and solvers.
 * @typedef OmTreeNode
 */
class OmTreeNode extends FilterCapableNode {
    constructor(origNode, attribNames) {
        super(origNode, attribNames);

        // Solver names may be empty, so set them to "None" instead.
        if (this.linear_solver == "") this.linear_solver = "None";
        if (this.nonlinear_solver == "") this.nonlinear_solver = "None";
        if (this.isComponent()) this.draw.boxChildren = true;
    }

    get absPathName() { return this.path; }
    set absPathName(newName) { this.path = newName; return newName; }

    _newDisplayData() { return new OmNodeDisplayData(); }

    addFilterChild(attribNames) {
        if (this.isComponent()) { super.addFilterChild(attribNames); }
    }

    /** Override superclass method to include 'autoivc_input'. */
    isInput() { return this.type.match(/^(input|unconnected_input|autoivc_input)$/); }

    /** Override superclass method to include 'autoivc_input'. */
    isInputOrOutput() { return this.type.match(/^(output|input|unconnected_input|autoivc_input)$/); }

    /** True if this is an input whose source is an auto-ivc'd output */
    isAutoIvcInput() { return (this.type == 'autoivc_input'); }

    /** True if this is an output and it's not implicit */
    isExplicitOutput() { return (this.isOutput() && !this.implicit); }

    /** True if this is an output and it is implicit */
    isImplicitOutput() { return (this.isOutput() && this.implicit); }

    /** True is this.type is 'subsystem' */
    isSubsystem() { return (this.type == 'subsystem'); }

    /** True if it's a subsystem and this.subsystem_type is 'component' */
    isComponent() { return (this.isSubsystem() && this.subsystem_type == 'component'); }

    /**
     * Add ourselves to the supplied array if we contain a cycleArrows property.
     * @param {Array} arr The array to add to.
     */
     _getNodesInChildrenWithCycleArrows(arr) {
        if (this.cycleArrows) {
            arr.push(this);
        }

        if (this.hasChildren()) {
            for (const child of this.children) {
                if (child instanceof OmTreeNode)
                    child._getNodesInChildrenWithCycleArrows(arr);
            }
        }
    }

    /**
     * Populate an array with nodes in our lineage that contain a cycleArrows member.
     * @returns {Array} The array containing all the found nodes with cycleArrows.
     */
    getNodesWithCycleArrows() {
        const arr = [];

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
     * Create a backup of our position and other info.
     * @param {number} leafNum Identify this as the nth leaf of the tree
     */
     preserveSolverDims(leafNum) {
        this.draw.solverDims.preserve();

        if (this.rootIndex < 0) this.rootIndex = leafNum;
        this.prevRootIndex = this.rootIndex;
    }
}
