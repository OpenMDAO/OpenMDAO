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
    static showLinearSolverNames = true;

    /**
     * Switch back and forth between showing the linear or non-linear solver names.
     */
    static toggleSolverNameType() {
        this.showLinearSolverNames = !this.showLinearSolverNames;
    }

    constructor(origNode, attribNames, parent) {
        super(origNode, attribNames, parent);

        // Solver names may be empty, so set them to "None" instead.
        if (this.linear_solver == "") this.linear_solver = "None";
        if (this.nonlinear_solver == "") this.nonlinear_solver = "None";
        if (exists(this.parentComponent)) this.parentComponent = parent;
    }

    get absPathName() { return this.path; }
    set absPathName(newName) { this.path = newName; return newName; }

    /**
     * Perform special checking for filter and auto-IVC nodes when determining
     * the node's name.
     * @returns {String} The label for the node.
     */
    getTextName() {
        let retVal = this.name;

        if (this.name == '_auto_ivc') {
            retVal = 'Auto-IVC';
        }
        else if (this.isFilter()) {
            retVal = super.getTextName();
        }
        else if (this.path.match(/^_auto_ivc.*/) && this.promotedName !== undefined) {
            retVal = this.promotedName;
        }
        return retVal;
    }

    /**
     * Return the name of the linear or non-linear solver depending
     * on the value of OmTreeNode.showLinearSolverNames.
     * @returns {String} The label for the solver node.
     */
    getSolverText() {
        const solverName = OmTreeNode.showLinearSolverNames? this.linear_solver : this.nonlinear_solver;
        let suffix = '';

        if (!OmTreeNode.showLinearSolverNames && 'solve_subsystems' in this && this.solve_subsystems) {
            suffix = ' (sub_solve)';
        }

        return `${solverName}${suffix}`;
    }

    /** Create and return a new  OmNodeDisplayData object. */
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

    /** True if it's a subsystem and this.subsystem_type is 'group'. Overrides base class */
    isGroup() { return (this.isSubsystem() && this.subsystem_type == 'group'); }

    /** True if it's a subsystem and this.subsystem_type is 'component' */
    isComponent() { return (this.isSubsystem() && this.subsystem_type == 'component'); }

    /** Not connectable if this is an input group or parents are minimized. */
    isConnectable() {
        if (this.isInputOrOutput() && !(this.hasChildren() ||
            this.parent.draw.minimized || this.parentComponent.draw.minimized)) return true;

        return this.draw.minimized;
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
            for (const child of this.children) {
                if (child instanceof OmTreeNode) {
                    child._getNodesInChildrenWithCycleArrows(arr);
                }
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

    _preCollapseDepth() { return this.isComponent()? Precollapse.cmpDepthStart : Precollapse.grpDepthStart; }

}
