/** Process the tree, connections, and other info provided about the model. */
class ModelData {

    /** Do some discovery in the tree and rearrange & enhance where necessary. */
    constructor(modelJSON) {
        this.root = modelJSON.tree;
        this.tree = modelJSON.tree;
        this.root.name = 'model'; // Change 'root' to 'model'
        this.sys_pathnames_list = modelJSON.sys_pathnames_list;
        this.conns = modelJSON.connections_list;
        this.abs2prom = modelJSON.abs2prom; // May be undefined.
        this.maxDepth = 1;
        this.idCounter = 0;

        this.changeBlankSolverNamesToNone(this.root);
        this.expandColonVars(this.root);
        this.flattenColonGroups(this.root);
        this.setParentsAndDepth(this.root, null, 1);
        this.identifyUnconnectedParams();
        this.initSubSystemChildren(this.root);
    }

    /** Solver names may be empty, so set them to "None" instead.
     * Recurses over children.
     * @param {Object} element The item with solver names to check.
     */
    changeBlankSolverNamesToNone(element) {
        if (element.linear_solver === "") element.linear_solver = "None";
        if (element.nonlinear_solver === "") element.nonlinear_solver = "None";
        if (element.children) {
            for (var i = 0; i < element.children.length; ++i) {
                this.changeBlankSolverNamesToNone(element.children[i]);
            }
        }
    }

    /**  */
    addChildren(originalParent, parent, arrayOfNames, arrayOfNamesIndex, type) {
        console.log("addChildren called");
        if (arrayOfNames.length == arrayOfNamesIndex) return;

        let name = arrayOfNames[arrayOfNamesIndex];

        if (!parent.hasOwnProperty("children")) {
            parent.children = [];
        }

        let parentIdx = indexForMember(parent.children, 'name', name);
        if (parentIdx == -1) { //new name not found in parent, create new
            let newChild = {
                "name": name,
                "type": type,
                "splitByColon": true,
                "originalParent": originalParent
            };

            // Was originally && instead of ||, which wouldn't ever work?
            if (type.match(/^param$|^unconnected_param$/)) {
                parent.children.splice(0, 0, newChild);
            }
            else {
                parent.children.push(newChild);
            }
            this.addChildren(originalParent, newChild, arrayOfNames, arrayOfNamesIndex + 1, type);
        }
        else { // new name already found in parent, keep traversing
            this.addChildren(originalParent, parent.children[parentIdx], arrayOfNames, arrayOfNamesIndex + 1, type);
        }
    }

    /** If an object has a child with colons in its name, split those the child
     * into multiple objects named from the tokens in the original name. Replace the
     * original child object with the new children. Recurse over the array of children.
     * @param {Object} element The object that may have children to check.
     */
    expandColonVars(element) {
        if (!Array.isArray(element.children)) return;

        for (let i = 0; i < element.children.length; ++i) {

            let splitArray = element.children[i].name.split(":");
            if (splitArray.length > 1) {
                if (!element.hasOwnProperty("subsystem_type") ||
                    element.subsystem_type !== "component") {
                    console.error("There is a colon-named object whose parent is not a component.");
                    return;
                }
                let type = element.children[i].type;
                element.children.splice(i--, 1);
                this.addChildren(element, element, splitArray, 0, type);
            }
        }

        for (var i = 0; i < element.children.length; ++i) {
            this.expandColonVars(element.children[i]);
        }
    }

    /** If an element formerly had a name with colons, but was split by expandColonVars()
     * and only ended up with one child, recombine the element and its child. Operate
     * recursively on all children.
     * @param {Object} element The object to check.
     */
    flattenColonGroups(element) {
        if (!Array.isArray(element.children)) return;

        while (element.splitByColon && exists(element.children) && element.children.length == 1 &&
            element.children[0].splitByColon) {
            let child = element.children[0];
            element.name += ":" + chilelement.name;
            element.children = (chilelement.hasOwnProperty("children") && chilelement.children.length >= 1) ?
                chilelement.children : null; //absorb childs children
            if (element.children == null) delete element.children;
        }

        if (!Array.isArray(element.children)) return;

        for (var i = 0; i < element.children.length; ++i) {
            this.flattenColonGroups(element.children[i]);
        }
    }

    /** Sets parents and depth of all nodes, and determine max depth. Flags the
     * element as implicit if any children are implicit.
     * @param {Object} element Item to process.
     * @param {Object} parent Parent of element, null for root node.
     * @param {number} depth Numerical level of ancestry.
     * @return True is element is implicit, false otherwise.
     */
    setParentsAndDepth(element, parent, depth) { // Formerly InitTree()
        element.numLeaves = 0; // for nested params
        element.depth = depth;
        element.parent = parent;
        element.id = ++this.idCounter; // id starts at 1 for if comparision
        element.absPathName = "";

        if (element.parent) { // not root node? element.parent.absPathName : "";
            if (element.parent.absPathName !== "") {
                element.absPathName += element.parent.absPathName;
                element.absPathName += (element.parent.splitByColon) ? ":" : ".";
            }
            element.absPathName += element.name;
        }

        if (element.type.match(/^unknown$|^param$|^unconnected_param$/)) {
            let parentComponent = (element.originalParent) ? element.originalParent : element.parent;
            if (parentComponent.type === "subsystem" &&
                parentComponent.subsystem_type === "component") {
                element.parentComponent = parentComponent;
            }
            else {
                console.error("Param or unknown without a parent component!");
            }
        }

        if (element.splitByColon) {
            element.colonName = element.name;
            for (let obj = element.parent; obj.splitByColon; obj = obj.parent) {
                element.colonName = obj.name + ":" + element.colonName;
            }
        }

        this.maxDepth = Math.max(depth, this.maxDepth);

        if (element.type === "subsystem") {
            this.maxSystemDepth = Math.max(depth, this.maxSystemDepth);
        }

        if (element.children) {
            for (var i = 0; i < element.children.length; ++i) {
                var implicit = this.setParentsAndDepth(element.children[i], element, depth + 1);
                if (implicit) {
                    element.implicit = true;
                }
            }
        }

        return (element.implicit) ? true : false;
    }


    /** Check the entire array of model connections for any with a target matching
     * the specified path.
     * @param {string} elementPath The full path of the element to check.
     * @return True if the path is found as a target in the connection list.
     */
    hasInputConnection(elementPath) {
        for (let conn of this.conns) {
            if (conn.tgt.match(elementPath)) { return true; }
        }

        return false;
    }

    /** Check the entire array of model connections for any with a source matching
     * the specified path.
     * @param {string} elementPath The full path of the element to check.
     * @return True if the path is found as a source in the connection list.
     */
    hasOutputConnection(elementPath) {
        for (let conn of this.conns) {
            if (conn.src.match(elementPath)) { return true; }
        }

        return false;
    }

    /** If an element has no connection naming it as a source or target,
     * relabel it as unconnected.
     */
    identifyUnconnectedParams() { // Formerly updateRootTypes
        let stack = [];
        this.root.children.forEach(function (child) { stack.push(child); })

        while (stack.length > 0) {
            let element = stack.pop();
            if (element.type === "param") {
                if (!this.hasInputConnection(element.absPathName) &&
                    !this.hasOutputConnection(element.absPathName)) {
                    element.type = "unconnected_param";
                }
            }

            if (Array.isArray(element.children)) {
                element.children.forEach(function (child) {
                    stack.push(child);
                });
            }
        }
    }

    /** Create an array in each element containing references to its
    * children that are subsystems. Runs recursively over the element's
    * children array.
    * @param {Object} element Element with children to check.
    */
    initSubSystemChildren(element) {
        let self = this; // To permit the forEach callback below.

        if (!Array.isArray(element.children)) { return; }

        element.children.forEach(function (child) {
            if (child.type === 'subsystem') {
                if (!Array.isArray(element.subsystem_children)) {
                    element.subsystem_children = [];
                }

                element.subsystem_children.push(child);
                self.initSubSystemChildren(child);
            }
        })
    }
}