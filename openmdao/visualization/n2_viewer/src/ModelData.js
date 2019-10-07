/** Process the tree, connections, and other info provided about the model. */
class ModelData {

    /** Do some discovery in the tree and rearrange & enhance where necessary. */
    constructor(modelJSON) {
        this.root = this.tree = modelJSON.tree;
        this.root.name = 'model'; // Change 'root' to 'model'
        this.sys_pathnames_list = modelJSON.sys_pathnames_list;
        this.conns = modelJSON.connections_list;
        // console.log('conns: ', this.conns);
        this.abs2prom = modelJSON.abs2prom; // May be undefined.
        this.declarePartialsList = modelJSON.declare_partials_list;
        this.maxDepth = 1;
        this.idCounter = 0;

        let startTime = Date.now();
        this.expandColonVars(this.root);
        console.log("ModelData.expandColonVars: ", Date.now() - startTime, "ms");

        startTime = Date.now();
        this.flattenColonGroups(this.root);
        console.log("ModelData.flattenColonGroups: ", Date.now() - startTime, "ms");

        startTime = Date.now();
        this.setParentsAndDepth(this.root, null, 1);
        console.log("ModelData.setParentsAndDepth: ", Date.now() - startTime, "ms");

        startTime = Date.now();
        this.performHouseKeeping(this.root);
        console.log("ModelData.performHouseKeeping: ", Date.now() - startTime, "ms");

        startTime = Date.now();
        this.initSubSystemChildren(this.root);
        console.log("ModelData.initSubSystemChildren: ", Date.now() - startTime, "ms");
    }

    /**
     * Recursively perform independant actions on the entire model tree
     * to prevent having to traverse it multiple times.
     * @param {Object} element The current element being updated.
     */
    performHouseKeeping(element) {
        this.changeBlankSolverNamesToNone(element);
        this.identifyUnconnectedParams(element);

        if (Array.isPopulatedArray(element.children)) {
            for (let i = 0; i < element.children.length; ++i) {
                this.performHouseKeeping(element.children[i]);
            }
        }
    }

    /**
     * Solver names may be empty, so set them to "None" instead.
     * @param {Object} element The item with solver names to check.
     */
    changeBlankSolverNamesToNone(element) {
        if (element.linear_solver == "") element.linear_solver = "None";
        if (element.nonlinear_solver == "") element.nonlinear_solver = "None";
    }

    /** Called by expandColonVars when splitting an element into children.
     * TODO: Document params and recursive functionality.
     */
    addChildren(originalParent, parent, arrayOfNames, arrayOfNamesIndex, type) {
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
            if (type.match(paramRegex)) {
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

    /**
     * If an object has a child with colons in its name, split those the child
     * into multiple objects named from the tokens in the original name. Replace the
     * original child object with the new children. Recurse over the array of children.
     * @param {Object} element The object that may have children to check.
     */
    expandColonVars(element) {
        if (!Array.isPopulatedArray(element.children)) return;

        for (let i = 0; i < element.children.length; ++i) {

            let splitArray = element.children[i].name.split(":");
            if (splitArray.length > 1) {
                if (!element.hasOwnProperty("subsystem_type") ||
                    element.subsystem_type != "component") {
                    throw ("There is a colon-named object whose parent is not a component.");
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

    /**
     * If an element formerly had a name with colons, but was split by expandColonVars()
     * and only ended up with one child, recombine the element and its child. Operate
     * recursively on all children.
     * @param {Object} element The object to check.
     */
    flattenColonGroups(element) {
        if (!Array.isPopulatedArray(element.children)) return;

        while (element.splitByColon && exists(element.children) &&
            element.children.length == 1 &&
            element.children[0].splitByColon) {
            let child = element.children[0];
            element.name += ":" + child.name;
            element.children = (Array.isArray(child.children) &&
                child.children.length >= 1) ?
                child.children : null; //absorb childs children
            if (element.children == null) delete element.children;
        }

        if (!Array.isArray(element.children)) return;

        for (var i = 0; i < element.children.length; ++i) {
            this.flattenColonGroups(element.children[i]);
        }
    }

    /**
     * Sets parents and depth of all nodes, and determine max depth. Flags the
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
            if (element.parent.absPathName != "") {
                element.absPathName += element.parent.absPathName;
                element.absPathName += (element.parent.splitByColon) ? ":" : ".";
            }
            element.absPathName += element.name;
        }

        if (element.type.match(paramOrUnknownRegex)) {
            let parentComponent = (element.originalParent) ? element.originalParent : element.parent;
            if (parentComponent.type == "subsystem" &&
                parentComponent.subsystem_type == "component") {
                element.parentComponent = parentComponent;
            }
            else {
                throw ("Param or unknown without a parent component!");
            }
        }

        if (element.splitByColon) {
            element.colonName = element.name;
            for (let obj = element.parent; obj.splitByColon; obj = obj.parent) {
                element.colonName = obj.name + ":" + element.colonName;
            }
        }

        this.maxDepth = Math.max(depth, this.maxDepth);

        if (element.type == "subsystem") {
            this.maxSystemDepth = Math.max(depth, this.maxSystemDepth);
        }

        if (Array.isPopulatedArray(element.children)) {
            for (let i = 0; i < element.children.length; ++i) {
                let implicit = this.setParentsAndDepth(element.children[i], element, depth + 1);
                if (implicit) {
                    element.implicit = true;
                }
            }
        }

        return (element.implicit) ? true : false;
    }

    /**
     * Check the entire array of model connections for any with a target matching
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

    /**
     * Check the entire array of model connections for any with a source matching
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

    /**
     * Check the entire array of model connections for any with a source OR
     * target matching the specified path.
     * @param {string} elementPath The full path of the element to check.
     * @return True if the path is found as a source in the connection list.
     */
    hasAnyConnection(elementPath) {
        for (let conn of this.conns) {
            if (conn.src == elementPath || conn.tgt == elementPath)
                return true;
        }

        return false;
    }

    /**
     * If an element has no connection naming it as a source or target,
     * relabel it as unconnected.
     */
    identifyUnconnectedParams(element) { // Formerly updateRootTypes
        if (element.type == "param" && !this.hasAnyConnection(element.absPathName))
            element.type = "unconnected_param";
    }

    /**
    * Create an array in each element containing references to its
    * children that are subsystems. Runs recursively over the element's
    * children array.
    * @param {Object} element Element with children to check.
    */
    initSubSystemChildren(element) {
        let self = this; // To permit the forEach callback below.

        if (!Array.isArray(element.children)) { return; }

        element.children.forEach(function (child) {
            if (child.type == 'subsystem') {
                if (!Array.isArray(element.subsystem_children)) {
                    element.subsystem_children = [];
                }

                element.subsystem_children.push(child);
                self.initSubSystemChildren(child);
            }
        })
    }

    /**
     * Build a string from the absoluate path names of the two elements and
     * try to find it in the declare partials list.
     * @param {Object} srcObj The source element.
     * @param {Object} tgtObj The target element.
     * @return {Boolean} True if the string was found.
     */
    isDeclaredPartial(srcObj, tgtObj) {
        let partialsString = tgtObj.absPathName + " > " + srcObj.absPathName;

        if (this.declarePartialsList.includes(partialsString)) return true;

        return false;
    }

}