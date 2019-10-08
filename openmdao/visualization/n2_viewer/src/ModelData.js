/** Process the tree, connections, and other info provided about the model. */
class ModelData {

    /** Do some discovery in the tree and rearrange & enhance where necessary. */
    constructor(modelJSON) {
        this.root = this.tree = modelJSON.tree;
        this.root.name = 'model'; // Change 'root' to 'model'
        this.sysPathnamesList = modelJSON.sys_pathnames_list;
        this.conns = modelJSON.connections_list;

        let startTime = 0;

        // console.log('conns: ', this.conns);
        this.abs2prom = modelJSON.abs2prom; // May be undefined.
        this.declarePartialsList = modelJSON.declare_partials_list;
        this.maxDepth = 1;
        this.idCounter = 0;

        startTime = Date.now();
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

        startTime = Date.now();
        this.computeConnections();
        console.log("ModelData.computeConnections: ", Date.now() - startTime, "ms");
    }

    /**
     * Recursively perform independant actions on the entire model tree
     * to prevent having to traverse it multiple times.
     * @param {Object} element The current element being updated.
     */
    performHouseKeeping(element) {
        this.changeBlankSolverNamesToNone(element);
        this.identifyUnconnectedParams(element);

        // From old ClearConnections():
        element.targetsParamView = new Set();
        element.targetsHideParams = new Set();

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

    /**
     * Traverse the tree trying to match a complete path to the element.
     * @param {Object} element Starting point.
     * @param {string[]} nameArray Path to element broken into components as array elements.
     * @param {number} nameIndex Index of nameArray currently being searched for.
     * @return {Object} Reference to element in the path.
     */
    getObjectInTree(element, nameArray, nameIndex) {
        // Reached the last name:
        if (nameArray.length == nameIndex) return element;

        // No children:
        if (!Array.isPopulatedArray(element.children)) return null;

        for (let child of element.children) {
            if (child.name == nameArray[nameIndex]) {
                return this.getObjectInTree(child, nameArray, nameIndex + 1);
            }
            else {
                let numNames = child.name.split(":").length;
                if (numNames >= 2 && nameIndex + numNames <= nameArray.length) {
                    let mergedName = nameArray[nameIndex];
                    for (let j = 1; j < numNames; ++j) {
                        mergedName += ":" + nameArray[nameIndex + j];
                    }
                    if (child.name == mergedName) {
                        return this.getObjectInTree(child, nameArray, nameIndex + numNames);
                    }
                }
            }
        }

        return null;
    }

    /** 
     * Add all leaf descendents of specified element to the array.
     * @param {Object} element Current element to work on.
     * @param {Object[]} objArray Array to add to.
     */
    addLeaves(element, objArray) {
        if (!element.type.match(paramRegex)) {
            objArray.push(element);
        }

        if (Array.isPopulatedArray(element.children)) {
            for (let child of element.children) {
                this.addLeaves(child, objArray);
            }
        }
    }

    /**
     * Iterate over the connections list, and find the two objects that
     * make up each connection.
     */
    computeConnections() {
        let sysPathnames = this.sysPathnamesList;

        for (let conn of this.conns) {
            // Process sources
            let srcSplitArray = conn.src.split(/\.|:/);
            let srcObj = this.getObjectInTree(this.root, srcSplitArray, 0);

            if (srcObj == null)
                throw ("Cannot find connection source " + conn.src);

            let srcObjArray = [srcObj];
            if (srcObj.type !== "unknown") // source obj must be unknown
                throw ("There is a source that is not an unknown.");

            if (Array.isPopulatedArray(srcObj.children))
                throw ("There is a source that has children.");

            for (let obj = srcObj.parent; obj != null; obj = obj.parent) {
                srcObjArray.push(obj);
            }

            // Process targets
            let tgtSplitArray = conn.tgt.split(/\.|:/);
            let tgtObj = this.getObjectInTree(this.root, tgtSplitArray, 0);

            if (tgtObj == null)
                throw ("Cannot find connection target " + conn.tgt);

            let tgtObjArrayParamView = [tgtObj];
            let tgtObjArrayHideParams = [tgtObj];

            // Target obj must be a param
            if (!tgtObj.type.match(paramRegex))
                throw ("There is a target that is NOT a param.");

            if (Array.isPopulatedArray(tgtObj.children))
                throw ("There is a target that has children.");

            if (! tgtObj.parentComponent)
                throw ("Target object " + conn.tgt + " has missing parentComponent.");

            this.addLeaves(tgtObj.parentComponent, tgtObjArrayHideParams); //contaminate
            for (let obj = tgtObj.parent; obj != null; obj = obj.parent) {
                tgtObjArrayParamView.push(obj);
                tgtObjArrayHideParams.push(obj);
            }

            for (let srcObj of srcObjArray) {
                if (!srcObj.hasOwnProperty('targetsParamView'))
                    srcObj.targetsParamView = new Set();
                if (!srcObj.hasOwnProperty('targetsHideParams'))
                    srcObj.targetsHideParams = new Set();

                tgtObjArrayParamView.forEach(item => srcObj.targetsParamView.add(item));
                tgtObjArrayHideParams.forEach(item => srcObj.targetsHideParams.add(item));
            }

            let cycleArrowsArray = [];
            if (Array.isPopulatedArray(conn.cycle_arrows)) {
                let cycleArrows = conn.cycle_arrows;
                for (let cycleArrow of cycleArrows) {
                    if (cycleArrow.length != 2)
                        throw ("cycleArrowsSplitArray length not 2, got " +
                            cycleArrow.length + ": " + cycleArrow);

                    let srcPathname = sysPathnames[cycleArrow[0]];
                    let tgtPathname = sysPathnames[cycleArrow[1]];

                    let splitArray = srcPathname.split(/\.|:/);
                    let arrowBeginObj = this.getObjectInTree(this.root, splitArray, 0);
                    if (arrowBeginObj == null)
                        throw ("Cannot find cycle arrows begin object " + srcPathname);

                    splitArray = tgtPathname.split(/\.|:/);
                    let arrowEndObj = this.getObjectInTree(this.root, splitArray, 0);
                    if (arrowEndObj == null)
                        throw ("Cannot find cycle arrows end object " + tgtPathname);

                    cycleArrowsArray.push({ "begin": arrowBeginObj, "end": arrowEndObj });
                }
            }

            if (cycleArrowsArray.length > 0) {
                if (!tgtObj.parent.hasOwnProperty("cycleArrows")) {
                    tgtObj.parent.cycleArrows = [];
                }
                tgtObj.parent.cycleArrows.push({ "src": srcObj, "arrows": cycleArrowsArray });
            }
        }
    }
}