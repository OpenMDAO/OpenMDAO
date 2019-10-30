/** Process the tree, connections, and other info provided about the model. */
class ModelData {

    /** Do some discovery in the tree and rearrange & enhance where necessary. */
    constructor(modelJSON) {
        modelJSON.tree.name = 'model'; // Change 'root' to 'model'
        this.conns = modelJSON.connections_list;
        this.abs2prom = modelJSON.abs2prom; // May be undefined.
        this.declarePartialsList = modelJSON.declare_partials_list;
        this.sysPathnamesList = modelJSON.sys_pathnames_list;

        this.maxDepth = 1;
        this.idCounter = 0;
        this.unconnectedParams = 0;

        console.time('ModelData._convertToN2TreeNodes');
        this.root = this.tree = modelJSON.tree = this._convertToN2TreeNodes(modelJSON.tree);
        console.timeEnd('ModelData._convertToN2TreeNodes');

        console.time('ModelData._expandColonVars');
        this._expandColonVars(this.root);
        console.timeEnd('ModelData._expandColonVars');

        console.time('ModelData._flattenColonGroups');
        this._flattenColonGroups(this.root);
        console.timeEnd('ModelData._flattenColonGroups');

        console.time('ModelData._setParentsAndDepth');
        this._setParentsAndDepth(this.root, null, 1);
        console.timeEnd('ModelData._setParentsAndDepth');

        if (this.unconnectedParams > 0)
            console.info("Unconnected nodes: ", this.unconnectedParams);

        console.time('ModelData._initSubSystemChildren');
        this._initSubSystemChildren(this.root);
        console.timeEnd('ModelData._initSubSystemChildren');

        console.time('ModelData._computeConnections');
        this._computeConnections();
        console.timeEnd('ModelData._computeConnections');

        // console.log("New model: ", modelJSON);
        // this.errorCheck();
    }

    /**
     * For debugging: Make sure every tree member is an N2TreeNode.
     * @param {N2TreeNode} [node = this.root] The node to start with.
     */
    errorCheck(node = this.root) {
        if (!(node instanceof N2TreeNode))
            console.log('Node with problem: ', node);

        for (let prop of ['parent', 'originalParent', 'parentComponent']) {
            if (node[prop] && !(node[prop] instanceof N2TreeNode))
                console.log('Node with problem ' + prop + ': ', node);
        }

        if (node.hasChildren()) {
            for (let child of node.children) {
                this.errorCheck(child);
            }
        }
    }

    /**
     * Recurse over the tree and replace the JSON objects 
     * provided by n2_viewer.py with N2TreeNodes.
     * @param {Object} element The current element being updated.
     */
    _convertToN2TreeNodes(element) {
        let newNode = new N2TreeNode(element);

        if (newNode.hasChildren()) {
            for (let i = 0; i < newNode.children.length; ++i) {
                newNode.children[i] = this._convertToN2TreeNodes(newNode.children[i]);
                newNode.children[i].parent = newNode;
                if (exists(newNode.children[i].parentComponent)) newNode.children[i].parentComponent = newNode;
            }
        }

        return newNode;
    }

    /** Called by _expandColonVars when splitting an element into children.
     * TODO: Document params and recursive functionality.
     */
    _addChildren(originalParent, parent, arrayOfNames, arrayOfNamesIndex, type) {
        if (arrayOfNames.length == arrayOfNamesIndex) return;

        let name = arrayOfNames[arrayOfNamesIndex];

        if (!parent.hasOwnProperty("children")) {
            parent.children = [];
        }

        let parentIdx = indexForMember(parent.children, 'name', name);
        if (parentIdx == -1) { //new name not found in parent, create new
            let newChild = new N2TreeNode({
                "name": name,
                "type": type,
                "splitByColon": true,
                "originalParent": originalParent
            });

            // Was originally && instead of ||, which wouldn't ever work?
            if (type.match(paramRegex)) {
                parent.children.splice(0, 0, newChild);
            }
            else {
                parent.children.push(newChild);
            }
            this._addChildren(originalParent, newChild, arrayOfNames,
                arrayOfNamesIndex + 1, type);
        }
        else { // new name already found in parent, keep traversing
            this._addChildren(originalParent, parent.children[parentIdx],
                arrayOfNames, arrayOfNamesIndex + 1, type);
        }
    }

    /**
     * If an object has a child with colons in its name, split that child
     * into multiple objects named from the tokens in the original name. Replace the
     * original child object with the new children. Recurse over the array of children.
     * @param {N2TreeNode} node The object that may have children to check.
     */
    _expandColonVars(node) {
        if (!node.hasChildren()) return;

        // Don't use an iterator here because we may modify the array
        for (let i = 0; i < node.children.length; ++i) {

            let splitArray = node.children[i].name.split(":");
            if (splitArray.length > 1) {
                if (!node.hasOwnProperty("subsystem_type") ||
                    node.subsystem_type != "component") {
                    throw ("There is a colon-named object whose parent is not a component.");
                }
                let type = node.children[i].type;
                node.children.splice(i--, 1);
                this._addChildren(node, node, splitArray, 0, type);
            }
        }

        for (let child of node.children) {
            this._expandColonVars(child);
        }
    }

    /**
     * If an node formerly had a name with colons, but was split by _expandColonVars()
     * and only ended up with one child, recombine the node and its child. Operate
     * recursively on all children.
     * @param {N2TreeNode} node The object to check.
     */
    _flattenColonGroups(node) {
        if (!Array.isPopulatedArray(node.children)) return;

        while (node.splitByColon && exists(node.children) &&
            node.children.length == 1 &&
            node.children[0].splitByColon) {
            let child = node.children[0];
            node.name += ":" + child.name;
            node.children = (Array.isArray(child.children) &&
                child.children.length >= 1) ?
                child.children : null; //absorb childs children
            if (node.children == null) delete node.children;
        }

        if (!Array.isArray(node.children)) return;

        for (let child of node.children) {
            this._flattenColonGroups(child);
        }
    }

    /**
     * Sets parents and depth of all nodes, and determine max depth. Flags the
     * node as implicit if any children are implicit.
     * @param {N2TreeNode} node Item to process.
     * @param {N2TreeNode} parent Parent of node, null for root node.
     * @param {number} depth Numerical level of ancestry.
     * @return True is node is implicit, false otherwise.
     */
    _setParentsAndDepth(node, parent, depth) { // Formerly InitTree()
        node.numLeaves = 0; // for nested params
        node.depth = depth;
        node.parent = parent;
        node.id = ++this.idCounter; // id starts at 1 for if comparision
        node.absPathName = "";

        if (node.parent) { // not root node? node.parent.absPathName : "";
            if (node.parent.absPathName != "") {
                node.absPathName += node.parent.absPathName;
                node.absPathName += (node.parent.splitByColon) ? ":" : ".";
            }
            node.absPathName += node.name;
        }

        this.identifyUnconnectedParam(node);

        if (node.isParamOrUnknown()) {
            let parentComponent = (node.originalParent) ? node.originalParent : node.parent;
            if (parentComponent.type == "subsystem" &&
                parentComponent.subsystem_type == "component") {
                node.parentComponent = parentComponent;
            }
            else {
                throw ("Param or unknown without a parent component!");
            }
        }

        if (node.splitByColon) {
            node.colonName = node.name;
            for (let obj = node.parent; obj.splitByColon; obj = obj.parent) {
                node.colonName = obj.name + ":" + node.colonName;
            }
        }

        this.maxDepth = Math.max(depth, this.maxDepth);

        if (node.isSubsystem()) {
            this.maxSystemDepth = Math.max(depth, this.maxSystemDepth);
        }

        if (node.hasChildren()) {
            for (let child of node.children) {
                let implicit = this._setParentsAndDepth(child, node, depth + 1);
                if (implicit) node.implicit = true;
            }
        }

        return (node.implicit) ? true : false;
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

        this.unconnectedParams++;

        return false;
    }

    /**
    * Create an array in each node containing references to its
    * children that are subsystems. Runs recursively over the node's
    * children array.
    * @param {N2TreeNode} node Node with children to check.
    */
    _initSubSystemChildren(node) {
        if (!node.hasChildren()) { return; }

        for (let child of node.children) {
            if (child.isSubsystem()) {
                if (!node.hasChildren('subsystem_children'))
                    node.subsystem_children = [];

                node.subsystem_children.push(child);
                this._initSubSystemChildren(child);
            }
        }
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
     * Traverse the tree trying to match a complete path to the node.
     * @param {N2TreeNode} node Starting point.
     * @param {string[]} nameArray Path to node broken into components as array nodes.
     * @param {number} nameIndex Index of nameArray currently being searched for.
     * @return {N2TreeNode} Reference to node in the path.
     */
    _getObjectInTree(node, nameArray, nameIndex) {
        // Reached the last name:
        if (nameArray.length == nameIndex) return node;

        // No children:

        for (let child of node.children) {
            if (child.name == nameArray[nameIndex]) {
                return this._getObjectInTree(child, nameArray, nameIndex + 1);
            }
            else {
                let numNames = child.name.split(":").length;
                if (numNames >= 2 && nameIndex + numNames <= nameArray.length) {
                    let mergedName = nameArray[nameIndex];
                    for (let j = 1; j < numNames; ++j) {
                        mergedName += ":" + nameArray[nameIndex + j];
                    }
                    if (child.name == mergedName) {
                        return this._getObjectInTree(child, nameArray, nameIndex + numNames);
                    }
                }
            }
        }

        return null;
    }

    /** 
     * Add all leaf descendents of specified node to the array.
     * @param {N2TreeNode} node Current node to work on.
     * @param {N2TreeNode[]} objArray Array to add to.
     */
    _addLeaves(node, objArray) {
        if (!node.isParam()) {
            objArray.push(node);
        }

        if (node.hasChildren()) {
            for (let child of node.children) {
                this._addLeaves(child, objArray);
            }
        }
    }

    /**
     * Iterate over the connections list, and find the two objects that
     * make up each connection.
     */
    _computeConnections() {
        let sysPathnames = this.sysPathnamesList;

        for (let conn of this.conns) {
            // Process sources
            let srcSplitArray = conn.src.split(/\.|:/);
            let srcObj = this._getObjectInTree(this.root, srcSplitArray, 0);

            if (srcObj == null)
                throw ("Cannot find connection source " + conn.src);

            let srcObjArray = [srcObj];
            if (srcObj.type !== "unknown") // source obj must be unknown
                throw ("There is a source that is not an unknown.");

            if (srcObj.hasChildren()) throw ("There is a source that has children.");

            for (let obj = srcObj.parent; obj != null; obj = obj.parent) {
                srcObjArray.push(obj);
            }

            // Process targets
            let tgtSplitArray = conn.tgt.split(/\.|:/);
            let tgtObj = this._getObjectInTree(this.root, tgtSplitArray, 0);

            if (tgtObj == null) throw ("Cannot find connection target " + conn.tgt);

            let tgtObjArrayParamView = [tgtObj];
            let tgtObjArrayHideParams = [tgtObj];

            // Target obj must be a param
            if (!tgtObj.isParam()) throw ("There is a target that is NOT a param.");

            if (tgtObj.hasChildren()) throw ("There is a target that has children.");

            if (!tgtObj.parentComponent)
                throw ("Target object " + conn.tgt + " has missing parentComponent.");

            this._addLeaves(tgtObj.parentComponent, tgtObjArrayHideParams); //contaminate
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
                    let arrowBeginObj = this._getObjectInTree(this.root, splitArray, 0);
                    if (arrowBeginObj == null)
                        throw ("Cannot find cycle arrows begin object " + srcPathname);

                    splitArray = tgtPathname.split(/\.|:/);
                    let arrowEndObj = this._getObjectInTree(this.root, splitArray, 0);
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

    /**
     * If an element has no connection naming it as a source or target,
     * relabel it as unconnected.
     * @param {N2TreeNode} node The tree node to work on.
     */
    identifyUnconnectedParam(node) { // Formerly updateRootTypes
        if (!node.hasOwnProperty('absPathName'))
            throw ("identifyUnconnectedParam error: absPathName not set for ", node);

        if (node.isParam() && !this.hasAnyConnection(node.absPathName))
            node.type = "unconnected_param";
    }
}