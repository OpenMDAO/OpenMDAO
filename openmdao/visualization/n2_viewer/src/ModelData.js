/** Process the tree, connections, and other info provided about the model. */
class ModelData {

    /** Do some discovery in the tree and rearrange & enhance where necessary. */
    constructor(modelJSON) {

        modelJSON.tree.name = 'model'; // Change 'root' to 'model'
        this.conns = modelJSON.connections_list;
        this.abs2prom = modelJSON.abs2prom; // May be undefined.
        this.declarePartialsList = modelJSON.declare_partials_list;
        this.useDeclarePartialsList = (this.declarePartialsList.length > 0);
        this.sysPathnamesList = modelJSON.sys_pathnames_list;

        this.maxDepth = 1;
        this.unconnectedInputs = 0;
        this.autoivcSources = 0;
        this.nodePaths = {};
        this.nodeIds = [];
        this.depthCount = [];

        startTimer('ModelData._convertToN2TreeNodes');
        this.root = this.tree = modelJSON.tree = this._convertToN2TreeNodes(modelJSON.tree);
        stopTimer('ModelData._convertToN2TreeNodes');

        startTimer('ModelData._setParentsAndDepth');
        this._setParentsAndDepth(this.root, null, 1);
        stopTimer('ModelData._setParentsAndDepth');

        if (this.unconnectedInputs > 0)
            console.info("Unconnected nodes: ", this.unconnectedInputs);

        startTimer('ModelData._initSubSystemChildren');
        this._initSubSystemChildren(this.root);
        stopTimer('ModelData._initSubSystemChildren');

        startTimer('ModelData._computeConnections');
        this._computeConnections();
        stopTimer('ModelData._computeConnections');

        debugInfo("New model: ", this);
        // this.errorCheck();
    }

    static uncompressModel(b64str) {
        const compressedData = atob(b64str);
        const jsonStr = window.pako.inflate(compressedData, { to: 'string' });
        return JSON.parse(jsonStr);
    }

    /**
     * For debugging: Make sure every tree member is an N2TreeNode.
     * @param {N2TreeNode} [node = this.root] The node to start with.
     */
    errorCheck(node = this.root) {
        if (!(node instanceof N2TreeNode))
            debugInfo('Node with problem: ', node);

        for (let prop of ['parent', 'originalParent', 'parentComponent']) {
            if (node[prop] && !(node[prop] instanceof N2TreeNode))
                debugInfo('Node with problem ' + prop + ': ', node);
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
            for (let i in newNode.children) {
                newNode.children[i] = this._convertToN2TreeNodes(newNode.children[i]);
                newNode.children[i].parent = newNode;
                if (exists(newNode.children[i].parentComponent))
                    newNode.children[i].parentComponent = newNode;
            }
        }

        return newNode;
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
        node.depth = depth;
        node.parent = parent;
        node.id = this.nodeIds.length;
        this.nodeIds.push(node);

        // Track # of nodes at each depth
        if (depth > this.depthCount.length) { this.depthCount.push(1); }
        else { this.depthCount[depth - 1]++; }

        if (node.parent) { // not root node? node.parent.absPathName : "";
            if (node.parent.absPathName != "") {
                node.absPathName += node.parent.absPathName + ".";
            }

            node.absPathName += node.name;

            this.nodePaths[node.absPathName] = node;
        }

        this.identifyUnconnectedInput(node);

        if (node.isInputOrOutput()) {
            let parentComponent = (node.originalParent) ? node.originalParent : node.parent;
            if (parentComponent.type == "subsystem" &&
                parentComponent.subsystem_type == "component") {
                node.parentComponent = parentComponent;
            }
            else {
                throw ("Input or output without a parent component!");
            }
        }

        this.maxDepth = Math.max(depth, this.maxDepth);

        if (node.isSubsystem()) {
            this.maxSystemDepth = Math.max(depth, this.maxSystemDepth);
        }

        node.childNames.add(node.absPathName); // Add the node itself

        if (node.hasChildren()) {
            node.numDescendants = node.children.length;
            for (let child of node.children) {

                let implicit = this._setParentsAndDepth(child, node, depth + 1);
                if (implicit) node.implicit = true;

                node.numDescendants += child.numDescendants;

                // Add absolute pathnames of children to a set for quick searching
                if (!node.isRoot()) { // All nodes are children of the model root 
                    node.childNames.add(child.absPathName);
                    for (let childName of child.childNames) {
                        node.childNames.add(childName);
                    }
                }
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
            if (conn.tgt.match(elementPath)) {
                return true;
            }
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
            if (conn.src.match(elementPath)) {
                return true;
            }
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

        debugInfo(elementPath + " has no connections.");
        this.unconnectedInputs++;

        return false;
    }

    hasAutoIvcSrc(elementPath) {
        for (let conn of this.conns) {
            if (conn.tgt == elementPath && conn.src.match(/^_auto_ivc.*$/)) {
                debugInfo(elementPath + " source is an auto-ivc output.");
                this.autoivcSources++;
                return true;
            }
        }

        return false;
    }

    /**
     * Create an array in each node containing references to its
     * children that are subsystems. Runs recursively over the node's
     * children array.
     * @param {N2TreeNode} node Node with children to check.
     */
    _initSubSystemChildren(node) {
        if (!node.hasChildren()) {
            return;
        }

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
        let partialsStr = tgtObj.absPathName + " > " + srcObj.absPathName;

        return this.declarePartialsList.includes(partialsStr);
    }

    /** 
     * Add all leaf descendents of specified node to the array.
     * @param {N2TreeNode} node Current node to work on.
     * @param {N2TreeNode[]} objArray Array to add to.
     */

    _addLeaves(node, objArray) {
        if (!node.isInput()) {
            objArray.push(node);
        }

        if (node.hasChildren()) {
            for (let child of node.children) {
                this._addLeaves(child, objArray);
            }
        }
    }

    /**
     * Iterate over the connections list, and find the objects that make up
     * each connection, and do some error checking. Store an array containing the
     * target object and all of its parents in the source object and all of *its*
     * parents. In the target object, store an array containing references to
     * the begin and end of all the cycle arrows.
     */
    _computeConnections() {
        let sysPathnames = this.sysPathnamesList;
        let throwLbl = 'ModelData._computeConnections: ';

        for (let conn of this.conns) {
            // Ignore connections from _auto_ivc, which is intentionally not included.
            if (conn.src.match(/^_auto_ivc.*$/)) continue;

            // Process sources
            let srcObj = this.nodePaths[conn.src];

            if (!srcObj) {
                console.warn(throwLbl + "Cannot find connection source " + conn.src);
                continue;
            }

            let srcObjParents = [srcObj];
            if (!srcObj.isOutput()) { // source obj must be output
                console.warn(throwLbl + "Found a source that is not an output.");
                continue;
            }

            if (srcObj.hasChildren()) {
                console.warn(throwLbl + "Found a source that has children.");
                continue;
            }

            for (let obj = srcObj.parent; obj != null; obj = obj.parent) {
                srcObjParents.push(obj);
            }

            // Process targets
            let tgtObj = this.nodePaths[conn.tgt];

            if (!tgtObj) {
                console.warn(throwLbl + "Cannot find connection target " + conn.tgt);
                continue;
            }

            // Target obj must be an input
            if (!tgtObj.isInput()) {
                console.warn(throwLbl + "Found a target that is NOT a input.");
                continue;
            }
            if (tgtObj.hasChildren()) {
                console.warn(throwLbl + "Found a target that has children.");
                continue;
            }

            if (!tgtObj.parentComponent) {
                console.warn(`${throwLbl} Target object ${conn.tgt} is missing a parent component.`);
                continue;
            }

            let tgtObjParents = [tgtObj];
            for (let parentObj = tgtObj.parent; parentObj != null; parentObj = parentObj.parent) {
                tgtObjParents.push(parentObj);
            }

            for (let srcParent of srcObjParents) {
                for (let tgtParent of tgtObjParents) {
                    if (tgtParent.absPathName != "")
                        srcParent.targetParentSet.add(tgtParent);

                    if (srcParent.absPathName != "")
                        tgtParent.sourceParentSet.add(srcParent);
                }
            }

            /*
             * The cycle_arrows object in each connection is an array of length-2 arrays,
             * each of which is an index into the sysPathnames array. Using that array we
             * can resolve the indexes to pathnames to the associated objects.
             */
            if (Array.isPopulatedArray(conn.cycle_arrows)) {
                let cycleArrowsArray = [];
                let cycleArrows = conn.cycle_arrows;
                for (let cycleArrow of cycleArrows) {
                    if (cycleArrow.length != 2) {
                        console.warn(throwLbl + "cycleArrowsSplitArray length not 2, got " +
                            cycleArrow.length + ": " + cycleArrow);
                        continue;
                    }

                    let srcPathname = sysPathnames[cycleArrow[0]];
                    let tgtPathname = sysPathnames[cycleArrow[1]];

                    let arrowBeginObj = this.nodePaths[srcPathname];
                    if (!arrowBeginObj) {
                        console.warn(throwLbl + "Cannot find cycle arrows begin object " + srcPathname);
                        continue;
                    }

                    let arrowEndObj = this.nodePaths[tgtPathname];
                    if (!arrowEndObj) {
                        console.warn(throwLbl + "Cannot find cycle arrows end object " + tgtPathname);
                        continue;
                    }

                    cycleArrowsArray.push({
                        "begin": arrowBeginObj,
                        "end": arrowEndObj
                    });
                }

                if (!tgtObj.parent.hasOwnProperty("cycleArrows")) {
                    tgtObj.parent.cycleArrows = [];
                }
                tgtObj.parent.cycleArrows.push({
                    "src": srcObj,
                    "arrows": cycleArrowsArray
                });
            }
        }
    }

    /**
     * If an element has no connection naming it as a source or target,
     * relabel it as unconnected.
     * @param {N2TreeNode} node The tree node to work on.
     */
    identifyUnconnectedInput(node) { // Formerly updateRootTypes
        if (!node.hasOwnProperty('absPathName')) {
            console.warn("identifyUnconnectedInput error: absPathName not set for ", node);
        }
        else {
            if (node.isInput()) {
                if (!node.hasChildren() && !this.hasAnyConnection(node.absPathName))
                    node.type = "unconnected_input";
                else if (this.hasAutoIvcSrc(node.absPathName))
                    node.type = "autoivc_input";
            }
        }
    }

}
