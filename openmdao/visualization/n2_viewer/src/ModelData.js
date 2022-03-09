// <<hpp_insert libs/pako_inflate.min.js>>
// <<hpp_insert libs/json5_2.2.0.min.js>>
// <<hpp_insert src/N2TreeNode.js>>

const defaultAttribNames = {
    name: 'name',              // Human-readable label for the node
    type: 'type',              // Label for the node's classification property
    descendants: 'children',   // Property that contains the node's descendants
    links: 'connections_list', // Property the contains the list of connections, in
                               // a [{src: "srcname1", tgt: "tgtname1", ...}] format
    srcType: 'output',         // Type of node a connection starts from
    tgtType: 'input'           // Type of node where a connection ends
};

class ModelData {
    constructor(modelJSON, attribNames = defaultAttribNames) {
        // console.log(modelJSON);

        this._attribNames = attribNames;
        this.conns = modelJSON[attribNames.links];
        this.maxDepth = 1;
        this.nodePaths = {};
        this.nodeIds = [];
        this.depthCount = [];

        this._init(modelJSON);

        this.root = this.tree = modelJSON.tree = this._adoptNodes(modelJSON.tree);
        this._setParentsAndDepth(this.root, null, 1);
        this._computeConnections();

    }

    _init(modelJSON) { }

    static uncompressModel(b64str) {
        const compressedData = atob(b64str);
        const jsonStr = window.pako.inflate(compressedData, { to: 'string' });

        // JSON5 can handle Inf and NaN
        return JSON5.parse(jsonStr); 
    }

    _newNode(element, attribNames) {
        return new N2TreeNode(element, attribNames);
    }

    /**
     * Recurse over the tree and replace the JSON objects
     * provided by n2_viewer.py with N2TreeNodes.
     * @param {Object} element The current element being updated.
     */
     _adoptNodes(element) {
        const newNode = this._newNode(element, this._attribNames);

        if (newNode.hasChildren()) {
            for (let i in newNode.children) {
                newNode.children[i] = this._adoptNodes(newNode.children[i]);
                newNode.children[i].parent = newNode;
                if (exists(newNode.children[i].parentComponent))
                    newNode.children[i].parentComponent = newNode;
            }
        }

        newNode.addFilterChild(this._attribNames);

        return newNode;
    }

   /**
     * Sets parents and depth of all nodes, and determine max depth.
     * @param {N2TreeNode} node Item to process.
     * @param {N2TreeNode} parent Parent of node, null for root node.
     * @param {number} depth Numerical level of ancestry.
     */
    _setParentsAndDepth(node, parent, depth) {
        node.depth = depth;
        node.parent = parent;
        node.id = this.nodeIds.length;
        this.nodeIds.push(node);

        // Track # of nodes at each depth
        if (depth > this.depthCount.length) { this.depthCount.push(1); }
        else { this.depthCount[depth - 1]++; }

        if (node.parent) {
            node.uuid = (node.parent.uuid == '')? node.name : `${node.parent.uuid}.${node.name}`;
            this.nodePaths[node.uuid] = node;
        }

        this.maxDepth = Math.max(depth, this.maxDepth);
        node.childNames.add(node.uuid); // Add the node itself

        if (node.hasChildren()) {
            node.numDescendants = node.children.length;
            for (const child of node.children) {
                this._setParentsAndDepth(child, node, depth + 1);
                node.numDescendants += child.numDescendants;

                // Add absolute pathnames of children to a set for quick searching
                if (!node.isRoot()) { // All nodes are children of the model root
                    node.childNames.add(child.uuid);
                    for (const childName of child.childNames) {
                        node.childNames.add(childName);
                    }
                }
            }
        }
    }

    /**
     * Check the entire array of model connections for any with a target matching
     * the specified path.
     * @param {string} elementPath The full path of the element to check.
     * @return True if the path is found as a target in the connection list.
     */
     hasInputConnection(elementPath) {
        for (const conn of this.conns) {
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
        for (const conn of this.conns) {
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
        for (const conn of this.conns) {
            if (conn.src == elementPath || conn.tgt == elementPath)
                return true;
        }

        debugInfo(elementPath + " has no connections.");
        this.unconnectedInputs++;

        return false;
    }

    /** A stub to be overridden by a derived class */
    _additionalConnProcessing(conn, srcObj, tgtObj) { }

    /**
     * Iterate over the connections list, and find the objects that make up
     * each connection, and do some error checking. Store an array containing the
     * target object and all of its parents in the source object and all of *its*
     * parents.
     */
    _computeConnections() {
        const throwLbl = 'ModelData._computeConnections: ';

        for (const conn of this.conns) {
            // Process sources
            const srcObj = this.nodePaths[conn.src];

            if (!srcObj) {
                console.warn(throwLbl + "Cannot find connection source " + conn.src);
                continue;
            }

            const srcObjParents = [srcObj];
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
            const tgtObj = this.nodePaths[conn.tgt];

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

            const tgtObjParents = [tgtObj];
            for (let parentObj = tgtObj.parent; parentObj != null; parentObj = parentObj.parent) {
                tgtObjParents.push(parentObj);
            }

            for (const srcParent of srcObjParents) {
                for (const tgtParent of tgtObjParents) {
                    if (tgtParent.absPathName != "")
                        srcParent.targetParentSet.add(tgtParent);

                    if (srcParent.absPathName != "")
                        tgtParent.sourceParentSet.add(srcParent);
                }
            }

            this._additionalConnProcessing(conn, srcObj, tgtObj);
        }
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
            for (const child of node.children) {
                this._addLeaves(child, objArray);
            }
        }
    }
}

/** Process the tree, connections, and other info provided about the model. */
class OmModelData extends ModelData {
    /** Do some discovery in the tree and rearrange & enhance where necessary. */
    constructor(modelJSON) {
        super(modelJSON);

        if (this.unconnectedInputs > 0)
            console.info("Unconnected nodes: ", this.unconnectedInputs);

        this._initSubSystemChildren(this.root);
        this._updateAutoIvcNames();

        debugInfo("New model: ", this);
        // this.errorCheck();
    }

    /** Tasks to perform early from the superclass constructor */
    _init(modelJSON) {
        modelJSON.tree.name = 'model'; // Change 'root' to 'model'
        this.abs2prom = modelJSON.abs2prom; // May be undefined.
        this.declarePartialsList = modelJSON.declare_partials_list;
        this.useDeclarePartialsList = (this.declarePartialsList.length > 0);
        this.sysPathnamesList = modelJSON.sys_pathnames_list;

        this.unconnectedInputs = 0;
        this.autoivcSources = 0;
        this.md5_hash = modelJSON.md5_hash; // compute here instead of python?
    }

    /**
     * For debugging: Make sure every tree member is an N2TreeNode.
     * @param {N2TreeNode} [node = this.root] The node to start with.
     */
    errorCheck(node = this.root) {
        if (!(node instanceof N2TreeNode))
            debugInfo('Node with problem: ', node);

        for (const prop of ['parent', 'originalParent', 'parentComponent']) {
            if (node[prop] && !(node[prop] instanceof N2TreeNode))
                debugInfo('Node with problem ' + prop + ': ', node);
        }

        if (node.hasChildren()) {
            for (let child of node.children) {
                this.errorCheck(child);
            }
        }
    }

    _newNode(element, attribNames) {
        return new OmTreeNode(element, attribNames);
    }

    /**
     * Sets parents and depth of all nodes, and determine max depth. Flags the
     * parent node as implicit if the node itself is implicit.
     * @param {N2TreeNode} node Item to process.
     * @param {N2TreeNode} parent Parent of node, null for root node.
     * @param {number} depth Numerical level of ancestry.
     */
     _setParentsAndDepth(node, parent, depth) {
        super._setParentsAndDepth(node, parent, depth);

        if (this.abs2prom.input[node.uuid] !== undefined) {
            node.promotedName = this.abs2prom.input[node.uuid];
        }
        else if (this.abs2prom.output[node.uuid] !== undefined) {
            node.promotedName = this.abs2prom.output[node.uuid];
        }

        this.identifyUnconnectedInput(node);
        if (node.isInputOrOutput()) {
            const parentComponent = (node.originalParent) ? node.originalParent : node.parent;
            if (parentComponent.type == "subsystem" &&
                parentComponent.subsystem_type == "component") {
                node.parentComponent = parentComponent;
            }
            else {
                throw ("Input or output without a parent component!");
            }
        }

        if (node.isSubsystem()) {
            this.maxSystemDepth = Math.max(depth, this.maxSystemDepth);
        }

        if (parent && node.implicit) { parent.implicit = true; }
    }

    hasAutoIvcSrc(elementPath) {
        for (const conn of this.conns) {
            if (conn.tgt == elementPath && conn.src.match(/^_auto_ivc.*$/)) {
                debugInfo(elementPath + " source is an auto-ivc output.");
                this.autoivcSources++;
                return true;
            }
        }

        return false;
    }

    /**
     * Find the target of an Auto-IVC variable.
     * @param {String} elementPath The full path of the element to check. Must start with _auto_ivc.
     * @return {String} The absolute path of the target element, or undefined if not found.
     */
    getAutoIvcTgt(elementPath) {
        if (!elementPath.match(/^_auto_ivc.*$/)) return undefined;

        for (const conn of this.conns) {
            if (conn.src == elementPath) {
                return conn.tgt;
            }
        }

        console.warn(`No target connection found for ${elementPath}.`)
        return undefined;
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

        for (const child of node.children) {
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
     * The cycle_arrows object in each connection is an array of length-2 arrays,
     * each of which is an index into the sysPathnames array. Using that array we
     * can resolve the indexes to pathnames to the associated objects.
     * @param {Object} conn Reference to the connection to operate on.
     */
    _additionalConnProcessing(conn, srcObj, tgtObj) {
        const sysPathnames = this.sysPathnamesList;
        const throwLbl = 'ModelData._computeConnections: ';

        if (Array.isPopulatedArray(conn.cycle_arrows)) {
            const cycleArrowsArray = [];
            for (const cycleArrow of conn.cycle_arrows) {
                if (cycleArrow.length != 2) {
                    console.warn(throwLbl + "cycleArrowsSplitArray length not 2, got " +
                        cycleArrow.length + ": " + cycleArrow);
                    continue;
                }

                const srcPathname = sysPathnames[cycleArrow[0]];
                const tgtPathname = sysPathnames[cycleArrow[1]];

                const arrowBeginObj = this.nodePaths[srcPathname];
                if (!arrowBeginObj) {
                    console.warn(throwLbl + "Cannot find cycle arrows begin object " + srcPathname);
                    continue;
                }

                const arrowEndObj = this.nodePaths[tgtPathname];
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

    /**
     * If the Auto-IVC component exists, rename its child variables to their
     * promoted names so they can be easily recognized instead of as v0, v1, etc.
     */
    _updateAutoIvcNames() {
        const aivc = this.nodePaths['_auto_ivc'];
        if (aivc !== undefined && aivc.hasChildren()) {
            for (const ivc of aivc.children) {
                if (!ivc.isFilter()) {
                    const tgtPath = this.getAutoIvcTgt(ivc.absPathName);

                    if (tgtPath !== undefined) {
                        ivc.promotedName = this.nodePaths[tgtPath].promotedName;
                    }
                }
            }
        }
    }

    /**
     * If an element has no connection naming it as a source or target,
     * relabel it as unconnected.
     * @param {N2TreeNode} node The tree node to work on.
     */
    identifyUnconnectedInput(node) {
        if (!node.hasOwnProperty('uuid')) {
            console.warn("identifyUnconnectedInput error: uuid not set for ", node);
        }
        else {
            if (node.isInput()) {
                if (!node.hasChildren() && !this.hasAnyConnection(node.uuid))
                    node.type = "unconnected_input";
                else if (this.hasAutoIvcSrc(node.uuid))
                    node.type = "autoivc_input";
            }
        }
    }
}
