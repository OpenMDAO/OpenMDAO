/**
 * Manage the search functions for N2, which uses the Awesomplete widget from
 * https://leaverou.github.io/awesomplete/
 * @typedef N2Search
 */
class N2Search {
    /**
     * Initialize N2Search object properties and Awesomeplete.
     * @param {N2TreeNode} zoomedElement The selected node in the model tree.
     * @param {N2TreeNode} root The base element of the model tree.
     */
    constructor(zoomedElement, root) {
        // Used for autocomplete suggestions:
        this.filteredWord = {
            'value': "",
            'containsDot': false,
            'baseName': ""

        }
        this.filterSet = {};

        this.updateRecomputesAutoComplete = true;

        this.wordIndex = 0;
        this.searchVals = [];
        this.inDataFunction = true;

        this.searchCollapsedUndo = []; // Non-matching nodes to be minimized/hidden.

        this.numMatches = 0;
        this.searchInputDiv = d3.select("#awesompleteId").node();
        this.searchCountDiv = d3.select("#searchCountId");

        this._setupAwesomplete();
        this._addEventListeners();
        this.update(zoomedElement, root);
    }

    /** Initialize the Awesomplete widget. */
    _setupAwesomplete() {
        let self = this;
        this.searchAwesomplete = new Awesomplete(self.searchInputDiv, {
            "minChars": 1,
            "maxItems": 15,
            "list": [],
            "filter": function(text, input) {
                if (self.inDataFunction) {
                    self.inDataFunction = false;
                    self.filterSet = {};
                }

                if (self.filteredWord.value.length == 0) return false;

                if (self.filterSet.hasOwnProperty(text)) return false;

                self.filterSet[text] = true;

                if (self.filteredWord.containsDot)
                    return Awesomplete.FILTER_STARTSWITH(text,
                        self.filteredWord.value);

                return Awesomplete.FILTER_CONTAINS(text,
                    self.filteredWord.value);
            },
            "item": function(text, input) {
                return Awesomplete.ITEM(text, self.filteredWord.value);
            },
            "replace": function(text) {
                let newVal = "";
                let cursorPos = 0;
                for (let i = 0; i < self.searchVals.length; ++i) {
                    newVal += ((i == self.wordIndex) ? text : self.searchVals[i]) + " ";
                    if (i == self.wordIndex) cursorPos = newVal.length - 1;
                }
                this.input.value = newVal;
                self.searchInputDiv.setSelectionRange(cursorPos, cursorPos);
            },
            "data": function(item /*, input*/ ) {
                self.inDataFunction = true;
                if (self.filteredWord.containsDot) {
                    let baseIndex = item.toLowerCase().indexOf("." +
                        self.filteredWord.baseName.toLowerCase() + ".");
                    if (baseIndex > 0) return item.slice(baseIndex + 1);
                }
                return item;
            }
        });
    }

    /**
     * Add a couple of event listeners that are easier to do from here
     * than in N2UserInterface.
     */
    _addEventListeners() {
        let self = this;

       d3.select('body').on('awesomplete-selectcomplete', function() {
           self.searchInputEventListener();
           self.searchAwesomplete.evaluate();
       });

       d3.select('body').on('input', this.searchInputEventListener.bind(this));
    }

    /**
     * Recurse through the tree and find nodes with pathnames that match
     * the computed regular expression. Minimize/hide nodes that don't match.
     * @param {N2TreeNode} node The current node to operate on.
     * @param {RegExp} regexMatch A regular expression assembled from the search values.
     * @param {Array} undoList List of nodes that have been hidden/minimized.
     * @returns {Boolean} True if a match was found, false otherwise.
     */
    _doSearch(node, regexMatch, undoList) {
        let didMatch = false;

        if (node.hasChildren() && !node.isMinimized) {
            // depth first, dont go into minimized children
            for (let child of node.children) {
                if (this._doSearch(child, regexMatch, undoList)) didMatch = true;
            }
        }

        if (node === this.zoomedElement) return didMatch;

        if (!didMatch && !node.hasChildren() && node.isInputOrOutput()) {
            didMatch = regexMatch.test(node.absPathName);
            if (didMatch) {
                // only inputs and outputs can count as matches
                ++this.numMatches;
            }
            else if (undoList) {
                // did not match and undo list is not null
                node.varIsHidden = true;
                undoList.push(node);
            }
        }

        if (!didMatch && node.hasChildren() && !node.isMinimized && undoList) {
            // minimizeable and undoList not null
            node.isMinimized = true;
            undoList.push(node);
        }

        return didMatch;
    }

    /**
     * Reset the number of matches to zero and execute the search with a null value
     * for undoList, so it's not changed.
     */
    _countMatches() {
        this.numMatches = 0;

        if (this.searchVals.length != 0)
            this._doSearch(this.zoomedElement, this._getSearchRegExp(this.searchVals), null);
    }

    /** Undo results of the previous search, and perform a new one. */
    performSearch() {
        for (let node of this.searchCollapsedUndo) {
            //auto undo on successive searches
            if (!node.hasChildren() && node.isInputOrOutput()) node.varIsHidden = false;
            else node.isMinimized = false;
        }

        this.numMatches = 0;
        this.searchCollapsedUndo = [];
        if (this.searchVals.length != 0)
            this._doSearch(this.zoomedElement, this._getSearchRegExp(this.searchVals),
                this.searchCollapsedUndo);

    }

    /** Do some escaping and replacing of globbing with regular expressions. */
    _getSearchRegExp(searchValsArray) {
        let regexStr = new String("(^" + searchValsArray.join("$|^") + "$)")
            .replace(/\./g, "\\.") //convert . to regex
            .replace(/\?/g, ".") //convert ? to regex
            .replace(/\*/g, ".*?") //convert * to regex
            .replace(/\^/g, "^.*?"); //prepend *

        return new RegExp(regexStr, "i"); // case insensitive
    }

    _isValid(value) {
        return value.length > 0;
    }

    /**
     * React to each value entered into the search input box.
     * @param {Event} e The object describing the keypress event.
     */
    searchInputEventListener() {
        testThis(this, 'N2Search', 'searchInputEventListener');

        let target = d3.event.target;
        if (target.id != "awesompleteId") return;

        //valid characters AlphaNumeric : _ ? * space .
        let newVal = target.value.replace(/([^a-zA-Z0-9:_\?\*\s\.])/g, "");

        if (newVal != target.value) {
            target.value = newVal; // won't trigger new event
        }

        this.searchVals = target.value.split(" ");

        let filtered = this.searchVals.filter(this._isValid);
        this.searchVals = filtered;

        let lastLetterTypedIndex = target.selectionStart - 1;

        let endIndex = target.value.indexOf(" ", lastLetterTypedIndex);
        if (endIndex == -1) endIndex = target.value.length;

        let startIndex = target.value.lastIndexOf(" ", lastLetterTypedIndex);
        if (startIndex == -1) startIndex = 0;

        let sub = target.value.substring(startIndex, endIndex).trim();
        // valid openmdao character types: AlphaNumeric : _ .
        this.filteredWord.value = sub.replace(/([^a-zA-Z0-9:_\.])/g, "");

        let i = 0;
        for (let val of this.searchVals) {
            if (val.replace(/([^a-zA-Z0-9:_\.])/g, "") == this.filteredWord.value) {
                this.wordIndex = i;
                break;
            }
            ++i;
        }

        this.filteredWord.containsDot = (this.filteredWord.value.indexOf(".") != -1);
        this.searchAwesomplete.list = this.filteredWord.containsDot ?
            this.autoComplete.paths.list : this.autoComplete.names.list;
        this.filteredWord.baseName = this.filteredWord.containsDot ?
            this.filteredWord.value.split(".")[0].trim() : "";

        this._countMatches();
        this.searchCountDiv.html("" + this.numMatches + " matches");
    }

    /**
     * Find the earliest minimized parent of the specified node.
     * @param {N2TreeNode} node The node to search from.
     * @returns {N2TreeNode} The earliest mimimized parent node.
     */
    findRootOfChangeForSearch(node) {
        let earliestObj = node;
        for (let obj = node; obj != null; obj = obj.parent) {
            if (obj.isMinimized) earliestObj = obj;
        }
        return earliestObj;
    }

    /**
     * Recurse through the children of the node and add their names to the
     * autocomplete list of names, if they're not already in it.
     * @param {N2TreeNode} node The node to search from.
     */
    _populateAutoCompleteList(node) {
        if (node.hasChildren() && !node.isMinimized) {
            // Depth first, dont go into minimized children
            for (let child of node.children) {
                this._populateAutoCompleteList(child);
            }
        }

        if (node === this.zoomedElement) return;

        let nodeName = node.name;
        if (! node.isInputOrOutput()) nodeName += ".";
        let namesToAdd = [nodeName];

        for (let name of namesToAdd) {
            if (!this.autoComplete.names.set.hasOwnProperty(name)) {
                this.autoComplete.names.set[name] = true;
                this.autoComplete.names.list.push(name);
            }
        }

        let localPathName = (this.zoomedElement === this.modelRoot) ?
            node.absPathName : node.absPathName.slice(this.zoomedElement.absPathName.length + 1);

        if (!this.autoComplete.paths.set.hasOwnProperty(localPathName)) {
            this.autoComplete.paths.set[localPathName] = true;
            this.autoComplete.paths.list.push(localPathName);
        }
    }

    /**
     * If the zoomed element has changed, update the auto complete lists.
     * @param {N2TreeNode} zoomedElement The selected node in the model tree.
     * @param {N2TreeNode} root The base element of the model tree.
     */
    update(zoomedElement, root) {
        this.zoomedElement = zoomedElement;
        this.modelRoot = root;

        if (!this.updateRecomputesAutoComplete) {
            this.updateRecomputesAutoComplete = true;
            return;
        }

        this.autoComplete = {
            'names': {
                'list': [],
                'set': {}
            },
            'paths': {
                'list': [],
                'set': {}
            }
        }

        this._populateAutoCompleteList(this.zoomedElement);
    }
}
