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
        this.searchVals0 = [];
        this.inDataFunction = true;
        
        this.autoCompleteListNames = [];
        this.autoCompleteListPathNames = [];
        this.autoCompleteSetNames = {};
        this.autoCompleteSetPathNames = {};
        this.searchCollapsedUndo = [];

        this.numMatches = 0;
        this.searchInputDiv = d3.select("#awesompleteId").node();
        this.searchCountDiv = d3.select("#searchCountId").node();

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
            "filter": function (text, input) {
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
            "item": function (text, input) {
                return Awesomplete.ITEM(text, self.filteredWord.value);
            },
            "replace": function (text) {
                let newVal = "";
                let cursorPos = 0;
                for (let i = 0; i < self.searchVals0.length; ++i) {
                    newVal += ((i == self.wordIndex) ? text : self.searchVals0[i]) + " ";
                    if (i == self.wordIndex) cursorPos = newVal.length - 1;
                }
                this.input.value = newVal;
                self.searchInputDiv.setSelectionRange(cursorPos, cursorPos);
            },
            "data": function (item/*, input*/) {
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

        window.addEventListener("awesomplete-selectcomplete", function (e) {
            // User made a selection from dropdown.
            // This is fired after the selection is applied
            self.searchInputEventListener(e);
            this.searchAwesomplete.evaluate();
        }.bind(self), false);

        // Use Capture not bubble so that this will be the first input event
        window.addEventListener('input', self.searchInputEventListener.bind(self), true);
    }

    _doSearch(node, regexMatch, undoList) {
        let didMatch = false;

        if (node.hasChildren() && !node.isMinimized) {
            // depth first, dont go into minimized children
            for (let child of node.children) {
                if (this._doSearch(child, regexMatch, undoList)) didMatch = true;
            }
        }

        if (node === this.zoomedElement) return didMatch;

        if (!didMatch && !node.hasChildren() && node.isParamOrUnknown()) {
            didMatch = regexMatch.test(node.absPathName);
            if (didMatch) {
                // only params and unknowns can count as matches
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

    _countMatches() {
        for (let node of this.searchCollapsedUndo) {
            if (!node.hasChildren() && node.isParamOrUnknown()) node.varIsHidden = true;
            else node.isMinimized = true;
        }

        this.numMatches = 0;

        if (this.searchVals0.length != 0)
            this._doSearch(this.zoomedElement,
                this._getSearchRegExp(this.searchVals0), null);

        for (let node of this.searchCollapsedUndo) {
            if (!node.hasChildren() && node.isParamOrUnknown()) node.varIsHidden = true;
            else node.isMinimized = true;
        }
    }

    performSearch() {
        for (let node of this.searchCollapsedUndo) {
            //auto undo on successive searches
            if (!node.hasChildren() && node.isParamOrUnknown()) node.varIsHidden = false;
            else node.isMinimized = false;
        }

        this.numMatches = 0;
        this.searchCollapsedUndo = [];
        if (this.searchVals0.length != 0)
            this._doSearch(this.zoomedElement, this._getSearchRegExp(this.searchVals0),
                this.searchCollapsedUndo);

    }

    _getSearchRegExp(searchValsArray) {
        let regexStr = new String("(^" + searchValsArray.join("$|^") + "$)")
            .replace(/\./g, "\\.") //convert . to regex
            .replace(/\?/g, ".") //convert ? to regex
            .replace(/\*/g, ".*?") //convert * to regex
            .replace(/\^/g, "^.*?"); //prepend *

        return new RegExp(regexStr, "i"); // case insensitive
    }

    _isValid(value) { return value.length > 0; }

    searchInputEventListener(e) {
        testThis(this, 'N2Search', 'searchInputEventListener');

        let target = e.target;
        if (target.id == "awesompleteId") {
            //valid characters AlphaNumeric : _ ? * space .
            let newVal = target.value.replace(/([^a-zA-Z0-9:_\?\*\s\.])/g, "");

            if (newVal != target.value) {
                target.value = newVal; // won't trigger new event
            }

            this.searchVals0 = target.value.split(" ");

            let filtered = this.searchVals0.filter(this._isValid);
            this.searchVals0 = filtered;

            let lastLetterTypedIndex = target.selectionStart - 1;

            let endIndex = target.value.indexOf(" ", lastLetterTypedIndex);
            if (endIndex == -1) endIndex = target.value.length;

            let startIndex = target.value.lastIndexOf(" ", lastLetterTypedIndex);
            if (startIndex == -1) startIndex = 0;

            let sub = target.value.substring(startIndex, endIndex).trim();
            // valid openmdao character types: AlphaNumeric : _ .
            this.filteredWord.value =
                sub.replace(/([^a-zA-Z0-9:_\.])/g, "");

            let i = 0;
            for (let val of this.searchVals0) {
                if (val.replace(/([^a-zA-Z0-9:_\.])/g, "") ==
                    this.filteredWord.value) {
                    this.wordIndex = i;
                    break;
                }
                ++i;
            }

            this.filteredWord.containsDot = (this.filteredWord.value.indexOf(".") != -1);
            this.searchAwesomplete.list = this.filteredWord.containsDot ?
                    this.autoCompleteListPathNames : this.autoCompleteListNames;
            this.filteredWord.baseName = this.filteredWord.containsDot ?
                    this.filteredWord.value.split(".")[0].trim() : "";

            this._countMatches();
            this.searchCountDiv.innerHTML = "" + this.numMatches + " matches";
        }
    }

    findRootOfChangeForSearch(node) {
        let earliestObj = node;
        for (let obj = node; obj != null; obj = obj.parent) {
            if (obj.isMinimized) earliestObj = obj;
        }
        return earliestObj;
    }

    /**
     * Recurse through the children of the node and add their names to the
     * autocomplete list of names if they're not already in it.
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
        if (node.splitByColon && node.hasChildren()) nodeName += ":";
        if (node.isParamOrUnknown()) nodeName += ".";
        let namesToAdd = [nodeName];

        if (node.splitByColon) namesToAdd.push(node.colonName +
            ((node.hasChildren()) ? ":" : ""));

        for (let name of namesToAdd) {
            if (!this.autoCompleteSetNames.hasOwnProperty(name)) {
                this.autoCompleteSetNames[name] = true;
                this.autoCompleteListNames.push(name);
            }
        }

        let localPathName = (this.zoomedElement === this.modelRoot) ?
            node.absPathName : node.absPathName.slice(this.zoomedElement.absPathName.length + 1);

        if (!this.autoCompleteSetPathNames.hasOwnProperty(localPathName)) {
            this.autoCompleteSetPathNames[localPathName] = true;
            this.autoCompleteListPathNames.push(localPathName);
        }
    }

    update(zoomedElement, root) {
        this.zoomedElement = zoomedElement;
        this.modelRoot = root;

        if (!this.updateRecomputesAutoComplete) {
            this.updateRecomputesAutoComplete = true;
            return;
        }

        this.autoCompleteSetNames = {};
        this.autoCompleteSetPathNames = {};

        this.autoCompleteListNames = [];
        this.autoCompleteListPathNames = [];

        this._populateAutoCompleteList(this.zoomedElement);
    }
}