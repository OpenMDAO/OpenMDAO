var filteredWordForAutoCompleteSuggestions = "", filteredWordForAutoCompleteSuggestionsContainsDot = false;
var filteredWordForAutoCompleteSuggestionsBaseName = "";
var wordIndex = 0;
var searchVals0 = [];
var inDataFunction = true;
var filterSet = {};
var callSearchFromEnterKeyPressed = false;
var autoCompleteListNames = [], autoCompleteListPathNames = [];
var searchCollapsedUndo = [];        
var autoCompleteSetNames = {}, autoCompleteSetPathNames = {};
var showParams = true; //default off

window.addEventListener("awesomplete-selectcomplete", function (e) {
    // User made a selection from dropdown.
    // This is fired after the selection is applied
    SearchInputEventListener(e);
    callSearchFromEnterKeyPressed = false;
    searchAwesomplete.evaluate();
}, false);

var numMatches = 0;
function DoSearch(d, regexMatch, undoList) {
    var didMatch = false;
    if (d.children && !d.isMinimized) { //depth first, dont go into minimized children
        for (var i = 0; i < d.children.length; ++i) {
            if (DoSearch(d.children[i], regexMatch, undoList)) didMatch = true;
        }
    }
    if (d === zoomedElement) return didMatch;
    if (!showParams && d.type === "param") return didMatch;
    if (!didMatch && !d.children && (d.type === "param" || d.type === "unknown")) {
        didMatch = regexMatch.test(d.absPathName);
        if (didMatch) {
            ++numMatches; //only params and unknowns can count as matches
        }
        else if (undoList) { //did not match and undo list is not null
            d.varIsHidden = true;
            undoList.push(d);
        }
    }

    if (!didMatch && d.children && !d.isMinimized && undoList) { //minimizeable and undoList not null
        d.isMinimized = true;
        undoList.push(d);
    }
    return didMatch;
}

function CountMatches() {
    searchCollapsedUndo.forEach(function (d) { //auto undo on successive searches
        if (!d.children && (d.type === "param" || d.type === "unknown")) d.varIsHidden = false;
        else d.isMinimized = false;
    });
    numMatches = 0;
    if (searchVals0.length != 0) DoSearch(zoomedElement, GetSearchRegExp(searchVals0), null);
    searchCollapsedUndo.forEach(function (d) { //auto undo on successive searches
        if (!d.children && (d.type === "param" || d.type === "unknown")) d.varIsHidden = true;
        else d.isMinimized = true;
    });
}

function SearchButtonClicked() {
    searchCollapsedUndo.forEach(function (d) { //auto undo on successive searches
        if (!d.children && (d.type === "param" || d.type === "unknown")) d.varIsHidden = false;
        else d.isMinimized = false;
    });
    numMatches = 0;
    searchCollapsedUndo = [];
    if (searchVals0.length != 0) DoSearch(zoomedElement, GetSearchRegExp(searchVals0), searchCollapsedUndo);

    FindRootOfChangeFunction = FindRootOfChangeForSearch;
    TRANSITION_DURATION = TRANSITION_DURATION_SLOW;
    lastClickWasLeft = false;
    updateRecomputesAutoComplete = false;
    updateFunc();
}

function GetSearchRegExp(searchValsArray) {
    var regexStr = "(^" + searchValsArray.join("$|^") + "$)"; //^ starts at beginning of string   $ is end of string
    regexStr = regexStr.replace(/\./g, "\\."); //convert . to regex
    regexStr = regexStr.replace(/\?/g, "."); //convert ? to regex
    regexStr = regexStr.replace(/\*/g, ".*?"); //convert * to regex
    regexStr = regexStr.replace(/\^/g, "^.*?"); //prepend *
    return new RegExp(regexStr, "i"); //case insensitive
}


function SearchInputEventListener(e) {
    var target = e.target;
    if (target.id === "awesompleteId") {
        var newVal = target.value.replace(/([^a-zA-Z0-9:_\?\*\s\.])/g, ""); //valid characters AlphaNumeric : _ ? * space .
        if (newVal !== target.value) {
            //e.stopPropagation();
            target.value = newVal; //won't trigger new event
            //searchAwesomplete.evaluate();
        }

        searchVals0 = target.value.split(" ");
        function isValid(value) {
            return value.length > 0;
        }
        var filtered = searchVals0.filter(isValid);
        searchVals0 = filtered;

        var lastLetterTypedIndex = target.selectionStart - 1;

        var endIndex = target.value.indexOf(" ", lastLetterTypedIndex);
        if (endIndex == -1) endIndex = target.value.length;
        var startIndex = target.value.lastIndexOf(" ", lastLetterTypedIndex);
        if (startIndex == -1) startIndex = 0;
        var sub = target.value.substring(startIndex, endIndex).trim();
        filteredWordForAutoCompleteSuggestions = sub.replace(/([^a-zA-Z0-9:_\.])/g, ""); //valid openmdao character types: AlphaNumeric : _ .



        for (var i = 0; i < searchVals0.length; ++i) {
            if (searchVals0[i].replace(/([^a-zA-Z0-9:_\.])/g, "") === filteredWordForAutoCompleteSuggestions) {
                wordIndex = i;
                break;
            }
        }

        filteredWordForAutoCompleteSuggestionsContainsDot = (filteredWordForAutoCompleteSuggestions.indexOf(".") != -1);
        searchAwesomplete.list = filteredWordForAutoCompleteSuggestionsContainsDot ? autoCompleteListPathNames : autoCompleteListNames;
        filteredWordForAutoCompleteSuggestionsBaseName = filteredWordForAutoCompleteSuggestionsContainsDot ? filteredWordForAutoCompleteSuggestions.split(".")[0].trim() : "";

        CountMatches();
        parentDiv.querySelector("#searchCountId").innerHTML = "" + numMatches + " matches";
    }
}
window.addEventListener('input', SearchInputEventListener, true);//Use Capture not bubble so that this will be the first input event

function SearchEnterKeyUpEventListener(e) {
    var target = e.target;
    if (target.id === "awesompleteId") {
        var key = e.which || e.keyCode;
        if (key === 13) { // 13 is enter
            if (callSearchFromEnterKeyPressed) {
                SearchButtonClicked();
            }
        }
    }
}
window.addEventListener('keyup', SearchEnterKeyUpEventListener, true);//keyup so it will be after the input and awesomplete-selectcomplete event listeners

function SearchEnterKeyDownEventListener(e) {
    var target = e.target;
    if (target.id === "awesompleteId") {
        var key = e.which || e.keyCode;
        if (key === 13) { // 13 is enter
            callSearchFromEnterKeyPressed = true;
        }
    }
}
window.addEventListener('keydown', SearchEnterKeyDownEventListener, true);//keydown so it will be before the input and awesomplete-selectcomplete event listeners

var searchInputId = parentDiv.querySelector("#awesompleteId");
var searchAwesomplete = new Awesomplete(searchInputId, {
    "minChars": 1,
    "maxItems": 15,
    "list": [],
    "filter": function (text, input) {
        if (inDataFunction) {
            inDataFunction = false;
            filterSet = {};
        }
        if (filteredWordForAutoCompleteSuggestions.length == 0) return false;
        if (filterSet.hasOwnProperty(text)) return false;
        filterSet[text] = true;
        if (filteredWordForAutoCompleteSuggestionsContainsDot) return Awesomplete.FILTER_STARTSWITH(text, filteredWordForAutoCompleteSuggestions);
        return Awesomplete.FILTER_CONTAINS(text, filteredWordForAutoCompleteSuggestions);
    },
    "item": function (text, input) {
        return Awesomplete.ITEM(text, filteredWordForAutoCompleteSuggestions);
    },
    "replace": function (text) {
        var newVal = "";
        var cursorPos = 0;
        for (var i = 0; i < searchVals0.length; ++i) {
            newVal += ((i == wordIndex) ? text : searchVals0[i]) + " ";
            if (i == wordIndex) cursorPos = newVal.length - 1;
        }
        this.input.value = newVal;
        parentDiv.querySelector("#awesompleteId").setSelectionRange(cursorPos, cursorPos);
    },
    "data": function (item/*, input*/) {
        inDataFunction = true;
        if (filteredWordForAutoCompleteSuggestionsContainsDot) {
            var baseIndex = item.toLowerCase().indexOf("." + filteredWordForAutoCompleteSuggestionsBaseName.toLowerCase() + ".");
            if (baseIndex > 0) return item.slice(baseIndex + 1);
        }
        return item;
    }
});

function FindRootOfChangeForSearch(d) {
    var earliestObj = d;
    for (var obj = d; obj != null; obj = obj.parent) {
        if (obj.isMinimized) earliestObj = obj;
    }
    return earliestObj;
}

function PopulateAutoCompleteList(d) {
    if (d.children && !d.isMinimized) { //depth first, dont go into minimized children
        for (var i = 0; i < d.children.length; ++i) {
            PopulateAutoCompleteList(d.children[i]);
        }
    }
    if (d === zoomedElement) return;
    if (!showParams && d.type === "param") return;

    var n = d.name;
    if (d.splitByColon && d.children && d.children.length > 0) n += ":";
    if (d.type !== "param" && d.type !== "unknown") n += ".";
    var namesToAdd = [n];

    if (d.splitByColon) namesToAdd.push(d.colonName + ((d.children && d.children.length > 0) ? ":" : ""));

    namesToAdd.forEach(function (name) {
        if (!autoCompleteSetNames.hasOwnProperty(name)) {
            autoCompleteSetNames[name] = true;
            autoCompleteListNames.push(name);
        }
    });

    var localPathName = (zoomedElement === root) ? d.absPathName : d.absPathName.slice(zoomedElement.absPathName.length + 1);
    if (!autoCompleteSetPathNames.hasOwnProperty(localPathName)) {
        autoCompleteSetPathNames[localPathName] = true;
        autoCompleteListPathNames.push(localPathName);
    }
}