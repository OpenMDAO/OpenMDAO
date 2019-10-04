/**
 * The outermost N2 class.
 * @typedef N2Diagram
 * @property {ModelData} model Processed model data received from Python.
 * @property {Object} parentDiv
 * @property {Object} d3ContentDiv The div containing all of the diagram's content.
 * @property {Object} svgDiv The div containing the SVG element.
 * @property {Object} svg The SVG element.
 * @property {Object} svgStyle Object where SVG style changes can be made.
 * @property {Object} toolTip Div to display tooltips.
 * @property {Boolean} showPath
 * @property {Array} backButtonHistory
 * @property {Array} forwardButtonHistory
 * @property {number} chosenCollapseDepth
 * @property {Boolean} updateRecomputesAutoComplete
 */
class N2Diagram {
    constructor(modelJSON) {
        this.model = new ModelData(modelJSON);
        this.showPath = false;

        this.setupContentDivs();
        this.transitionStartDelay = N2Diagram.defaultTransitionStartDelay;

        this.backButtonHistory = [];
        this.forwardButtonHistory = [];
        this.chosenCollapseDepth = -1;
        this.updateRecomputesAutoComplete = true;
    }

    /**
     * Find the divs for D3 content in the existing document, and add a style section.
     */
    setupContentDivs() {
        this.parentDiv = document.getElementById("ptN2ContentDivId");

        this.d3ContentDiv = this.parentDiv.querySelector("#d3_content_div");        
        this.svgDiv = d3.select("#svgDiv");
        this.svg = d3.select("#svgId");

        this.svgStyle = document.createElement("style");
        this.parentDiv.querySelector("#svgId").appendChild(this.svgStyle);

        this.toolTip = d3.select(".tool-tip");
    }

    updateSvgStyle() {
        // Define as JSON first
        let newCssJson = {
            'rect' : {
                'stroke': PT_STROKE_COLOR
            },
            'g.unknown > rect': {
                'fill': UNKNOWN_EXPLICIT_COLOR,
                'fill-opacity': '.8'
            },
            'g.unknown_implicit > rect': {
                'fill': UNKNOWN_IMPLICIT_COLOR,
                'fill-opacity': '.8'
            },
            
        } 
    }
}

N2Diagram.defaultTransitionStartDelay = 100;