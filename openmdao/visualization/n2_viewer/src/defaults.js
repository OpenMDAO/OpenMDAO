// Constants and default valus
const EMBEDDED = (d3.selectAll("#all_pt_n2_content_div").classed("embedded-n2"));
const _DEFAULT_N2_DIAGRAM_UNIT = 'px';
const _DEFAULT_N2_DIAGRAM_HEIGHT = window.innerHeight * .95;
const _DEFAULT_FONT_SIZE = 11;
const _DEFAULT_GAP_SIZE = _DEFAULT_FONT_SIZE + 4;

defaultDims = {
    'size': {
        'unit': _DEFAULT_N2_DIAGRAM_UNIT,
        'n2matrix': { // Dimensions of N2 matrix diagram
            'height': _DEFAULT_N2_DIAGRAM_HEIGHT,
            'width': _DEFAULT_N2_DIAGRAM_HEIGHT,
            'margin': _DEFAULT_GAP_SIZE
        },
        'partitionTree': { // Dimensions of the tree on the left side of the diagram
            'width': 0,
            'height': _DEFAULT_N2_DIAGRAM_HEIGHT,
        },
        'solverTree': { // Dimensions of the tree on the right side of the diagram
            'width': 0,
            'height': _DEFAULT_N2_DIAGRAM_HEIGHT,
        },
        'font': _DEFAULT_FONT_SIZE,
        'minColumnWidth': 2,
        'rightTextMargin': 8,
        'parentNodeWidth': 30,
        'partitionTreeGap': _DEFAULT_GAP_SIZE, // Pixels between partition tree and N2 matrix
        'svgMargin': 1,
    }
};

Object.freeze(defaultDims);

// TODO: Probably move this into N2Diagram or other class
let N2TransitionDefaults = {
    'startDelay': 100,
    'duration': 0,
    'durationFast': 1000,
    'durationSlow': 3000,
    'maxNodes': 150
}

const Precollapse = {
    'minimumNodes': 200, // Precollapse nodes in models larger than this
    'threshold': 0, // Only precollapse nodes with more descendants than this
    'grpDepthStart': 3, // Only precollapse group nodes at least this deep
    'cmpDepthStart': 2, // Only precollapse component nodes at least this deep
    'depthLimit': 6, // Only precollapse nodes with more than this many others at the same depth
    'children': 6, // Only precollapse nodes with more direct children than this
                   // (decreases by 1 w/each depth level)
}

Object.freeze(Precollapse);

let DebugFlags = {
    'timings': false,
    'info': false
}
