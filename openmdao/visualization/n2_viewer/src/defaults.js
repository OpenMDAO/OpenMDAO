// Just to make things easier to find:
const _DEFAULT_N2_DIAGRAM_UNIT = 'px';
const _DEFAULT_N2_DIAGRAM_HEIGHT = 600;
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
        'minColumnWidth': 5,
        'rightTextMargin': 8,
        'parentNodeWidth': 40,
        'partitionTreeGap': _DEFAULT_GAP_SIZE, // Pixels between partition tree and N2 matrix
        'svgMargin': 1,
    }
};

Object.freeze(defaultDims);

// TODO: Probably move this into N2Diagram or other class
let N2TransitionDefaults = {
    'startDelay': 100,
    'duration': 1000,
    'durationFast': 1000,
    'durationSlow': 1500,
    'maxNodes': 150
}

let DebugFlags = {
    'timings': false,
    'info': false
}

let colonVarNameAppend = ' '; // Used internally. Appended to vars split by colon vars
                              // Allows user to have inputs like f_approx:f, f_approx:r
                              // and outputs on the same comp as f_approx


// Object.freeze(N2TransitionDefaults);