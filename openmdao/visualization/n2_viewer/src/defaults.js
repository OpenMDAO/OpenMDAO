// Just to make things easier to find:
const _DEFAULT_N2_DIAGRAM_UNIT = 'px';
const _DEFAULT_N2_DIAGRAM_HEIGHT = 600;

defaultDims = {
    'size': {
        'unit': _DEFAULT_N2_DIAGRAM_UNIT,
        'diagram': { // Overall dimensions of diagram
            'height': _DEFAULT_N2_DIAGRAM_HEIGHT,
            'width': _DEFAULT_N2_DIAGRAM_HEIGHT,
        },
        'partitionTree': { // Dimensions of the tree on the left side of the diagram
            'width': 0,
            'height': _DEFAULT_N2_DIAGRAM_HEIGHT
        },
        'solverTree': { // Dimensions of the tree on the right side of the diagram
            'width': 0,
            'height': _DEFAULT_N2_DIAGRAM_HEIGHT
        },
        'font': 11,
        'minColumnWidth': 5,
        'rightTextMargin': 8,
        'parentNodeWidth': 40,
        'partitionTreeGap': 10, // Pixels between partition tree and N2 matrix
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
    'info': true
}

// Object.freeze(N2TransitionDefaults);