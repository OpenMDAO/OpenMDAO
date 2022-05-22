// Constants and default values
const EMBEDDED = (d3.selectAll("#all-diagram-content").classed("embedded"));
const _DEFAULT_DIAGRAM_UNIT = 'px';
const _DEFAULT_DIAGRAM_HEIGHT = window.innerHeight * .95;
const _DEFAULT_FONT_SIZE = 11;
const _DEFAULT_GAP_SIZE = _DEFAULT_FONT_SIZE + 4;

defaultDims = {
    'size': {
        'unit': _DEFAULT_DIAGRAM_UNIT,
        'matrix': {
            // Dimensions of the matrix grid
            'height': _DEFAULT_DIAGRAM_HEIGHT,
            'width': _DEFAULT_DIAGRAM_HEIGHT,
            'margin': _DEFAULT_GAP_SIZE
        },
        'partitionTree': {
            // Dimensions of the tree on the left side of the diagram
            'width': 0,
            'height': _DEFAULT_DIAGRAM_HEIGHT,
        },
        'font': _DEFAULT_FONT_SIZE,
        'minColumnWidth': 2,
        'rightTextMargin': 8,
        'parentNodeWidth': 30,
        'partitionTreeGap': _DEFAULT_GAP_SIZE, // Pixels between model tree and matrix grid
        'svgMargin': 1,
    }
};

Object.freeze(defaultDims);

// TODO: Probably move this into Diagram or other class
const transitionDefaults = {
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
    'cmpDepthStart': 2, // Only precollapse nodes w/variable children at least this deep
    'depthLimit': 6, // Only precollapse nodes with more than this many others at the same depth
    'children': 6, // Only precollapse nodes with more direct children than this
                   // (decreases by 1 w/each depth level)
}

Object.freeze(Precollapse);

const DebugFlags = {
    'timings': false,
    'info': false
}

/**
 * Create a D3 transition that creates animations for selected geometry changes.
 * @param {Number} transitionStartDelay Number of milliseconds before animation starts.
 * @returns {Transition} The newly created transition.
 */
function getTransition(transitionStartDelay = transitionDefaults.startDelay) {
    return d3.transition('global')
    .duration(transitionDefaults.duration)
    .delay(transitionStartDelay)
    // Hide the transition waiting animation when it ends:
    .on('end', () => d3.select('#waiting-container').attr('class', 'no-show'));
}
