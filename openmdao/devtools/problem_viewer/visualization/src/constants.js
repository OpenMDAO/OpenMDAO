//color constants
var CONNECTION_COLOR = "black",
    UNKNOWN_IMPLICIT_COLOR = "orange",
    UNKNOWN_EXPLICIT_COLOR = "#AAA",
    RED_ARROW_COLOR = "salmon",
    GREEN_ARROW_COLOR = "seagreen",
    PARAM_COLOR = "Plum",
    UNCONNECTED_PARAM_COLOR = "#F42E0C",
    GROUP_COLOR = "steelblue",
    COMPONENT_COLOR = "DeepSkyBlue",
    COLLAPSED_COLOR = "#555";
HIGHLIGHT_HOVERED_COLOR = "blue"

var widthPTreePx = 1,
    kx = 0, ky = 0, kx0 = 0, ky0 = 0,
    HEIGHT_PX = 600,
    PARENT_NODE_WIDTH_PX = 40,
    MIN_COLUMN_WIDTH_PX = 5,
    SVG_MARGIN = 1,
    TRANSITION_DURATION_FAST = 1000,
    TRANSITION_DURATION_SLOW = 1500,
    TRANSITION_DURATION = TRANSITION_DURATION_FAST,
    xScalerPTree = d3.scaleLinear().range([0, widthPTreePx]),
    yScalerPTree = d3.scaleLinear().range([0, HEIGHT_PX]),
    xScalerPTree0 = null,
    yScalerPTree0 = null,
    LEVEL_OF_DETAIL_THRESHOLD = HEIGHT_PX / 3; //3 pixels