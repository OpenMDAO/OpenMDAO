// From Isaias Reyes
var CONNECTION_COLOR = "black",
   UNKNOWN_IMPLICIT_COLOR = "#c7d06d",
   UNKNOWN_EXPLICIT_COLOR = "#9ec4c7",
   N2_COMPONENT_BOX_COLOR = "#555",
   N2_BACKGROUND_COLOR = "#eee",
   N2_GRIDLINE_COLOR = "white",
   PT_STROKE_COLOR = "#eee",
   UNKNOWN_GROUP_COLOR = "#888",
   PARAM_COLOR = "#32afad",
   PARAM_GROUP_COLOR = "Orchid",
   GROUP_COLOR = "#3476a2",
   COMPONENT_COLOR = "DeepSkyBlue",
   COLLAPSED_COLOR = "#555",
   UNCONNECTED_PARAM_COLOR = "#F42E0C",
   HIGHLIGHT_HOVERED_COLOR = "blue",
   RED_ARROW_COLOR = "salmon",
   GREEN_ARROW_COLOR = "seagreen";


// This is how we want to map solvers to colors and CSS classes
//    Linear             Nonlinear
//    ---------          ---------
// 0. None               None
// 1. LN: LNBJ           NL: NLBJ
// 2. LN: SCIPY
// 3. LN: RUNONCE        NL: RUNONCE
// 4. LN: Direct
// 5. LN: PETScKrylov
// 6. LN: LNBGS          NL: NLBGS
// 7. LN: USER
// 8.                    NL: Newton
// 9.                    BROYDEN
// 10. solve_linear      solve_nonlinear
// 11. other             other

// Later add these for linesearch ?
// LS: AG
// LS: BCHK

// From https://colorbrewer.org
var colors = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f'];

var linearSolverColors = {};
var linearSolverClasses = {} ;
var nonLinearSolverColors = {};
var nonLinearSolverClasses = {} ;
var linearSolverNames = [];
var nonLinearSolverNames = [] ;

function setSolverColorAndCSSClass(ln_solver, nl_solver, idx){
    if (ln_solver){
        linearSolverColors[ln_solver] = colors[idx];
        linearSolverClasses[ln_solver] = "solver_" + idx;
        linearSolverNames.push(ln_solver);
    }
    if (nl_solver){
        nonLinearSolverColors[nl_solver] = colors[idx];
        nonLinearSolverClasses[nl_solver] = "solver_" + idx
        nonLinearSolverNames.push(nl_solver);
    }
}

setSolverColorAndCSSClass( "None", "None", 0);
setSolverColorAndCSSClass( "LN: LNBJ", "NL: NLBJ", 1);
setSolverColorAndCSSClass( "LN: SCIPY", "", 2);
setSolverColorAndCSSClass( "LN: RUNONCE", "NL: RUNONCE", 3);
setSolverColorAndCSSClass( "LN: Direct", "", 4);
setSolverColorAndCSSClass( "LN: PETScKrylov", "", 5);
setSolverColorAndCSSClass( "LN: LNBGS", "NL: NLBGS", 6);
setSolverColorAndCSSClass( "LN: USER", "", 7);
setSolverColorAndCSSClass( "", "NL: Newton", 8);
setSolverColorAndCSSClass( "", "BROYDEN", 9);
setSolverColorAndCSSClass( "solve_linear", "solve_nonlinear", 10);
setSolverColorAndCSSClass( "other", "other", 11);

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

var widthPSolverTreePx = 1,
    kSolverx = 0, kSolvery = 0, kSolverx0 = 0, kSolvery0 = 0,
    xScalerPSolverTree = d3.scaleLinear().range([0, widthPSolverTreePx]),
    yScalerPSolverTree = d3.scaleLinear().range([0, HEIGHT_PX]),
    xScalerPSolverTree0 = null,
    yScalerPSolverTree0 = null;

var showLinearSolverNames = true;
