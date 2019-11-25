let sharedTransition = null;

let enterIndex = 0,
    exitIndex = 0;

let colonVarNameAppend = '['; // Used internally. Appended to vars split by colon vars
                              // Allows user to have inputs like f_approx:f, f_approx:r
                              // and outputs on the same comp as f_approx

// The modelData object is generated and populated by n2_viewer.py
let n2Diag = new N2Diagram(modelData);
let n2MouseFuncs = n2Diag.getMouseFuncs();
n2Diag.update(false);
SetupLegend(d3, n2Diag.dom.d3ContentDiv);