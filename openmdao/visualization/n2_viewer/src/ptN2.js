let sharedTransition = null;

let enterIndex = 0,
    exitIndex = 0;

// The modelData object is generated and populated by n2_viewer.py
let n2Diag = new N2Diagram(modelData);
let n2MouseFuncs = n2Diag.getMouseFuncs();
n2Diag.update(false);