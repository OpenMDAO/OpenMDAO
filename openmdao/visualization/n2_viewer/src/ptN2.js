let sharedTransition = null;

var enterIndex = 0;
var exitIndex = 0;

// The modelData object is generated and populated by n2_viewer.py
var modelData  = ModelData.uncompressModel(compressedModel);
delete compressedModel;

let n2Diag = new N2Diagram(modelData);
let n2MouseFuncs = n2Diag.getMouseFuncs();
n2Diag.update(false);

