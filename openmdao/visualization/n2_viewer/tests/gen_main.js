// <<hpp_insert gen/utils.js>>
// <<hpp_insert gen/defaults.js>>
// <<hpp_insert gen/ModelData.js>>
// <<hpp_insert gen/Diagram.js>>

var sharedTransition = null;

var enterIndex = 0;
var exitIndex = 0;

// The modelData object is generated and populated by test_generic_model.py
let modelData = ModelData.uncompressModel(compressedModel);
delete compressedModel;

var n2MouseFuncs = null;

function gen_main() {
    const diag = new Diagram(modelData);
    n2MouseFuncs = diag.getMouseFuncs();

    diag.update(false);
}

gen_main();
