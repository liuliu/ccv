var ccv = require("./build/Release/ccv_node");

var bbf = new ccv.bbf('../samples/face');
var faces = bbf.detect('./face.jpg');

console.log(faces);
