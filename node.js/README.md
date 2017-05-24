Node.js Bindings for CCV 
========================

Build Instructions
------------------

Assuming you have a functional node and npm install:

- Install node-gyp, `npm install node-gyp -g`
- Change to ccv's node.js dir, `cd node.js`
- Compile the bindings, `node-gyp configure build`

Note: on OS X, you'll need to have the dylib version of libccv compiled (pending merge, pull request #44).

- `cd .. && make libccv.dylib`

You should now have a shiny native node.js module for libccv: `node.js/build/Release/ccv_node.node`


Usage
-----
The module is in its infancy, only bbf detection is supported. Expect problems. Try out the example file `node face.js`

You may need to prepend the executable with a platform-specific environment variable to let the module know where to find libccv.

On OS X, for example: ``` DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:`pwd`/../lib node face.js```
