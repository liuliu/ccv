Cache: We are Terrible Magicians
=================

ccv uses an application-wide transparent cache to de-duplicate matrix computations.
In the following chapters, I will try to outline how that works, and expose you
to the inner-working of ccv's core functionalities.

Initial Signature
-----------------

ccv_make_matrix_immutable computes the SHA-1 hash on matrix raw data, and will
use the first 64-bit as the signature for that matrix.

Derived Signature
-----------------

Derived signature is computed from the specific operation that is going to perform.
For example, matrix A and matrix B used to generate matrix C through operation X.
C's signature is derived from A, B and X.

A Radix-tree LRU Cache
----------------------

ccv uses a custom radix-tree implementation with generation information. It imposes
a hard limit on memory usage of 64 MiB, you can adjust this value if you like.
The custom radix-tree data structure is specifically designed to satisfy our 64-bit
signature design. If compile with jemalloc, it can be both fast and memory-efficient.

Garbage Collection
------------------

The matrix signature is important. For every matrix that is freed with ccv_matrix_free
directive, it will first check the signature. If it is a derived signature,
ccv_matrix_free won't free that matrix to OS immediately, instead, it will put
that matrix back to the application-wide cache. Sparse matrix, matrix without
signature / with initial signature will be freed immediately.

Shortcut
--------

For operation X performed with matrix A and B, it will first generate the derived
signature. The signature will be searched in the application-wide cache in hope
of finding a result matrix. If such matrix C is found, the operation X will take
a shortcut and return that matrix to user. Otherwise, it will allocate such matrix,
set proper signature on it and perform the operation honestly.

After finish this, I found that it may not be the most interesting bit of ccv.
But still, hope you found it otherwise :-)
