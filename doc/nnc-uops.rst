NNC Micro Ops
=============

A machine-learning framework constructed with differentiable ops can have a problem with too many of them. This is necessary though. As we entering into the ever fast-growing AI landscape, each model is a bit of different. Even with intention to limit how many ops are,  NNC has accumulated, over years, around 50 ops. PyTorch and TensorFlow, each has more than 100 ops and counting.

Implementing these ops are not the most annoying part. The annoying part comes from implementing them for different platforms, with special optimization techniques in mind. That means, for example, Convolution can have at least 6 implementations: A CUDA direct convolution, A 3x3 optimized CUDA Winograd implementation, x86_64 direct, x86_64 3x3 Winograd, aarch64 direct, aarch64 3x3 Winograd. This is just a tip of iceberg, there are probably more than 7 separate implementations for convolution in cuDNN alone.

Apache TVM tries to solve this effectively heterogeneous computation problem by automatically generating and scheduling new op kernels on different platforms.

I am not interested in performance. NNC is a small project, performance tuning on heterogeneous computation environment can easily absorb all my time with various degrees of failures. It is better left that for capable hands.

I am interested in correctness of these ops, and a quick way to validate the correctness claims of various kernel implementations (in NNC terminology, *backend* implementations). The best to have confidence in correctness for ops, is through randomized Oracle tests.

Oracle tests, however, require a canonical implementation to begin with. Now, the question turns to: how to have confidence in the canonical implementation? I don't have a definitive answer, but from my experience, the easiest way to gain confidence in a piece of code is to have less lines of code and run the same code with as much scenarios as possible.

Thus, I am searching for something that can describe all the ops with as few lines of code as possible, and underneath, has a very small kernel to execute the description.

`Halide language <https://halide-lang.org/>`_ and works inspired by Halide such as JAX comes close to that realization. But it is not really an option to ship Halide or JAX as a dependency for NNC. On the other hand, having support for a fully-featured language such as Halide or Python may not be as small surface area as I want. The kernels for these, such as XLA is not exactly small and verifiable as I desired.

`Jittor <https://github.com/Jittor/jittor>`_ has an interesting take on this problem. To be fair, `Jittor`'s choice won't be very interesting if you already have a fully-featured system like XLA or TVM. The choice becomes more interesting if you want to bootstrap machine-learning ops with minimal work / code.

`Jittor` defines a small set of meta-ops, these meta-ops themselves are quite straightforward to implement separately, and can be differentiated easily. These includes *reindex*, *element-wise unary / binary ops*, *reduce* and *select*. These meta-ops run separately can be costly, mostly for memory. To transform complex op into combinations of these meta-ops, it often requires to unroll values into higher dimensions and then does element-wise operations there. The unrolling to higher dimensions can be costly on memory for that reason. Some usual optimizations such as loop-fusion pretty much required for this approach to be practical.

NNC implemented the exactly approach as `Jittor` did. This enables us to describe our ops in concise and simple manner (just a few reindex and element-wise ops). Because these meta-ops (NNC called it micro-ops, or uops) are easy to implement separately, the whole implementation should only be a few thousands lines of code and will be regularly exercised.

Describe Micro Ops
------------------

To do loop-fusion or generate code from micro ops, the system should not only be able to execute micro ops, but to understand them and optimize them. That requires us to describe the micro ops with a intermediate representation.

NNC uses a very specific IR to describe these micro ops. Broadly speaking, NNC's IR has blocks and these blocks can run serially. Each block is a nested loop. Each loop contains their start index, end index, an array of loop-carried variable ids, an array of statements to run inside this loop. A nested loop always runs its inner loop before run any of its statements. Start / end index are described with index expressions, so it can be flexible.

Statements inside a loop have two types. Assignment statement assigns evaluated expression value to a tensor. Compound assignment statement does reduce with a loop-carried variable and an evaluated expression. There is no branching whatsoever with this simple intermediate representation.

`Jittor`'s implementation describes meta-ops with a DSL. This DSL requires to be parsed into its own IR. NNC's implementation describes micro-ops by constructing the IR directly with helper functions. NNC does parsing for index expressions so it is easier to describe reindex op. That is a fairly straightforward recursive-descent parser. We want to minimize parsing in general to reduce the implementation surface.

We don't apply SSA for tensors in this intermediate representation otherwise some easy operations such as zero out an tensor and accumulate would be very complex.

Parameters
----------

Unless `Jittor`, NNC's implementation doesn't want to do aggressive JIT. Supporting JIT assumes many details of the architecture where NNC runs, and that will be major departure of NNC's design philosophy. Unless we introduce something light-weight such as `QBE <https://github.com/8l/qbe>`_ or `MIR <https://github.com/vnmakarov/mir>`_, NNC will use micro-ops to generate code to be ahead-of-time compiled. That means we cannot aggressive JIT parameters. These need to be passed into the ahead-of-time generated code. Thus, we support "$param" for reindex and these parameters will be retained when generating the code.


Simplification
--------------

We want to perform at least two types of optimizations on top of the intermediate representation such that the generated code will not be horribly slow or memory hungry. Thus, we implemented loop-fusion and variable substitution for our uops.

Loop-Fusion
-----------

NNC does very aggressive loop-fusion over the intermediate representation. As we described earlier, the IR is very loop-centric. As such, we were able to match same nested loops (For terminology sake, we call nested loops that we use to match and merge `blocks`) together even if they are with different orders.

A loop matches another loop when their start index and end index matches.

When two nested loops has 1:1 mapping between their loops, it may not mean these nested loops can be matched together. For example:

.. code-block:: c

  for (i = 0; i < 10; i++) { // L1
    for (j = 2; j < 5; j++) { // L2
      float a = 0;
      for (k = 0; k < 2; k++) { // L3
        a += x[i, j, k];
      }
      y[i, j] = a;
    }
  }

.. code-block:: c

  for (i = 0; i < 10; i++) { // M1
    for (k = 0; k < 2; k++) { // M2
      float b = 0;
      for (j = 2; j < 5; j++) { // M3
        b += x[i, j, k];
      }
      z[i, k] = b;
    }
  }

These two nested loops has 1:1 mapping between L1:M1, L2:M3, L3:M2. However, they cannot be merged otherwise we will end up with wrong code:

.. code-block:: c

  for (i = 0; i < 10; i++) { // L1
    float b = 0;
    for (j = 2; j < 5; j++) { // L2
      float a = 0;
      for (k = 0; k < 2; k++) { // L3
        a += x[i, j, k];
        b += x[i, j, k]; // WRONG!
        z[i, k] = b; // WRONG!
      }
      y[i, j] = a;
    }
  }

Note that this is only problematic because we use loop-carried variables ``a`` and ``b``. If instead we simply do:

.. code-block:: c

  for (i = 0; i < 10; i++) { // L1
    for (j = 2; j < 5; j++) { // L2
      for (k = 0; k < 2; k++) { // L3
        y[i, j] += x[i, j, k];
        z[i, k] += x[i, j, k];
      }
    }
  }

This code will be correct.

This is another design choice we diverged from `Jittor` because I want to generate more idiomatic code that looks like someone hand-wrote it.

Thus, to matching two nested loops, we should treat loops with statements or loop-carried variables as pivot points. Anything before this pivot point or after can be reordered, but not the pivot point. We enforced this invariant for our nested loop matching and only merge loops if they can be reordered without violate this invariant. We devised a simple O(n^2) algorithm to do this reordering.

Besides finding the suitable loops to merge, cares need to be taken for data dependencies. For example, even if block 1 and block 3 matches, if block 3 reads any variables that was written by block 2, we cannot merge these two blocks.

Luckily, we generate block-level dependency information for loop fusion, as well as later dead-code elimination. This block-level dependency information gives us which variables we read and write in a given block. For a given variable, it can tell us in which block we read and in which block we write this variable. Armed with this information, we can determine if block 3 has data dependency with block 2, and either refuse to merge block 1 and block 3, or do what we call `merge to right`.

Normally in our loop fusion, we do `merge to left`. Thus, in above case, statements from block 3 all moved to block 1, and emptied block 3. In this way, we check whether two blocks can be merged in O(n^2) fashion. If there are data dependencies between block 3 and block 2, we cannot simply merge block 3 to block 1 (merge to left). However, we can then check whether block 2 has data dependency on block 1. If block 2 has no data dependency on block 1, we can merge block 1 to block 3 instead (hence, `merge to right`). In this case, the final blocks would look like this: [block 2, block 1 & 3].

This particular optimization technique turns out to be profitable because in our gradient passes, there is often a `reset block` that sets all values in a variable to be 0. This block often has no data dependency on other blocks and can be in between two otherwise merge-able blocks. Supporting `merge to right` mechanism moved these reset blocks to the front to merge blocks before and after this reset block.

Variable Substitution
---------------------

After loop-fusion, many intermediate tensors only write once, and used immediately in the next statement. These tensors can be removed entirely and replaced with their right-side values (for their assignment).

NNC does variable substitution conservatively. We only substitute a tensor if it is only used in one loop. And in that one loop, only if their index accessor is exactly the same. Even very conservatively, after loop-fusion, this can replace many variables as most of them now exist within one loop.

Automatic Differentiation
-------------------------

There could be two ways to implement automatic differentiation for uops.

1. Automatic differentiation applied to the opcode directly;
2. Automatic differentiation pass implemented per uops, and use these generated opcode directly.

There are pros and cons to both methods. For the first method, once we implemented, we can use that to differentiate any opcode sequence we put in. However, since opcode works with loops, the auto-diff'ed code will be auto-diff'ed within these loops, which may make our opcode more complicated to account for different principled loops usage pattern.

The second method is easier to implement, but it also limits us to first-order gradients in the beginning. This is not set in stone though. It is possible to have a closed-circle within uops (thus, using other uops to represent the gradient of a given uop). That likely involves adding a bit more uops (reindex-reduce, broadcast and ternary ops).

For our particular implementation, we went with the second method. This means we need to expand our optimization passes a tiny little bit more, to include dead-code elimination.

Our uops are designed as an easy starting point to implement other macro ops. Our macro ops for gradient pass has this particular inputs and outputs format. Given its forward pass has ``|x| -> |y|``, the gradient pass will have ``|g(y)|x|y| -> |g(x)|``. Our macro ops has a particular ``bitmask`` method to denote which exactly inputs are required, and which can be omitted. For example, the gradient pass for ``exp(x) -> y`` doesn't need to know the ``y`` value, simply ``|g(y)|x|-| -> |g(x)|`` would be sufficient.

Thus, when ``ccv_nnc_micro_combine_new``, it is helpful for us to specify exactly what's the inputs (including both inputs / outputs from forward pass and the input gradients) and what's the expected output gradients. We introduced a simple ``ccv_nnc_micro_grad`` function to represent the particular gradient for a uop.

Taking it all in, here are the steps we need to perform automatic differentiation with uops:

1. Implement ``emit_grad`` for each uop to generate their corresponding opcodes for gradient pass;
2. For gradient program, go through topological order to ``emit`` forward pass opcodes and then go through reverse topological order to ``emit_grad`` the gradient pass opcodes;
3. Annotate specified inputs in the gradient program as termination points;
4. Go from specified outputs in the gradient program backwards to mark required blocks, stops if encountered annotated inputs;
5. Perform dead-code elimination based on the liveness analysis in step 4;
6. Perform fore-mentioned other optimization passes.
