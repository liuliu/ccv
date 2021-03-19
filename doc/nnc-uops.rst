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

We want to performance at least two types of optimizations on top of the intermediate representation such that the generated code will not be horribly slow or memory hungry. Thus, we implemented loop-fusion and variable substitution for our uops.

Loop-Fusion
-----------

NNC does very aggressive loop-fusion over the intermediate representation. As we described earlier, the IR is very loop-centric. As such, we were able to match same nested loops (blocks) together even if they are with different orders.

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

Thus, to matching two nested loops, we should treat loops with statements or loop-carried variables as pivot points. Anything before this pivot point or after can be reordered, but not the pivot point. We enforced this invariant for our nested loop matching and only merge loops if they can be reordered without violate this invariant. We devised a simle O(n^2) algorithm to do this reordering.

Variable Substitution
---------------------

After loop-fusion, many intermediate tensors only write once, and used immediately in the next statement. These tensors can be removed entirely and replaced with their right-side values (for their assignment).

NNC does variable substitution conservatively. We only substitute a tensor if it is only used in one loop. And in that one loop, only if their index accessor is exactly the same. Even very conservatively, after loop-fusion, this can replace many variables as most of them now exist within one loop.
