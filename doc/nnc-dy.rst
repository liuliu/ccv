NNC Dynamic Graph Execution
===========================

Frameworks such as **PyTorch** or **TensorFlow Eager** nowadays have dynamic graph support, which is a fancy word to describe when a computation is carried out while constructing the computation graph.

If **dynamic graph execution** is just about executing a command when issuing it, this is not interesting. **Dynamic graph execution** by these frameworks also supports *automatic differentiation*. A good **dynamic graph execution** framework such as **PyTorch** enables easier debugging, more intuitive coding thus quicker experimentation cycle.

That has been said, there are a few drawbacks when you support **dynamic graph execution** naively.

1. Limited optimization opportunities. With **dynamic graph execution**, the framework lacks the foresight, makes optimizations such as *common sub-expression elimination* or *data layout optimization* hard to implement;
2. Unbounded memory usage. Since a **dynamic graph execution** engine needs to be able to differentiate arbitrary variables within the framework, a Wengert list (a tape) has to be kept. In many situations, to trim that list requires user attention otherwise the memory usage will continue to grow.

To work-around 1., mixing **static graph execution** with **dynamic graph execution** is desirable. However, that imposes its own set of problems: when a **static graph** contains a **dynamic graph**, and if the **static graph** contains a loop structure, the tape for the **static graph** need to cross into the **dynamic graph** to continue work. When a **dynamic graph** contains a **static graph**, the Wengert list (the tape) of the **dynamic graph** need to not only store the tensors, but also the **static graph** as a whole.

NNC's **dynamic graph execution** design will attempt to address above problems with reasonable compromises. It borrows some good ideas from 10 years ago when I first started to implement ccv.

Naming The Variable
-------------------

Like in most frameworks, **dynamic graph execution** in NNC operates at variables. **Dynamic graph** executes command on a set of input variables, writes the result to a set of output variables. Variables can be inspected anytime with ``ccv_nnc_tensor_from_variable``. The underlying tensor may not be allocated when the variable is created. ``ccv_nnc_tensor_variable_t`` is an opaque structure and its inner work shouldn't be of an interest to users.

Tracing The Operation
---------------------

Frameworks such as **PyTorch** or **TensorFlow Eager** use the tape to record which operations are executed, and record the inputs / outputs along the way. *automatic differentiation* was implemented (its reverse mode) by walking back on the tape. This is simple to implement, and easier to support higher order gradients (by record another tape while walking back on the existing tape). This also makes optimizations on the *automatic differentiation* pass difficult because no data dependencies are specified. It is definitely possible to infer the data dependencies from the tape, and then employ optimizations or automatic parallelization. For mature framework such as **TensorFlow**, that kind of work is to reimplement some of the fundamental pieces of the software.

NNC uses its **symbolic graph** (Level-3 APIs) to trace the operation. When a command executed on a **dynamic graph**, we can figure out data dependencies with input variables (each input variable has a unique tensor symbol assigned). Even though the variables in the **dynamic graph** don't follow the *static single assignment* (SSA) rule, the underlying tensors and tensor symbols do. Thus, through the normal execution of the **dynamic graph**, we have formed a full **symbolic graph** for later computation.

Upon *automatic differentiation*, no tape is used (or, the **symbolic graph** serves as an advanced tape). We simply leverage the ahead of time *automatic differentiation* system implemented in **symbolic graph** to optimize, compile and schedule the actual computation. That means any optimization techniques we implemented on Level-2 or Level-3 APIs will be available to **dynamic graph** as well.

Optimizations Part 1
--------------------

In **PyTorch**, there is a need to ``requires_grad`` such that the framework knows which variable should be discarded to save memory. If it is not done carefully, the memory usage can grow unbounded. **Dynamic graph** here provides ``ccv_nnc_tensor_variable_free`` where when a tensor variable is freed, we will release its memory when it is safe. This method meant to hook up with object finalization methods in host languages (C++'s destructor, Objective-C's ``dealloc``, ``deinit`` in Swift, ``finalize`` in Java, ``tp_dealloc`` in Python).

Optimizations Part 2 (Not Ready)
--------------------------------

At this point, **dynamic graph** looks suspiciously like just another function dispatching mechanism. Ten years ago, when I started ccv, one of the motivation is to implement a function memorization technique, at that time, it is called *cached image processing* to workaround issues that in traditional computer vision pipeline, low level feature extraction passes often shared between different components (face detector, motion tracker etc.). In **symbolic graph**, this is trivially implemented as *common sub-expression elimination* (CSE). CSE cannot be implemented in **dynamic graph** because it cannot look ahead. However, the same memorization technique can be used to avoid duplicate computations.

Because **symbolic graph** formed from **dynamic graph execution** contains the proper data dependencies, memory reduction techniques such as automatic binomial checkpointing can be implemented with a change of cache eviction policy. If we implemented binomial checkpointing in **symbolic graph** as one optimization pass, we can also leverage that upon *automatic differentiation* in **dynamic graph**. The flexibility of sharing the same underlying infrastructure is very satisfying.

Interoperability (Not Ready)
----------------------------

There are some sticky issues with interoperability between **static graph** (the **symbolic graph** we formed by hand) with **dynamic graph**. The way they interoperate is through ``CCV_NNC_CUSTOM_FORWARD`` / ``CCV_NNC_CUSTOM_BACKWARD`` functions. When a **static graph** includes a **dynamic graph**, its tape needs to book-keeping for the **dynamic graph**. When a **dynamic graph** includes a **static graph**, it also needs to create a tape at that point for the execution. All these implies significant changes for the ``ccv_nnc_tensor_tape_t`` implementation to accommodate these new requirements.

Some Maybes
-----------

One of the major reason (or the reason) to use **dynamic graph** is its unparalleled debuggability. You can inspect tensors as you go in the code. However, this ability can be retained if the execution is separated from the **dynamic graph** forming. Your code can go a long way by forming computations and the underlying execution could be asynchronous. The synchronization happens only when you inspect these tensors to either debug, or practically, determine the control flow. This also offers limited look ahead ability to **dynamic graph** that enables more shared optimizations from Level-3 APIs. Implementing this is complicated. Synchronization point can easily turned into deadlock point, and the inter-play of **static graph** inside a **dynamic graph** inside a **static graph** could be more delicate. In a world where we modify languages to extract **static graph** (Swift for TensorFlow), the reason to have this kind of sophisticated **dynamic graph** implementation may be mooted.
