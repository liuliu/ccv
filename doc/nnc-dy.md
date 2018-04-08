NNC Dynamic Graph Execution
===========================

Frameworks such as **PyTorch** or **TensorFlow Eager** nowadays have dynamic graph support, which is a fancy word to describe when a computation is carried out while constructing the computation graph.

If *dynamic graph execution* is just about executing a command when issuing it, this is not interesting. *Dynamic graph execution* by these frameworks also supports automatic differentiation. A good *dynamic graph execution* framework such as **PyTorch** enables easier debugging, more intuitive coding thus quicker experimentation cycle.

That has been said, there are a few drawbacks when you support *dynamic graph execution* naively.

 1. Limited optimization opportunities. With *dynamic graph execution*, the framework lacks the foresight, makes optimizations such as *common sub-expression elimination* or *data layout optimization* hard to implement;
 2. Unbounded memory usage. Since a *dynamic graph execution* engine needs to be able to differentiate arbitrary variables within the framework, a Wengert list (a tape) has to be kept. In many situations, to trim that list requires user attention otherwise the memory usage will continue to grow.

To work-around 1., mixing *static graph execution* with *dynamic graph execution* is desirable. However, that imposes its own set of problems: when a *static graph* contains a *dynamic graph*, and if the *static graph* contains a loop structure, the tape for the *static graph* need to cross into the *dynamic graph* to continue work. When a *dynamic graph* contains a *static graph*, the Wengert list (the tape) of the *dynamic graph* need to not only store the tensors, but also the *static graph* as a whole.

NNC's *dynamic graph execution* design will attempt to address above problems with reasonable compromises. It borrows some good ideas from 10 years ago when I first started to implement ccv.

Naming The Variable
-------------------

Like in most frameworks, *dynamic graph execution* in NNC operates at *variables*.
