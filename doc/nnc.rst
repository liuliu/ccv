NNC: Neural Network Collection
==============================

What's NNC?
-----------

NNC is the natural progression against ``ccv_convnet``, which is a couple of years old now. ``ccv_convnet``'s monolithic, single path neural layer design didn't really feel right with more advanced network architectures.

NNC took some good ideas from more recent neural network frameworks and did a long re-think on how to achieve both efficiency and expressiveness. The design itself is layered. At the highest layer, you have ordinary neural network primitives that reflect real-world usage such as Inception module, LSTM, RNN et al. At the lowest layer, depending on the infrastructure, it maps to allocated tensors on GPU, computations backed by CuDNN, and computation graphs driven with CUDA streams (or you can exchange that with CPU, Metal, and libdispatch). For the abstractions in between, there are trade-offs and constraints to accommodating both the library design and usage.

In a few sentences, here is how the layered design works.

NNC starts with tensors, commands and streams, which closely map to low level computation primitives. On top of that, a concrete computation graph can be constructed and executed directly. Above that, a symbolic graph can express everything about the concrete computation graph, without actual tensor and stream allocations. A dynamic graph can contain both symbolic graph representation and the concrete computation graph, thus, carries out the computation immediately while retain the ability to do series of processes on top of the symbolic graph representation. Familiar primitives such as LSTM or RNN then were built on top of either the symbolic graph or the dynamic graph constructs.

There are roughly **5 layers** built on top of each other.

1. Tensors, Commands and Streams
--------------------------------

Tensors are multi-dimensional arrays at its basic level.

Commands are ops (operations) in other framework's terminology.

Streams are the synchronization mechanism. Each command instance executed serially on a given stream. Different command instances on different streams will be scheduled in parallel if the underlying infrastructure permits.

A command is identified by its ``cmd`` identifier. It processes a set of input tensors, and write output to a set of output tensors. There is no limits on how many input tensors it can accept or how many output tensors it can write to.

A command can only have one set of attributes (recognized by NNC) specified. These attributes (such as whether this can be an *inplace* operation) help on symbolic processes. If you find that you need to implement the same command but these attributes cannot be hold, you need to rename the command to avoid invalid symbolic processes.

One command, however, can be backed by several **backend** implementations. Command backend implementors, besides the ones who implement ``*_REF`` free to only support specific cases of the input tensors (for example, a particular tensor layout, or a specific tensor size (3x3?), or half precision numbers). But once a backend accepts the input, it follows exactly the command attributes specified above (for example, any backend that implements a *inplace* command, will allow any parts of its input to be overwritten by this command at time while this command is executing without affecting the correctness of the output).

At runtime, a command will select the appropriate backend based on the input type and execution time.

2. Computation Graph
--------------------

**Computation graph** expresses how the computation carries out. The output tensors can be used as input for the next command, so on and so forth. That is where *TensorFlow* got its name from. At this layer, **computation graph** knows the execution orders (data dependencies) between each command instances, and will schedule them on proper streams to ensure these execution orders are respected. Tensors themselves are not associated with the execution order at this point.

A **computation graph** can contain a sub-graph, which is a **computation graph** itself. It is executed as a single command instance by the parent **computation graph**. As of now, a *``while`` type sub-graph* (for looping) and a *``case..of`` type sub-graph* (for branching) are supported.

A **computation graph** can be auto-tuned to find the best backend implementations that minimize the total execution time. There may be future optimizations to allow modifying the graph itself to do more aggressive tuning (such as including tensor conversions to trade between slower implementation and conversion + faster implementation).

In short, once you have a **computation graph**, the computation can be carried out naturally because there is no extra assumptions about execution environment and no more parameters or allocations need to be specified.

3. Symbolic Graph
-----------------

**Symbolic graph** expresses commands, the associated tensors and their execution orders (dependencies). This may sound very similar to the **computation graph** above, but there are several important differences:

1. there is no concept of *stream* at this point, because the **symbolic graph** doesn't carry out the actual computation, and *stream* can be determined purely by the execution order;

2. there is no tensor allocation. **Symbolic graph** uses the tensor metadata (layout, precision, dimensions, even which GPU it is associated with), but no actual allocation took place until it is compiled to a **computation graph**;

3. There is no 1:1 mapping guarantee about the commands in the **symbolic graph** with the command instances in the **computation graph**.

In fact, **symbolic graph** doesn't take tensors. It takes tensor symbols. The tensor symbol usage within the symbolic graph follows strict *static single assignment (SSA)* rule. It can only be used as a command instance's output once. This is important because by following *SSA*, potential data races are completely eliminated. More over, certain processes and the actual tensor allocation algorithm are much easier to implement with this assumption. With *SSA* rule, the execution orders (dependencies) can be generated trivially.

It may feel like the tensor metadata is over-specified. For example, why precision or layout, or which GPU it resides is relevant? Because tensor symbols have many to 1 mapping with the actual tensors. Specifications on the tensor symbol avoid processes on the **symbolic graph** resulting a tensor symbol that needs to be backed with conversions. Any conversions on the **symbolic graph** has to be explicit command instances.

Having that in mind, however, you can take an *alias* of a tensor symbol, which is a sliced / reshaped tensor symbol from the original. It allows several operations to be zero effort on the actual **computation graph**. The *alias* itself still have to follow the same *SSA* rule, which means all the *aliases* and the original tensor symbol can only be written once (if two *aliases* as outputs point to non-overlapping parts of the original tensor, the written-once rule is not violated).

Processes can be carried out on the **symbolic graph** ranging from *automatic differentiation*, to *common sub-expression elimination* (CSE), or *operator fusion* (finer-grained set of commands be replaced by a combined command implementation).

When the actual computation is needed. A **symbolic graph** can be compiled to a **computation graph**. The compilation process can involve optimizations that previously already possible on the given **computation graph** (such as CSE). More importantly, this step performs additional optimization passes that will violate the *SSA* rule above. Currently, it will perform following processes that are not available as pure optimization passes:

1. In-place safe command instance will operate on the same tensor symbol inputs / outputs whenever possible (for example, ``1.23 * x => y`` will be re-written to ``1.23 * x => x`` if no other places use ``x``);

2. Tensor allocation based on the liveness analysis for the tensor symbols. This step will generate the many to 1 mapping between tensor symbols with the actual tensors;

3. Emit implicit commands for tensor initialization. Certain tensor symbols need to be initialized before use (zero init for now), which is impossible to know when until tensor allocation was taken place. This is one reason why there is no 1:1 mapping between **symbolic graph** and **computation graph**.

All above steps are carried out recursively for its *``while`` / ``case..of`` type sub-graphs* too.

4. Dynamic Graph
----------------

**Dynamic graph** operates on concrete tensor instances. It took input tensors, executed a command on them, and took the outputs. From this perspective, it is very similar to the **computation graph**. The conceptual difference, is that the **computation graph** carries out execution from a specification, while **dynamic graph** forms a specification from the actual execution.

Thus, **dynamic graph** will construct a **symbolic graph** along its execution. It enables the **dynamic graph** to perform the same kind of sophisticated optimization passes and analysis once needed (such as *automatic differentiation*)

More over, **dynamic graph** implements a simple memorization mechanism. The tensors it uses will carry a hash, as well as a specific command. The output tensors can be retrieved from the cache by the generated hash if it is possible, to avoid repetitive computations.

5. Common Neural Network Primitives
-----------------------------------

A set of **common neural network primitives** for modeling as well as parameter updates is provided. The API looks very much like **Sonnet** or **Keras**. **Common neural network primitives** implemented these interfaces at a common language layer (C language). Thus, variety of host languages to implement a simple shim layer on top to enable these high-level APIs.

Supplementary Materials
-----------------------

Toll-Free Bridging
~~~~~~~~~~~~~~~~~~

*Toll-free bridging* here means that a ``ccv_dense_matrix_t`` struct, without any conversions at all, can be cast to a ``ccv_nnc_tensor_t`` struct and then used with nnc directly. The byte pattern is specifically arranged such that a 3 dimensional ``ccv_nnc_tensor_t`` can be cast back to ``ccv_dense_matrix_t`` vice versa. This allows seamless integration with the rest of image process primitives provided by ccv.

Automatic Differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~

*Automatic differentiation* supported by nnc is its reverse mode. The implementation is simple enough because we enforced *SSA* throughout the **symbolic graph**.

Each command need to implement its forward function, as well as its backward function. The backward function takes the input / output of the its forward function, as well as the gradients (matching the output tensors) as its input. It outputs the gradients with respect to the input (matching the input tensors of the forward function).

When doing *automatic differentiation*, from its **symbolic graph**, a backward command matching each forward command is created. The execution order (dependencies) is exactly reverse. *SSA* guarantees each tensor symbol is written once, that means the gradient w.r.t. that symbol needs to only be summed once as well.

*alias* introduced some complexities to the implementation. Namely, because an alias can be used as input for follow-up commands, its reverse suggests different gradients w.r.t. different *aliases* required to be summed at certain point. That means these gradients need to be potentially zero init to avoid generating garbage results. This is done by inserting zero init tensor symbol property, which indicated an implicit zero init command will be injected at **symbolic graph** compilation time.

The specific implementation also means taking second order derivative isn't possible with nnc at this point. It will be possible however in the future once the backward function can be specified by a set of forward functions and then we can do command substitution on the **symbolic graph**.

``while`` Type Sub-Graph
~~~~~~~~~~~~~~~~~~~~~~~~

The *``while`` type sub-graph* is a special type of a **symbolic graph** or a **computation graph**. This is because it expresses a generic loop structure with custom evaluation function supplied.

The loop execution within a *``while`` type sub-graph* looks like this:

1. The sub-graph starts the execution from a set of source command instances;
2. It proceeds either serially or in parallel until all evaluation command instances executed. The subsequent command instances are on hold;
3. The evaluation function is called, and depends on the result, the execution within the sub-graph will either abort (break), or continue, until all the destination command instances executed and reached;
4. Once all destination command instances executed and reached, we will start from step 1. again.

For *``while`` type symbolic sub-graph*, the obvious question would be how *SSA* rule plays out in the loop structure. We allow in the sub-graph to specify certain output tensor symbols carry over to the input tensor symbols in the next round, practically made these input tensor symbols parameters. The *compilation* step will handle this properly and allocate the input tensors at the same memory locations as the output tensors (there are ``ccv_nnc_tensor_multiview_t`` workaround if the condition cannot be satisfied).

When doing *automatic differentiation*, a ``ccv_nnc_tensor_tape_t`` need to be provided for the *``while`` type sub-graph* to record the outputs properly.

``case..of`` Type Sub-Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The *``case..of`` type sub-graph* is another special type of a **symbolic graph** or a **computation graph**. It expresses a generic branch structure with custom evaluation function supplied.

The *``case..of`` type sub-graph* contains several separate sub-graphs identified by indexes from 0 to n:

1. The evaluation function is called, if the result is >= 0, a sub-graph is selected for execution, otherwise, jump to step 3.;
2. The selected sub-graph executed from beginning to end;
3. If the result is < 0, no sub-graph executed.

For *``case..of`` type symbolic sub-graph*, if a tensor symbol is *written-once*, how to proceed if all sub-graphs skipped (in typical case, if a sub-graph executed, presumably, the tensor you want will be written by a command in that sub-graph)? We allow you to specify for these output tensor symbols, which symbol from the input can be supplied as *replacement*. The *compilation* step will ensure a ``ccv_nnc_tensor_multiview_t`` is created to handle these cases.

When doing *automatic differentiation*, a ``ccv_nnc_tensor_tape_t`` need to be provided for the *``case..of`` type sub-graph* to record the outputs properly.

Limits and Constraints
~~~~~~~~~~~~~~~~~~~~~~

1. Tensor itself supports up to 8 dimensions. This is defined in ``CCV_NNC_MAX_DIM_ALLOC``.

2. Tensor's dimension can only reach to up ``INT_MAX``. That may be a limiting factor for some of the tensors if they need more than 8GiB (32-bit floating point assumed) on one dimension.

3. The limit on number of inputs and output tensors is ``INT_MAX``. To perform *automatic differentiation* properly, this number drops to ``floor(INT_MAX / 3)``. However, for more than 64 parameters, there are internal heap allocation required, which makes previously deterministic execution none-deterministic (it may take arbitrarily long depending on the ``malloc`` you use).

4. The allocated tensor size can go up to ``min(UINT64_MAX, SIZE_MAX)``.

5. A computation can only depend on no more than ``2^16`` other computations. This is determined by a core macro ``CCV_NNC_GRAPH_VISIT``.

6. The sub-graph can go as deep as ``2^(31 - 4)``, otherwise the outer-most while count tensor cannot be referenced by the inner-most sub-graph.

7. The maximum number of GPU devices per machine or NUMA nodes per machine is 4095. This is defined in ``CCV_COMPUTE_DEVICE_ANY``.
