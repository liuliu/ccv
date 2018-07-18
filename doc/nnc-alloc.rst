The NNC Tensor Allocation Algorithm
===================================

Today, most neural network computations are organized as acyclic graph with each node represent a computation with a list of tensors (or multi-dimensional arrays) associated with it. Certain modifications are added to support control structures such as *if* or *do...while* loop. Given that for most implementations, they represent the acyclic graph in symbolic form (thus, no computation has been executed and no tensor has been allocated.), an comprehensive and efficient allocation algorithm is desirable and has been shown not only improve space utilization but also the speed (due to data locality).

The NNC tensor allocation algorithm is a take on the problem, it has the following properties:

1. Treat tensors as a region of memory, enable reuse part of the previously allocated memory;

2. Support loop structure, thus, if there is a while loop, this allocation algorithm will handle tensor reuse properly, without introduce any extra data transfer operation;

3. Enable efficient memory reuse when branching.

Tensor Representation
---------------------

To simplify the description, we will assume tensors are represented as a continuous memory region. A extension of this algorithm allows "alias" representation for a given tensor, thus, pointing to a sub-range of the memory region that tensor represents.

Each tensor is represented symbolically. Static single assignment form is assumed, thus, a tensor can be only assigned as node output once.

Loop Representation
-------------------

A *do...while* loop is represented as a sub-graph of existing graph. The sub-graph is represented as one node in the parent graph. All inputs for this while loop are captured as node inputs, and the outputs for this while loop are captured as node outputs. A condition is evaluated at the beginning each round of the loop, and at the end of the loop, inputs are updated with the new outputs based specifications (thus, which input is replaced by which output). Although not proved mathematically, this should enable all types of *do...while* loop construction.

The Problem Definition
----------------------

Before we get into details of the NNC tensor allocation algorithm, let's get the problem definition straight. Given a graph with above tensors and loops, the problem asks to assign ``n`` tensors to one memory region buffer such that for each node operation, the input tensors and the output tensors have non-overlap memory regions assigned. each of the ``n`` tensors also have an offset and size associated to denote from where within the buffer is the tensor memory region. We want to find an arrangement so that the size of the buffer is smallest.

It is easy to see this problem is NP-Complete. Therefore, the challenge is to find a good enough approximation to the optimal solution.

The Core Algorithm
------------------

Before stating the core algorithm, there are a few principles we want to follow, and hopefully you will find these principles make sense.

1. Deterministic and reproducible allocations;

2. Handle common neural network structures well (ResNet, DenseNet, LTSM etc.);

3. Handle ties well, and have a well-reasoned way to break the tie.

With these in mind, I will first discuss the basic structure of the algorithm, then some alternatives we may have, but why not pursuit. Afterwards, I will discuss one important extension of this algorithm to support *do...while* loop construction.

Basic Structure
---------------

The algorithm consider the above problem as the so-called interference graph, which is widely known for register allocations. In this graph, a node represents a tensor. If there is an edge between two tensors, that means these two tensors has to be allocated to non-overlap memory region.

The interference graph captured the non-overlap nature, however, the partial reuse of tensors is under specified with the interference graph. Thus, we have our first structure to represent the constraints, and now we need our second structure to represent the solution.

The second structure is an acyclic graph with edge weights for our solution. The acyclic graph with edge weights (the allocation graph) has one source node to represent the memory region buffer, a directional edge associated itself with a weight, that represent an allocation of ``x`` bytes from the edge's source node to the its destination node. There is one dummy sink node that represents the buffer when its allocation is reclaimed. In this structure, two tensors could be connected only if they don't interfere with each other, thus, the destination tensor can reuse part of the memory region from the source tensor.

Based on the second structure, an iterative construction of the solution would be to insert tensor nodes into acyclic graph with infinite reservoir of outflow from source node to the sink node until all tensor nodes are inserted. At that point, the size of the buffer would be all weights of the source node's outgoing edges. Our algorithm now reduced to the candidate tensor selection when forming this graph structure.

Candidate Selection
-------------------

A set of candidates are maintained for the graph insertion. Each candidate is a tuple of tensors (max of 3) that doesn't interfere with each other. The candidate selection algorithm like this:

1. Go through all tensors that hasn't been inserted, find the tensor that has the most number of edges in the interfere graph, if multiple tensors have the same number edges, add them all to the set of candidates;

2. For each candidate tensor in the set 1, try to find another tensor that doesn't interfere with it and has larger size than the candidate tensor. Making them a tuple and add to the set of candidates;

3. Order the set of candidates first by the maximum size of tensors in the tuple, then by the total number of edges on the interference graph of tensors in the tuple.

4. Go through the ordered set of candidates, try to find an edge on the allocation graph such that the source of the edge and the destination of the edge don't interfere with any tensors in the tuple and none of the source or the destination of the edge are the dummy source or sink nodes. If such edge is found, we find the candidate.

5. If none of the candidate can have such edge found in the allocation graph, we select the first candidate.

Insertion
---------

The selected tuple of tensors then need to be inserted into the allocation graph. The insertion is straight-forward.

1. During selection, an edge is already picked, if it is not, we make a new edge with weight as the maximum size among the tuple of tensors from dummy source to the dummy sink node.

2. Tensors in the tuple ordered by its order of access. The order must be available on the computation graph otherwise these tensors will interfere with each other.

3. The weight of previous edge decreased by the maximum size among the tuple of tensors.

4. An edge from the previous edge's source to the first tensor is inserted. The weight on the edge will be the size of the first tensor.

5. An edge from the first tensor to the second tensor is inserted. If the size of the second tensor is larger than the first tensor, the weight on the new edge will be the size of the first tensor, and another edge is inserted from the source to the second tensor with weight of the difference. Otherwise, the weight on the new edge will be the size of the second tensor.

6. Similarly, edges from the first tensor, second tensor, or the source will be inserted with respected weights.

7. Finally, edges from the all tensors to the destination will be inserted with the remaining weights.

Repeat above until all tensors are connected in the allocation graph.

Intuition
---------

Go with the tensor that has most interference is a common greedy strategy in register allocation. It removes most uncertainty that otherwise needs to branch over.

However, unlike register allocation, in tensor computation graphs, there are less cases that one tensor will span over a large chunk of computations especially in inference stage. Thus, a lot of tensors will have identical number of edges in the interference graph. For these cases, how to break the tie is crucial.

For our allocation algorithm, the allocation size is used as the the tie-breaker. If applying allocation size naively as the second sorting key, in tensor computation graphs, you may still find a lot of cases that you have tie. It is because the tensors that has similar life-span tends to be of the similar usage, thus, has similar dimensionality. For large class of neural networks, we found that by pairing up the tensor has the most interference with the tensor that has larger size (these two have to not interfere with each other), it is more likely for us to reach the trivial solution.

Loop
----

Tensor allocation with loop has to have a very specific definition of what a loop is. More broadly speaking, the types of control structure in a computation graph to support directly relevant to the allocation algorithm. The loop we specifically concerned are the ones with one conditional statement to exit the loop (traditional while-loop). For NNC tensor allocation algorithm to work, a new construct, called multi-view tensor, need to be introduced. Alternatively, the algorithm introduced here will be applicable to a specific loop that contains multiple conditional exits and phi function.

If you accept that certain data transfer is required for loop to work, the loop handling for tensor allocation algorithm is trivial. **A loop can be considered as a sub-computation graph**, and the same allocation algorithm can be applied to the sub-computation graph. When reached the end of the graph and we need to loop over again, data can then be transferred to the parameters.

For example, if you have:

::

    (while (c < 5) { // c is the loop counter
      y = Convolution(x, w, b)
    })(x <= y) // This syntax means in the next loop, x will contain the content of y, you can think of this as x = Convolution(x, w, b), but such representation is forbidden in static single assignment form.

The tensor allocation algorithm is trivial is we accept that we need to transfer data from ``y`` to ``x`` every time. This section however, we will discuss how to completely eliminate such data transfer with a novel and generic tensor allocation scheme.

Multi-view Tensor
-----------------

This is a special tensor that with nested structure. For a leaf multi-view tensor, it can point to multiple memory regions based on the loop counter. Particularly, a multi-view tensor can be configured with a repeat length. Its pointer will be updated prior to the actual computation each round the the correct memory region: ``ptr = ptrs[loop_counter % repeat_length]``. There are some complications such as the support for two types of multi-view tensors. Type I will be the one described above. Type II will have a special memory region that only used when ``loop_counter == 0``.

A multi-view tensor can not only points to memory regions, but to a set of other multi-view tensors, following the same semantics, thus, the nested structure.

Loop with Efficient Tensor Allocation
-------------------------------------

Above are all the constructs we need to implement efficient tensor allocation algorithm (the efficient here means no data transfer required).

For each parameter, we first identify whether co-allocating them to the same memory region is sufficient. In some cases, they are, thus, we can simply do that and then apply our tensor allocation algorithm to the sub-computation graph.

However, in some cases (like the superficial case we made above), it is not possible. For these, we need to *unroll* the loop.

For example, unrolled above loop will be:

::

    while (a < 5) {
      z = Convolution(x, w, b)
      b = a + 1
      if (b) exit
      y = Convolution(z, w, b)
      c = b + 1
    }(x <= y, a <= c)

One extra conditional exit added to make the loop syntactically equivalent to the one we had before.

When a loop unrolled as above, for the particular case, we can see that now ``y`` can be co-allocated with ``x`` (They are not interfere with each other).

It can be proved that any loop can be unrolled into a form that the parameters can be co-allocated. The exercise will be left to readers on how to use this to tackle something like ``x[c] = Convolution(x[c - 4], w, b)`` which requires to access variable from several loops before.

Once a loop can co-allocate all its parameters after unrolling, we can apply the tensor allocation algorithm on the unrolled computation graph.

The allocation on the unrolled computation graph then can be used to create the multi-view tensors. Now, the repeat length on the multi-view tensors correspond to how many times we unrolled the loop. Each memory region will be pointing to corresponding tensor on the unrolled computation graph as well.

Sub-Computation Graph
---------------------

Sub-computation graph's tensor allocation generated number of buffers and each buffer size. These will be used as regular tensor in the parent computation graph. The whole allocation algorithm then becomes recursive.

Conclusion
----------

I believe the above algorithm is the first to address the tensor allocation problem with partial memory reuse and loop efficiency in mind. This algorithm is also presented as an extensible framework that can be considered in the future to support more control structures.
