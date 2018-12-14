NNC Static Schedule A Graph
===========================

A concrete graph runs in topological order sequentially when you invoke :cpp:func:`ccv_nnc_graph_run`. Thus, all dependencies executed before the subsequent command got executed. This default behavior doesn't leverage massive parallelism built in today's CPU / GPU environment, leaves too much computation power on the table.

NNC supports to static schedule a graph such that all the parallelism are utilized. That's been said, **static scheduling** shouldn't be confused with **automatic parallelization**. The later term suggests to turn a non-parallelizable graph into a parallelizable ones by adding additional data transfers or graph transformations. **Static scheduling** considers dependencies of a command, and try to schedule independent commands onto different streams.

Stream
------

One of the core abstraction for the execution model is the **stream**. This models closely to `CUDA stream <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM>`__. Two commands scheduled on the same stream guaranteed to be executed sequentially. Different streams can collaborate with **signals**. The **signal** models closely to `CUDA event <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT>`__, with the same limitation that if a signal is not scheduled on a stream, wait on it will be a no-op.

That all sounds very much like a thin wrapper around CUDA runtime. The key difference is that you can also schedule a graph onto a stream, and they will be executed serially with other graphs scheduled on the same stream.

Static Schedule
---------------

Conceptually, the **static scheduling** algorithm for a graph is trivial. A new stream can be spawned whenever there is a split. The new stream can be either recycled from a stream that is terminated or a newly created. However, there are some factors at play. For example, if there are repeated branch-and-merge, you can alternate streams for your execution. Consider ``N1 -> (N2, N3) -> N4``, you can assign ``N1``, ``N2`` to stream 0, and ``N3``, ``N4`` to stream 1. Alternatively, you can assign ``N1``, ``N2``, ``N4`` to stream 0 and ``N3`` to stream 1. Both are equivalent if stream only maintains the execution order. In NNC's implementation however, stream also maintains execution context and workspace memory for BLAS, CuDNN etc. We prefer the second scheduling.

The **static scheduling** algorithm implemented in NNC went through a few iterations. The current iteration first do a reverse traversal of the graph, assign each node a rank. The rank is the length of the longest chain follows the current node. When traverse the graph, if the current node hasn't assigned stream yet, we will find a recyclable stream (a stream that is deterministically terminated before the current node), or create a new stream. From the current node, we will find its highest ranked unassigned following node, assign the new stream to it. We use this node as the new node, repeat steps until no unassigned following node can be found. If two nodes have the same rank, we break the tie by checking whether in this given stream, we already encountered the same command before (thus, sharing workspace memory and execution context is possible).

As part of the static scheduling work, a node can be associated with multiple streams. This is useful for commands that need to communicate across devices because each stream can only be associated with one device.

``while`` and ``case..of``
--------------------------

A concrete graph in NNC can contain branches and loops. A naive implemenation such as CUDA streams / events cannot handle these. A lightweight coroutine implementation based on ``<ucontext.h>`` enables us to implement ``while`` and ``case..of`` properly while still leveraging CUDA streams / events construction.

For loops and branches to work, the expressions should be evaluated each loop or before branching to determine whether we continue looping or where to branch to. Some tensor reconfigurations need to happen as well for each loop or after branching. Since streams are asynchronous, it is not obvious how to do it efficiently and correctly with only CUDA streams / events.

We effectively implemented some coroutine helpers such that the current task can yield while waiting for CUDA stream to finish. This is transparent and if other nodes are not blocked by the loop or branching, can still continue be scheduled until everything is blocked by a loop or branching. At that point, we will create a new thread to wait, to maintain the illusion that the interface is asynchronous. The new thread is only for scheduling, and no matter how many streams we allocate, we only have this one thread to schedule.
