NNC Static Schedule A Graph
===========================

A concrete graph runs in topological order sequentially when you invoke ``ccv_nnc_graph_run``. Thus, all dependencies executed before the subsequent command got executed. This default behavior doesn't leverage massive parallelism built in today's CPU / GPU environment, leaves too much computation power on the table.

NNC supports to static schedule a graph such that all the parallelism are utilized. That's been said, **static scheduling** shouldn't be confused with **automatic parallelization**. The later term suggests to turn a non-parallelizable graph into a parallelizable ones by adding additional data transfers or graph transformations. **Static scheduling** considers dependencies of a command, and try to schedule independent commands onto different streams.

Stream
------

One of the core abstraction for the execution model is the **stream**. This models closely to `CUDA stream <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM>`__. Two commands scheduled on the same stream guaranteed to be executed sequentially. Different streams can collaborate with **signals**. The **signal** models closely to `CUDA event <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT>`__, with the same limitation that if a signal is not scheduled on a stream, wait on it will be a no-op.

That all sounds very much like a thin wrapper around CUDA runtime. The key difference is that you can also schedule a graph onto a stream, and they will be executed serially with other graphs scheduled on the same stream.

Static Schedule
---------------

The **static schedule** for a graph is really simple. Each node in the graph will be assigned to a stream depending on its incoming nodes. It will inherit the last incoming node's stream if that is not inherited yet by other nodes. If it is, a new stream is created (or if another stream is deterministically idle for this node, thus, is last used by a node that is an ancestor of the current node) and assigned to this node. Proper signals are created to make sure the current node will only be executed when every incoming node finishes.

``while`` and ``case..of``
--------------------------

A concrete graph in NNC can contain branches and loops. A naive implemenation such as CUDA streams / events cannot handle these. A lightweight coroutine implementation based on ``<ucontext.h>`` enables us to implement ``while`` and ``case..of`` properly while still leveraging CUDA streams / events construction.

For loops and branches to work, the expressions should be evaluated each loop or before branching to determine whether we continue looping or where to branch to. Some tensor reconfigurations need to happen as well for each loop or after branching. Since streams are asynchronous, it is not obvious how to do it efficiently and correctly with only CUDA streams / events.

We effectively implemented some coroutine helpers such that the current task can yield while waiting for CUDA stream to finish. This is transparent and if other nodes are not blocked by the loop or branching, can still continue be scheduled until everything is blocked by a loop or branching. At that point, we will create a new thread to wait, to maintain the illusion that the interface is asynchronous. The new thread is only for scheduling, and no matter how many streams we allocate, we only have this one thread to schedule.
