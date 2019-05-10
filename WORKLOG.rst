2019-05-09
----------
I don't know why my graph traversal code doesn't properly address "don't visit nodes that not contribute to the destination". Initially, how the graph was driven done with flood fill.It is all fine until I want to get more serious.

The compounding problem is that I want to, eventually, making the concrete graph computation as fast as do the computation directly (even if the tensors are as simple as scalar (0-dimension tensor)). That means have a more compact representation of the graph, better interpreter (right, you can think the ``ccv_nnc_graph_run`` as "interpreting"), and doesn't do topsort every time.

Unfortunately, that's the absurd world I am in now. Right now, if a graph is not ``ccv_nnc_graph_static_schedule``, running it requires to traverse the graph 4 times: 1. Collect statistics about how many incoming edges for each node; 2. Collect exactly which are the incoming edges; 3. Reverse traverse from destinations to the sources, marking node that can be reached this way; 4. The final traversal, only call node that is marked in step 3. All these is because I don't want the graph representation including both outgoing nodes and incoming nodes. Including incoming nodes is obvious but a struggle for me because I don't want to maintain two sources of truth about the graph structure. Then, I end up with this 4-pass graph traversal.

There are ways to optimize this though. First, let's be honest, flood fill won't give me efficient interpreter. I need the topsorted result available already to be efficient. It seems more and more likely, that "cache" topsorted result thing could be another layer "cache" the opcode for graph interpreter. Very interesting.

After 3 months with the new machine built (4xRTX2080Ti), and fixed the AMD freeze issue, I finally can work on the fp16 support again. Long time indeed!


2019-05-06
----------
Designing API is hard.

This can be seen by the expansion of ``ccv_nnc_symbolic_graph_minimize`` parameters. Previously, the parameters are a lot, but makes sense. The parameters you try to optimize, the minimizer, the losses, and the sources / destinations for the graph. The output from this function is the list of gradients, updated parameters. However, it is not flexible enough for the case where I need to compute the gradients against input, but not necessarily create ops to "optimize" inputs. This is expected to implement outgrad support for ccv_cnnp_model in multi-stage mode. Otherwise, we need to essentially reimplement the minimize function (i.e., first compute gradients, and then insert minimizers). For this case, on the API side, I added additional parameters called inputs, which is the tensors we want to compute gradients, but not optimize for (not free parameters). The side effect, as you can see now, is a more complex API.


2019-05-05
----------
Debuggability in framework is a big issue. There are a few things I should do earlier but haven't that bites me now. One example is how we handle symbolic graph compilation. When it works, it is pretty cool, but when it doesn't, there are some hard time to look through what's going on. Example: 1. When a tensor is used before initialization, we didn't provide init with some harder value (nan). This is simple to solve though, as long as we do that initialization when create tensor arena; 2. Wish this is as that simple, tensor areas are reused, thus, it could be uninitialized but with some value in it already, this may be solved if we force to init some values (using ``CMD_SET_FORWARD``), but that has consequences such as violate SSA during the compilation; 3. That leaves me to conclude that I really should do the simple allocation implementation much earlier, which is the debug mode for our tensor reuse logic, as well can be coupled with default initialization mode. In this way, each new tensor will be allocated from the heap directly without reuse, and set default initialization value. This helps to check reuse logic (however, less useful since our reuse logic is really robust nowadays), but also, makes the uninitialized tensor case much easier to surface. This mode however, is not simple to implement now, because additional tensor transfer logic required for while loop / case of where we relies on tensor reuse. Especially for while loop, we don't really do any data transfer at all (this is also understandable because if we do fresh allocation in while loop, memory will grow unbounded).

More over, debuggability concerns grow beyond just for this framework. It is now a concern for any frameworks for computation graphs. Here is my take: you pretty much need have a textual representation for any computation graph before debuggability comes into play. In this way, you can treat computation graph as imperative programming language, thus, step over, step into, rewind comes naturally. Inspecting variables in a scope, visualize it, inject some new values can also be beneficial. This is almost pointing to implement some form of Debug Adapter Protocol in VSCode and beyond. TensorBoard, on the other hand, doesn't make me feel is an adequate debugger, visualization, sure. Debugger requires two way communication which is not well-defined for TensorBoard with TF driver.


2019-05-03
----------
Have a rough implementation where for high level API such as ccv_cnnp_model, we can do forward pass, and then do backward pass separately.

This is helpful because we can customize losses (thinking about RL), accumulate gradients (useful for detection), and even use ccv_cnnp_model as a imperative part of a bigger model (i.e. using dynamic_graph to drive the computation, and use well-made ccv_cnnp_model for parts of it). I am very happy with where the abstraction goes.

However, the issue rises when I need to support outgrad in ccv_cnnp_model_backward. During backward, ingrad is provided (gradients corresponding to outputs). outgrad is not required, but if you provided, the gradients can flow over all the way to the input. In this way, ccv_cnnp_model can truly be part of a bigger model. This imposes a challenge though. To get the gradient, ccv_nnc_symbolic_graph_backward need to know which tensor we need to compute gradient against. The inputs are not provided in ccv_cnnp_model_evaluate / ccv_cnnp_model_fit's jitting. Thus, there is no such tensor symbol we can bind to as outgrad. This is relatively easy to resolve. We simply need to add these to the list of tensors requires gradients.

nnc's implementation optimizes both memory usage and computation aggressively. Thus, allocating additional memory and computation doesn't settle well. Alternatively, I can re-jit if outgrad provided, adding even more modes. Now, imagining we'd like to take some memory penalty for greater goods, thus, for multistage mode, we will generate a graph that computes the input gradient as well, is there a way for us to say, skip the computation penalty at least? Even this, unfortunately, doesn't seem obviously to me. For most ops, it is safe to pass that gradient in as 0, and it can skip. But for 1, it is not universal, we simply haven't enforced this and don't know if the outgrad is aggregated. Second, we cannot actually pass 0 after compiling symbolic graph to concrete one. The reason is because tensor can be unwrapped, therefore, we cannot simply assign a tensor to 0. Alternatively, safer option would be make tensor.data.u8 == 0, this is not ideal because either during command execution, we need to copy all tensor parameters out and make these tensors 0 if its underlying data.u8 is 0. Otherwise, in every single op implementation, we need to check both the tensor and its data.u8 for emptiness.

Probably complicating the interface more is a better solution (adding a 3rd parameter along requires_grad and is_test).


2019-05-01
----------
Start a worklog entry. Some of the thought process I had working on this project cannot be documented in the commit history. A worklog is a better place to write these down.
