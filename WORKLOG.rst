2020-01-06
----------
Get myself more familiar with LLVM. I am surprised the design separation of Function v.s. Basic Block v.s. Instruction, and then fact that Basic Block itself is not recursive. The loop structure, in particular, loop-closed SSA form is not something intrinsic to Basic Blocks. If the design is more functional, there shouldn't be a separation of Basic Block and function, while Basic Block would be enough to express loop structure. What I do learnt though, is how easy LLVM is to manipulate BB / Func / Inst through CGF / CGM. Comparing to how hard to create a phi node inside nnc (not explicitly, through the mapping when add case..of sub-graph), or assigning loop carry-overs, LLVM is so much easy to remove a BB, create a BB, and hook up one BB with another. Not to mention to iterate over Inst and BB, it is something builtin while there is still no easy way to iterate over nodes and manipulating them at the same time inside nnc.

While it is very inspirational, I will punt more work in defining a better symbolic graph interface. After all, Relay and MIIR all try to do better job at expressing computation graph, I can learn one or two from their experimentation first.


2019-08-22
----------
Implementing named models and proper tensor init seems not so easy. Particularly, for complex training setup, such as: having new model share some weights with simpler models (for example, seed ResNet101 with ResNet50 parameters), or fix the training on certain weights, and continue on the others. The former one requires us to keep some consistency between different models, the second requires us to mark the model somehow while adding trainables.

Thus, we should be able to name a given model (or layer). The trainables weights will be fixed to that name, thus, adding new layers won't impact the old weights, and these can be loaded successfully. To accomplish this, I added the new ``ccv_nnc_tensor_read`` and ``ccv_nnc_tensor_write`` methods to keep tensors. This also marked a departure for how persistence should be done. Rather than ad-hoc with SQLite, it will all be marked, now with tensor and names.

Persistence worth a rethink in general, it starts by just names and tensors. I will remove persisting symbolic graph support. Instead, will enable persisting graph and tensor arena.


2019-08-12
----------
Revamp the persistence for networks. Comparing to other solutions such as protobuf, I would rather just use SQLite. But it will be different from previously I do this. Previously, when I use SQLite as persistence, it is not composable. Thus, different algorithm will use SQLite differently, there is not shared schema. The revamped way will have all tensors saved into the "tensors" table, and everything else reference to it by name. For example, for CNNP, there is no persistence other than "tensors", the model itself is not persisted at all. However, for tensor arena / concrete graph, we will persist both the tensor allocation, tensors and the graph. I don't think we want to persist symbolic graph any more. It is likely I will delete that code later.

In this way, one can query the SQLite and navigate the database as if it is just a "workspace" file (in Matlab sense). These data can be easily ported to pandas or other places because you only need to write a tensor loader once, everything else just a naming convention afterwards.


2019-07-15
----------
Moved to SF. It seems Nesterov is important for ResNet-50. Moved to Nesterov, the final result is much more comprehensible.

I am currently working on a concept called LSSC (Linear Scaling Spatial Compression). The insight is simple. Unlike weights, activations have more spatial redundancy. These activations get used during back propagation. It is conceivable if we can have some way to compress the activation, and during back propagation, decompress these activation back, we can save some amount of memory. Given these kind of compression ratio (Bitmap to JPEG etc.) are surprisingly high, we can expect a big reduction in memory usage if the compression scheme used during training process. Currently, I am prototyping this, the big unknown is the quality of the compression (I am pretty confident about this, because the decompressed activations only used during back propagation anyway), and speed (I am more worried about this, because it is unclear how to implement this efficiently on GPU).

Stay tuned.


2019-05-31
----------
Weight decay as the regularization has to be one of the most non-obvious thing in my implementation. The theoretical background for weight decay is to minimize weights, thus, loss^{wd} = loss + c * sum{||w||^2}. Thus, the selection of c would be important. Somehow in the CIFAR-10 implementation, I choose a very aggressive c. In implementing imageNet, that bites me. Too aggressive c makes the weight too heavily regularized, therefore, cannot converge on larger dataset such as imageNet unfortunately.

I think this is time for me to implement RMSProp or ADAM for faster iteration. Hyperparameters for SGD are too much and not universal.


2019-05-28
----------
Debugging memory related issues is hard. I've been battling against a bug when loading trained ResNet model into memory and continue the training, it will mysteriously halt at certain GPU operations. Debugging GPU related issues is always difficult. It often involves first identifying exactly which CUDA API call failed (that is why you see the codebase littered with ``CUDA_ENFORCE``, ``CUBLAS_ENFORCE``, ``CUDNN_ENFORCE``, ``NCCL_ENFORCE`` to make sure we fail early).

This time it is relatively easy. The fault command is the softmax fused cross entropy loss backward op. However, because it only happens when I enabled parallel mode, I was confident this is somewhat related to I haven't ``cudaSetDevice`` properly in some methods. Furthermore, if I moved weights loading after the data prefetching, it seems all worked. Thus, I've been trying to identify which function call happens on which GPU device for extended time with no progress made. A lot of assertions added but no bug was caught.

Then when searching for 700 error ``cudaErrorIllegalAddress``, I came across `cuda-memcheck`. It is a little nice tool very much like `valgrind`, it is plug-and-play. With `cuda-memcheck`, within minutes, I identified the illegal memory access (related to how we handle fp16 the same as fp32 when copy value over). It also helped me to identify a double-free bug as well.

It seems reasonable to say that I need to include `cuda-memcheck` in the buildbot script to help protect against memory issues from GPU side in the future. Definitely a good learning experience today.


2019-05-22
----------
Besides lacking of debugger.

Without debugger, currently, to run cnnp programs, there are several issues.

 1. Ad-hoc looking at GPU tensors and getting statistics are hard (this is partially addressed by having GPU tensor's first 3 values in the VERBOSE output now, but we don't have statistics);
 2. There are issues with nan if the learn rate is too large (of course!). Since GPU is running asynchronously, it poses challenges to scream at the point when we hit nan, and give enough trace to look back to see whether it is because we have some faulty ops, learn rate too high, initial gradient is too much (not an issue until we implement non-1 gradient propagation, this is useful to increase / decrease scales for fp16);
 3. Extract loss / accuracy from the data is far from obvious. I need to manually transfer the data to the CPU, and write some code to collect the accuracy;

There are several ways to do this. I can have a stats function that given a list of tensors, generate statistics (min, max, average, std), and then transfer these stats back to CPU for inspection. This requires to modify the graph, but could be relatively easy. To gather accuracy would actually be harder. For one, we use one hot, and later we are going to use mixup, which means the ground truth is actually not inside cnnp itself. Not to mention we want a way to extract accuracy from cnnp when evaluate against test set.

Stats are fine, we can have assertion enabled mode and assertion disabled mode which will be faster but no protection from abnormal stats. Accuracy seems to be something you need to track over time, therefore, the overhead need to be very low. I think the asynchronous execution nature on GPU really makes the debug process harder. Maybe we should call this debug mode, where we always copy out the tensor stats.

Another thing, is to backtrack and restart from a given epoch. We currently cannot do that because the checkpoint file gets consistently rewritten. We don't keep a journal of the checkpoints, thus, we cannot restart from a given checkpoint. This shouldn't be that hard, it just feels like something we can leverage SQLite, but it is not obvious how (SQLite supports WAL and MVCC, but that is for different use cases).

BTW, the ``ccv_resample`` method seems to be broken and can end up with nans. I need to dig into why (it seems from CUBIC, but I need more data).


2019-05-14
----------
Autotune implementation needs some work.

I didn't spend much time on autotune. It only surfaced this issue when I tries to implement the fp16 support. The original issue is from cudnn's ``cudnnGetConvolutionBackwardDataAlgorithm`` method. For fp16, this method will return a wrong preferred algorithm, thus, failed the following operation. The find method doesn't have this bug. That triggered me to look into why the ``cudnnFindConvolutionBackwardDataAlgorithmEx`` method is not called because it is part of the autotune process.

It turns out that there is a bug in the ``ccv_nnc_graph_autotune`` where given 0 sources and 0 destinations, it doesn't run the full graph. Then there is a bug in the convolution's autotune implementation where given 0 workspace size, it will skip the autotune completely. On top of that, we cannot really use the autotune as it is on the complete graph. The autotune process will run the command multiple times against different backends, therefore, if the command is not idempotent (it shouldn't), this will contaminant the final output.

I think the proper autotune implementation should allocate some inputs and outputs. When autotuning, copying the original inputs over. This can be repeated as much time as you would like. The only gotcha: there are some commands require inputs and outputs to be the same (enforce_inplace), that allocation need to handle this as well.

As of now, I workaround this problem by only autotune until backward finishes, and the autotune function avoid repeat too much times by identify there is only one backend. It is not as ideal.


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
