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
