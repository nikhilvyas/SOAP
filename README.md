# SOAP

This is the official (preliminary) implementation of the SOAP optimizer from [SOAP: Improving and Stabilizing Shampoo using Adam](https://arxiv.org/abs/2409.11321). To use, copy the soap.py file to your codebase and use SOAP optimizer in the following fashion:

```
from soap import SOAP

optim = SOAP(lr = 3e-3, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)
```

We recommend trying it with as large batch size as possible, as expected from second order optimizers, the benefits are larger at larger batch sizes.

While in the paper our experiments are restricted to Transformers which only have 2D layers, the code supports nD layers. If you are using the optimizer for (n > 2) nD layers please see additional hyperparameters in soap.py.


We will release an improved version of the optimizer with support for lower precision and distributed training. 


Haydn Jones has implemented a JAX version at https://github.com/haydn-jones/SOAP_JAX, though we have not yet verified the implementation.

## Complex parameters

SOAP supports `complex64` and `complex128` by optimizing the corresponding
`torch.view_as_real` tensor, with moments and preconditioners in the matching
real precision. No separate optimizer configuration is required.

All options apply to that real tensor, including its trailing size-2 axis.
This affects dimension merging, preconditioning, normalization, and
`channels_last` handling; a complex vector is treated as a real matrix.
This is real-coordinate SOAP, not a unitary-invariant complex optimizer.

Resume checkpoints with the same complex precision and `data_format`.
Mismatched checkpoint precision raises `ValueError`; `data_format` is not
stored in the optimizer state dictionary.
