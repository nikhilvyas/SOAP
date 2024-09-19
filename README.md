# SOAP

This is the official (preliminary) implementation of the SOAP optimizer from [SOAP: Improving and Stabilizing Shampoo using Adam](https://arxiv.org/abs/2409.11321). To use, copy the soap.py file to your codebase and use SOAP optimizer in the following fashion:

```
from soap import SOAP

optim = SOAP(lr = 3e-3, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)
```

We recommend trying it with as large batch size as possible, as expected from second order optimizers, the benefits are larger at larger batch sizes.

While in the paper our experiments are restricted to Transformers which only have 2D layers, the code supports nD layers. If you are using the optimizer for (n > 2) nD layers please see additional hyperparameters in soap.py.


We will release an improved version of the optimizer with support for lower precision and distributed training. 
