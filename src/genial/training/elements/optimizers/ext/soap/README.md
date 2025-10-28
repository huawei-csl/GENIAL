<!-- # Taken From https://github.com/nikhilvyas/SOAP 

MIT License

Copyright (c) 2024 Nikhil Vyas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. -->

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