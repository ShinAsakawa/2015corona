from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T

N = T.iscalar('N')

def calc(n, fn1, fn2):
    return fn1 + fn2, fn1

outputs, _ = theano.scan(
    fn=calc,
    sequences=T.arange(N),
    n_steps=N,
    outputs_info=[1., 1.]
)

fibonacci = theano.function(
    inputs=[N],
    outputs=outputs
)

print(fibonacci(8))
