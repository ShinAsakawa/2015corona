from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T

x = T.vector('x')

def outf(x):
	return np.tanh(x*x)

outputs, updates = theano.scan(
	fn=outf,
	sequences=x,
	n_steps=x.shape[0]
)

op = theano.function(
    inputs=[x],
    outputs=[outputs],
    allow_input_downcast=True,
)

out = op(np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5]))

print(out)
