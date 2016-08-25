from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

X = 0.2 * np.random.randn(200) + np.sin(np.linspace(0, 4*np.pi, 200))
plt.plot(X)
plt.title('raw data')
plt.show()

factor = T.scalar('f')
sequence = T.vector('seq')

def smooth(x, last, factor):
    return (1-factor)*x + factor * last

outputs, _ = theano.scan(
    fn=smooth,
    sequences=sequence,
    n_steps=sequence.shape[0],
    outputs_info=[np.float64(0)],
    non_sequences=[factor],
)

linf = theano.function(
    inputs=[sequence, factor],
    outputs=outputs,
    allow_input_downcast=True,
)

Y = linf(X, 0.99)
plt.plot(Y)
plt.title('smoothed')
plt.show()


