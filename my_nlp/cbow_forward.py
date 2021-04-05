import sys
sys.path.append("..")
import numpy as np
from my_nlp.ch1.common.layers import MatMul

# Sample context data
c0 = np.array([1, 0, 0, 0, 0, 0, 0])
c1 = np.array([0, 0, 1, 0, 0, 0, 0])

# Initialize weights
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

# Create layers
in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

# Forward prop
mid = 0.5 * (in_layer0.forward(c0) + in_layer1.forward(c1))
out = out_layer.forward(mid)
print(out)

